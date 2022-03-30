from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor


class BaseLSTM(LightningModule):
    def __init__(self, input_size: int, hidden_size: int, include_intensity: bool, learning_rate: float):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size + int(include_intensity), hidden_size=hidden_size, num_layers=2,
                            batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.include_intensity = include_intensity
        self.learning_rate = learning_rate
        self.return_sequence = True

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, peaks: Tensor, intensities: Optional[Tensor] = None, init: Optional[Tensor] = None):
        if self.include_intensity:
            peaks = torch.cat((peaks, intensities[:, :, None]), dim=2)

        output, (h_n, c_n) = self.lstm(peaks, init)
        output = output if self.return_sequence else h_n[-1]

        output = self.linear(output)
        output = self.log_softmax(output)

        return output, (h_n, c_n)

    def _step(self, batch) -> float:
        (peaks, intensities), y = batch
        y_hat, _ = self(peaks, intensities)
        return F.nll_loss(y_hat, y)

    def training_step(self, batch: Any, batch_idx: int) -> float:
        loss = self._step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss = self._step(batch)
        self.log('val_loss', loss)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss = self._step(batch)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class PureLSTM(BaseLSTM):
    """
    Generative model composed of one-hot encoding layer, LSTM module, and LogSoftmax layer. The model can optionally
    take spectral intensity into consideration.
    """
    def __init__(self, num_classes: int, hidden_size: int, include_intensity: bool, learning_rate: float):
        super().__init__(input_size=num_classes, hidden_size=hidden_size, include_intensity=include_intensity,
                         learning_rate=learning_rate)
        self.num_classes = num_classes

    def forward(self, peaks: Tensor, intensities: Optional[Tensor] = None, init: Optional[Tensor] = None):
        super().forward(F.one_hot(peaks, num_classes=self.num_classes), intensities, init)


class EmbeddingLSTM(BaseLSTM):
    """
    Generative model composed of embedding layer, LSTM module, and LogSoftmax layer. The model can optionally
    take spectral intensity into consideration.
    """
    def __init__(self, embeddings: Tensor, hidden_size: int, freeze_embeddings: bool, include_intensity: bool,
                 learning_rate: float):
        super().__init__(input_size=embeddings.shape[1], hidden_size=hidden_size, include_intensity=include_intensity,
                         learning_rate=learning_rate)
        self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=freeze_embeddings)

    def forward(self, peaks: Tensor, intensities: Optional[Tensor] = None, init: Optional[Tensor] = None):
        super().forward(self.embedding(peaks), intensities, init)
