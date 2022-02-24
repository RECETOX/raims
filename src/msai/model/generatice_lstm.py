import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule


class PureLSTM(LightningModule):
    def __init__(self, size, hidden_size, add_intensities=False):
        super().__init__()

        input_size = size + 1 if add_intensities else size
        output_size = size

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, seq, return_sequence=True, init_state=None):
        output, (h_n, c_n) = self.lstm(seq, init_state)
        lstm_y = output if return_sequence else h_n[-1:]
        result = [self.log_softmax(self.linear(h)) for h in lstm_y]
        return torch.stack(result), (h_n, c_n)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # TODO: discuss with Michal modes od LSTM modules
        y_hat, _ = self(x)
        loss = F.nll_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class EmbeddingLSTM(LightningModule):
    def __init__(self, hidden_size, embeddings: torch.Tensor, freeze_embeddings: bool = True):
        super().__init__()

        self.in_features = self.embedding.embedding_dim

        self.embedding = torch.nn.Embedding.from_pretrained(embeddings, freeze=freeze_embeddings)
        self.lstm = torch.nn.LSTM(input_size=self.in_features, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, size)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, seq, init_state=None):
        embedded_seq = self.embedding(seq)
        output, (h_n, c_n) = self.lstm(embedded_seq, init_state)
        return self.softmax(self.linear(output)), (h_n, c_n)


class MostFrequentDummy(LightningModule):
    pass