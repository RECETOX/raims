from abc import ABC
from collections import Counter

import torch
import torch.nn.functional

from pytorch_lightning import LightningModule


class BaseLstmModule(LightningModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _step(self, batch):
        x, y = batch
        y_hat, _ = self(x, return_sequence=False)
        return torch.nn.functional.nll_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_lost', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_lost', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class PureLSTM(LightningModule):
    def __init__(self, size, hidden_size, add_intensities=False):
        super().__init__()

        input_size = size + 1 if add_intensities else size
        output_size = size

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def _step(self, batch):
        x, y = batch
        # TODO: discuss with Michal modes od LSTM modules
        y_hat, _ = self(x, return_sequence=False)
        return torch.nn.functional.nll_loss(y_hat, y)

    def forward(self, seq, return_sequence, init_state=None):
        output, (h_n, c_n) = self.lstm(seq, init_state)
        lstm_y = output if return_sequence else h_n[-1:]
        result = [self.log_softmax(self.linear(h)) for h in lstm_y]
        return torch.stack(result), (h_n, c_n)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_lost', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_lost', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class EmbeddingLSTM(LightningModule):
    def __init__(self, size, hidden_size, embeddings: torch.Tensor, freeze_embeddings: bool = True,
                 include_intensity: bool = False):
        super().__init__()

        self.in_features = self.embedding.embedding_dim + int(include_intensity)
        self.include_intensity = include_intensity

        self.embedding = torch.nn.Embedding.from_pretrained(embeddings, freeze=freeze_embeddings)
        self.lstm = torch.nn.LSTM(input_size=self.in_features, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, size)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, seq, return_sequence, init_state=None):
        if self.include_intensity:
            indices, intensities = seq[:, :, 0].int(), seq[:, :, 1]
            embedded_seq = torch.cat([self.embedding(indices), intensities.reshape(*intensities.shape, 1)], dim=-1)
        else:
            embedded_seq = self.embedding(seq)

        output, (h_n, c_n) = self.lstm(embedded_seq, init_state)
        output = output if return_sequence else h_n[-1]

        return self.log_softmax(self.linear(output)), (h_n, c_n)

    def _step(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss)


class MostFrequentDummy(LightningModule):
    def __init__(self, ds):
        super().__init__()

        self.log_probabilities = MostFrequentDummy.compute_log_probabilities(ds)
        self.log_probabilities_per_k = MostFrequentDummy.compute_log_probabilities_per_k(ds)

    @staticmethod
    def get_occurrence(ds):
        return [int(str.split(peak, '@')[1]) for doc in ds.ref_docs for peak in doc.words]

    @staticmethod
    def get_occurrence_per_k(ds):
        max_len = max(len(peak.words) for peak in ds.ref_docs)
        k_all = [[1]] * (max_len * 2)
        for doc in ds.ref_docs:
            for i, peak in enumerate(doc.words):
                k_all[i].append(int(str.split(peak, '@')[1]))
        return k_all

    @staticmethod
    def compute_log_probabilities(ds):
        counter = Counter(MostFrequentDummy.get_occurence(ds))
        log_probabilities = torch.ones(len(ds.vocab))
        for d, n in counter.items():
            log_probabilities[ds.vocab[f'peak@{d}']] = n
        return torch.log(log_probabilities / log_probabilities.sum())

    @staticmethod
    def compute_log_probabilities_per_k(ds):
        counters = [Counter(k) for k in MostFrequentDummy.get_occurence_per_k(ds)]
        result = torch.ones(len(counters), len(ds.vocab))

        for counter in counters:
            pass
        return torch.log(result / result.sum(dim=1, keepdims=True))

    def forward(self, seq, return_sequence=True, init_state=0):
        state = init_state + len(seq[0])

        return torch.tensor(self.log_probabilities).repeat(len(seq), 1), state
