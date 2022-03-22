import os
from typing import Dict, List, Tuple

import numpy
import torch
import torch.nn.functional
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from .util import load_msp_documents


class GenerativeDataset(Dataset):
    def __init__(self, filename: str, vocabulary: Dict[str, int]):
        super().__init__()

        self.spectrum_documents = load_msp_documents(filename=filename)
        self.vocabulary = vocabulary

    def __len__(self) -> int:
        return len(self.spectrum_documents)

    def __getitem__(self, item: int) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        document = self.spectrum_documents[item]

        indices = numpy.argsort(document.peaks.intensities)[::-1]
        indices = numpy.asarray([i for i in indices if document.words[i] in self.vocabulary])

        peaks = torch.tensor([self.vocabulary[w] for w in document.words[indices]], dtype=torch.int)
        intensities = torch.tensor(document.peaks.intensities[indices], dtype=torch.float32)

        return (peaks[:-1], intensities[:-1]), peaks[1:]


class GenerativeDataModule(LightningDataModule):
    def __init__(self, path: str, vocabulary: Dict[str, int], onehot: bool, intensity: bool, batch_size: int,
                 n_workers: int):
        super().__init__()

        self.vocabulary = vocabulary
        self.onehot = onehot
        self.intensity = intensity
        self.batch_size = batch_size
        self.n_workers = n_workers

        self.train_dataset = self._create_dataset(path, 'train.msp')
        self.test_dataset = self._create_dataset(path, 'test.msp')
        self.val_dataset = self._create_dataset(path, 'val.msp')

    def _create_dataset(self, prefix: str, filename: str) -> GenerativeDataset:
        return GenerativeDataset(filename=os.path.join(prefix, filename), vocabulary=self.vocabulary)

    def _create_loader(self, dataset: Dataset, shuffle) -> DataLoader:
        return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=shuffle,
                          collate_fn=GenerativeDataModule._collate)

    def train_dataloader(self) -> DataLoader:
        return self._create_loader(dataset=self.train_dataset, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return self._create_loader(dataset=self.test_dataset, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return self._create_loader(dataset=self.val_dataset, shuffle=False)

    def predict_dataloader(self) -> List[DataLoader]:
        return [self.test_dataloader(), self.test_dataloader()]

    @staticmethod
    def _collate(batch):
        padded = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
        target = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True, padding_value=-100)
        return [padded, torch.LongTensor(target)]
