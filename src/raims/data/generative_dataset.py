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

#    def __getitem__(self, item: int) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        document = self.spectrum_documents[item]

#        print(f"item {item}\ndocument {document}\n")
#        print(f"peaks {document.peaks}\n")
#        print(f"intensities {document.peaks.intensities}\n")
        indices = numpy.argsort(document.peaks.intensities)[::-1]
#        print(f"indices1: {indices}\n")
        indices = numpy.asarray([i for i in indices if document.words[i] in self.vocabulary])

#        print(f"indices2: {indices}\n")
#        print(f"words: {document.words}\n")

        # peaks = torch.tensor([self.vocabulary[w] for w in document.words[indices]], dtype=torch.int)
        peaks = torch.tensor([self.vocabulary[document.words[i]] for i in indices], dtype=torch.int)
#        print(f"peaks2: {peaks}\n")
        # intensities = torch.tensor(document.peaks.intensities[indices], dtype=torch.float32)
        intensities = torch.tensor([document.peaks.intensities[i] for i in indices], dtype=torch.float32)
#        print(f"intensities: {intensities}\n")

# ma vracet o jednu posunute
        return (peaks[:-1], intensities[:-1]), peaks[1:]
#        return (peaks[:-1], peaks[1:])


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
        peaks_in = torch.nn.utils.rnn.pad_sequence([item[0][0] for item in batch], batch_first=True, padding_value=0)
        intensities_in = torch.nn.utils.rnn.pad_sequence([item[0][1] for item in batch], batch_first=True, padding_value=0)
# XXX: -100 Jirka, nejde pak onehot
        peaks_out = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True, padding_value=-100)
        return ((peaks_in,intensities_in),peaks_out)
#        return [padded, target]
