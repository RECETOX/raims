import os
from typing import Dict, List

import numpy
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from .util import load_msp_documents


class HuggingfaceDataset(Dataset):
    def __init__(self, filename: str, vocabulary: Dict[str, int], max_length: int = 256, include_intensity: bool = False,
                 quadratic_bins: bool = False):
        super().__init__()

        self.spectrum_documents = load_msp_documents(filename)
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.include_intensity = include_intensity

        if quadratic_bins:
            self.bins = ((numpy.arange(max_length) ** 2) / ((max_length - 1) ** 2))[::-1]
        else:
            self.bins = (numpy.arange(max_length) / (max_length - 1))[::-1]

    def __len__(self) -> int:
        return len(self.spectrum_documents)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        document = self.spectrum_documents[index]
        indices = numpy.argsort(document.peaks.intensities)[::-1]
        indices = indices[:self.max_length]

#        print("document",document,"\n\n\nindices:",indices,"\n\n\n")

#        x = self.encode_spectrum(torch.as_tensor(document.words)[indices])
        x = torch.tensor([self.vocabulary[document.words[i]] for i in indices if document.words[i] in self.vocabulary], dtype=torch.long)
        x_padded = torch.cat([x, torch.zeros(self.max_length - len(x), dtype=torch.long)])

        attention_mask = torch.zeros(self.max_length, dtype=torch.int)
        attention_mask[:len(x)] = 1

        result = {"input_ids": x_padded[:-1], "attention_mask": attention_mask[:-1], "labels": x_padded[1:]}

        if self.include_intensity:
            position_ids = torch.as_tensor(numpy.digitize(document.peaks.intensities[indices], self.bins, right=False))
            padding = torch.zeros(self.max_length - len(position_ids), dtype=torch.int) + self.max_length # - 1
            result["position_ids"] = torch.cat([position_ids[:-1], padding])
        return result

    def size(self) -> int:
        return len(self.vocabulary)

#    def encode_peak(self, peak) -> torch.Tensor:
#        return torch.tensor(self.vocabulary[peak], dtype=torch.int)
#
#    def encode_spectrum(self, spectrum) -> torch.Tensor:
#        encoded_peaks = [self.encode_peak(peak) for peak in spectrum if peak in self.vocabulary]
#        return torch.tensor(encoded_peaks, dtype=torch.int)


class HuggingfaceDataModule(LightningDataModule):
    def __init__(self, path: str, vocabulary: Dict[str, int], max_length: int = 256, include_intensity: bool = False,
                 quadratic_bins: bool = False, batch_size: int = 64, num_workers: int = 8):
        super().__init__()

        self.vocabulary = vocabulary
        self.max_length = max_length
        self.include_intensity = include_intensity
        self.quadratic_bins = quadratic_bins
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = self._create_dataset(filename=os.path.join(path, 'train.msp'))
        self.test_dataset = self._create_dataset(filename=os.path.join(path, 'test.msp'))
        self.val_dataset = self._create_dataset(filename=os.path.join(path, 'val.msp'))

    def _create_dataset(self, filename: str) -> HuggingfaceDataset:
        return HuggingfaceDataset(filename=filename, vocabulary=self.vocabulary, max_length=self.max_length,
                                  include_intensity=self.include_intensity, quadratic_bins=self.quadratic_bins)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> List[DataLoader]:
        return [self.train_dataloader(), self.val_dataloader()]
