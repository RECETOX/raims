import os.path
from typing import Tuple, Optional, Generator, Dict, Union, Callable, Any

import matchms.filtering
import matchms.importing
import numpy
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from spec2vec import SpectrumDocument
from torch import Tensor
from torch.utils.data import Dataset


def process_spectrum(spectrum: Optional[matchms.Spectrum], n_required_peaks: Optional[int] = 10,
                     n_max_peaks: Optional[int] = None, min_relative_intensity: Optional[int] = None) -> Optional[
    matchms.Spectrum]:
    spectrum = matchms.filtering.select_by_mz(spectrum, mz_from=0, mz_to=1000)
    spectrum = matchms.filtering.normalize_intensities(spectrum)

    if n_required_peaks is not None:
        spectrum = matchms.filtering.require_minimum_number_of_peaks(spectrum, n_required=n_required_peaks)
    if n_max_peaks is not None:
        spectrum = matchms.filtering.reduce_to_number_of_peaks(spectrum, n_max=n_max_peaks)
    if min_relative_intensity is not None:
        spectrum = matchms.filtering.select_by_relative_intensity(spectrum, intensity_from=min_relative_intensity)
    return spectrum


def load_msp_documents(filename: str, n_max_peaks: Optional[int] = None) -> Generator[SpectrumDocument, None, None]:
    spectra = (process_spectrum(spectrum, n_max_peaks=n_max_peaks) for spectrum in
               matchms.importing.load_from_msp(filename))
    spectra = (SpectrumDocument(spectrum, n_decimals=0) for spectrum in spectra if spectrum is not None)
    return spectra


class HuggingfaceDataset(Dataset):
    def __init__(self, filename: str, vocabulary: Dict, max_length: int = 256, include_intensity: bool = False,
                 quadratic_bins: bool = False):
        super().__init__()

        self.spectrum_documents = list(load_msp_documents(filename))
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.include_intensity = include_intensity

        if quadratic_bins:
            self.bins = ((numpy.arange(max_length) ** 2) / ((max_length - 1) ** 2))[::-1]
        else:
            self.bins = (numpy.arange(max_length) / (max_length - 1))[::-1]

    def __len__(self) -> int:
        return len(self.spectrum_documents)

    def __getitem__(self, index) -> Dict[str, Tensor]:
        document = self.spectrum_documents[index]
        indices = torch.argsort(document.peaks.intensities)[::-1]
        indices = indices[:self.max_length]

        x = self.encode_spectrum(torch.asarray(document.words)[indices])
        x_padded = torch.cat([x, torch.zeros(self.max_length - len(x), dtype=torch.int)])

        attention_mask = torch.zeros(self.max_length, dtype=torch.int)
        attention_mask[:len(x)] = 1

        result = {"input_ids": x_padded, "attention_mask": attention_mask}

        if self.include_intensity:
            position_ids = torch.asarray(numpy.digitize(document.peaks.intensities[indices], self.bins, right=False))
            padding = torch.zeros(self.max_length - len(position_ids), dtype=torch.int) + self.max_length - 1
            result["position_ids"] = torch.cat([position_ids, padding])
        return result

    def size(self) -> int:
        return len(self.vocabulary)

    def encode_peak(self, peak) -> Tensor:
        return torch.tensor(self.vocabulary[peak], dtype=torch.int)

    def encode_spectrum(self, spectrum) -> Tensor:
        encoded_peaks = [self.encode_peak(peak) for peak in spectrum if peak in self.vocabulary]
        return torch.tensor(encoded_peaks, dtype=torch.int)


class GenerativeDataset(Dataset):
    def __init__(self, filename: str, vocabulary: Dict, onehot: bool = True, include_intensity: bool = True):
        super().__init__()

        self.spectrum_documents = list(load_msp_documents(filename))
        self.vocabulary = vocabulary
        self.onehot = onehot
        self.include_intensity = include_intensity

    def __len__(self) -> int:
        return len(self.spectrum_documents)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        indices = numpy.argsort(self.spectrum_documents[index].peaks.intensities)[::-1]
        indices = [i for i in indices if self.spectrum_documents[index].words[i] in self.vocabulary]

        x = self.encode_spectrum_document_with_intensities(self.spectrum_documents[index], indices[:-1])
        y = self.encode_spectrum_document(self.spectrum_documents[index], indices[1:])

        return x, y

    def size(self) -> int:
        return len(self.vocabulary)

    def encode_peak(self, word: int) -> Tensor:
        return torch.tensor(self.vocabulary[word], dtype=torch.int)

    def encode_peak_onehot(self, word: int) -> Tensor:
        return F.one_hot(self.encode_peak(word), num_classes=len(self.vocabulary))

    def encode_spectrum_document(self, document: SpectrumDocument, indices) -> Tensor:
        encoding: Callable[[int], Tensor] = self.encode_peak_onehot if self.onehot else self.encode_peak
        return torch.stack([encoding(word) for word in document.words[indices]])

    def encode_spectrum_document_with_intensities(self, document: SpectrumDocument, indices) -> Tensor:
        intensities = torch.as_tensor(document.peaks.intensities[indices]).reshape((-1, 1))
        encoded_spectrum = self.encode_spectrum_document(document, indices)
        return torch.cat((encoded_spectrum, intensities), dim=1)


class FixedSizedDataset(Dataset):
    def __init__(self, filename: str):
        self.spectrum_documents = list(load_msp_documents(filename))

    def __len__(self) -> int:
        return len(self.spectrum_documents)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        vector = spectrum_to_vector(self.spectrum_documents[index], self.size)



def spectrum_to_vector(spectrum: SpectrumDocument, size: int = 1001):
    result = numpy.zeros(size, dtype=numpy.float)
    result[numpy.asarray(spectrum.peaks.mz, dtype=int)] = numpy.asarray(spectrum.peaks.intensities)
    return result

from pytorch_lightning import LightningDataModule

class BaseDataModule(LightningDataModule):
    def __init__(self, path, constructor):
        super().__init__()
        self.train_dataset = constructor(os.path.join(path, 'train.msp'))
        self.val_dataset = constructor(os.path.join(path, 'val.msp'))
        self.test_dataset = constructor(os.path.join(path, 'test.msp'))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset)


class FixedSizedDataModule(BaseDataModule):
    def __init__(self, path):
        super().__init__(path, FixedSizedDataset)


class GenerativeDataModule(BaseDataModule):
    def __init__(self, path):
        super().__init__(path, GenerativeDataset)
