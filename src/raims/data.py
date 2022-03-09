import os
from typing import Tuple, Optional, Callable, Dict, Generator

import matchms.filtering
import matchms.importing
import numpy
import torch
import torch.nn.functional
from matchms import Spectrum
from numpy.random import SeedSequence
from pytorch_lightning import LightningDataModule
from spec2vec import SpectrumDocument
from torch.utils.data import Dataset, DataLoader


def _process_spectrum(spectrum: Optional[Spectrum], n_required_peaks: Optional[int] = 10,
                      n_max_peaks: Optional[int] = None, min_relative_intensity: Optional[int] = None) -> Optional[
    Spectrum]:
    spectrum = matchms.filtering.select_by_mz(spectrum, mz_from=0, mz_to=1000)
    spectrum = matchms.filtering.normalize_intensities(spectrum)

    if n_required_peaks is not None:
        spectrum = matchms.filtering.require_minimum_number_of_peaks(spectrum, n_required=n_required_peaks)
    if n_max_peaks is not None:
        spectrum = matchms.filtering.reduce_to_number_of_peaks(spectrum, n_max=n_max_peaks)
    if min_relative_intensity is not None:
        spectrum = matchms.filtering.select_by_relative_intensity(spectrum, intensity_from=min_relative_intensity)
    return spectrum


def _load_msp_documents(filename: str, n_max_peaks: Optional[int] = None) -> Generator[SpectrumDocument, None, None]:
    spectra = (_process_spectrum(spectrum, n_max_peaks=n_max_peaks) for spectrum in
               matchms.importing.load_from_msp(filename))
    spectra = (SpectrumDocument(spectrum, n_decimals=0) for spectrum in spectra if spectrum is not None)
    return spectra


def _spectrum_binning(spectrum: SpectrumDocument, size: int = 1001):
    result = numpy.zeros(size, dtype=numpy.float32)
    result[numpy.asarray(spectrum.peaks.mz, dtype=int)] = numpy.asarray(spectrum.peaks.intensities)
    return result


def _get_histogram_size(spectrum: SpectrumDocument, cumulative_level: float = 0.95):
    sorted_intensities = numpy.sort(spectrum.peaks.intensities)
    normalized_intensities = sorted_intensities / numpy.sum(sorted_intensities)
    return numpy.argmax(numpy.cumsum(normalized_intensities[::-1]) > cumulative_level)


class HuggingfaceDataset(Dataset):
    def __init__(self, filename: str, vocabulary: Dict, max_length: int = 256, include_intensity: bool = False,
                 quadratic_bins: bool = False):
        super().__init__()

        self.spectrum_documents = list(_load_msp_documents(filename))
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

    def encode_peak(self, peak) -> torch.Tensor:
        return torch.tensor(self.vocabulary[peak], dtype=torch.int)

    def encode_spectrum(self, spectrum) -> torch.Tensor:
        encoded_peaks = [self.encode_peak(peak) for peak in spectrum if peak in self.vocabulary]
        return torch.tensor(encoded_peaks, dtype=torch.int)


class GenerativeDataset(Dataset):
    def __init__(self, filename: str, vocabulary: Dict, onehot: bool = True, include_intensity: bool = True):
        super().__init__()

        self.spectrum_documents = list(_load_msp_documents(filename))
        self.vocabulary = vocabulary
        self.onehot = onehot
        self.include_intensity = include_intensity

    def __len__(self) -> int:
        return len(self.spectrum_documents)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = numpy.argsort(self.spectrum_documents[index].peaks.intensities)[::-1]
        indices = [i for i in indices if self.spectrum_documents[index].words[i] in self.vocabulary]

        x = self.encode_spectrum_document_with_intensities(self.spectrum_documents[index], indices[:-1])
        y = self.encode_spectrum_document(self.spectrum_documents[index], indices[1:])

        return x, y

    def size(self) -> int:
        return len(self.vocabulary)

    def encode_peak(self, word: int) -> torch.Tensor:
        return torch.tensor(self.vocabulary[word], dtype=torch.int)

    def encode_peak_onehot(self, word: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(self.encode_peak(word), num_classes=len(self.vocabulary))

    def encode_spectrum_document(self, document: SpectrumDocument, indices) -> torch.Tensor:
        encoding: Callable[[int], torch.Tensor] = self.encode_peak_onehot if self.onehot else self.encode_peak
        return torch.stack([encoding(word) for word in document.words[indices]])

    def encode_spectrum_document_with_intensities(self, document: SpectrumDocument, indices) -> torch.Tensor:
        intensities = torch.as_tensor(document.peaks.intensities[indices]).reshape((-1, 1))
        encoded_spectrum = self.encode_spectrum_document(document, indices)
        return torch.cat((encoded_spectrum, intensities), dim=1)


class FixedSizedDataset(Dataset):
    def __init__(self, filename: str, prob: float = 0.2, max_mz: int = 1001, cumulative_level: float = 0.95,
                 seed: Optional[SeedSequence] = None):
        self.spectrum_documents = list(_load_msp_documents(filename))
        self.prob = prob
        self.max_mz = max_mz
        self.cumulative_level = cumulative_level
        self.rng = numpy.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.spectrum_documents)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spectrum_document = self.spectrum_documents[index]
        binned_spectrum = _spectrum_binning(spectrum_document, self.size)

        missing_mask = self.rng.uniform(0, 1, self.max_mz) < self.prob

        histogram_size = _get_histogram_size(spectrum_document, self.cumulative_level)
        histogram_indices = numpy.argpartition(binned_spectrum, -histogram_size)[-histogram_size:]
        histogram_mask = numpy.zeros_like(binned_spectrum, dtype=bool)
        histogram_mask[histogram_indices] = True

        x = numpy.where(missing_mask, 0, binned_spectrum)
        y = numpy.where(missing_mask & histogram_mask, 1, 0)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class FixedSizedDataModule(LightningDataModule):
    def __init__(self, path: str, prob: float = 0.2, max_mz: int = 1001, cumulative_level: float = 0.95,
                 batch_size: int = 1024, shuffle: bool = True, seed: Optional[SeedSequence] = None,
                 num_workers: int = 8):
        super().__init__()

        if seed is None:
            seed = numpy.random.SeedSequence()
        child_seeds = seed.spawn(3)

        self.prob = prob
        self.max_mz = max_mz
        self.cumulative_level = cumulative_level
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.train_dataset = self._create_dataset(os.path.join(path, 'train.msp'), seed=child_seeds[0])
        self.test_dataset = self._create_dataset(os.path.join(path, 'test.msp'), seed=child_seeds[1])
        self.val_dataset = self._create_dataset(os.path.join(path, 'val.msp'), seed=child_seeds[2])

    def _create_dataset(self, path: str, seed: Optional[SeedSequence]) -> Dataset:
        return FixedSizedDataset(filename=path, seed=seed, prob=self.prob, max_mz=self.max_mz,
                                 cumulative_level=self.cumulative_level)

    def _create_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def train_dataloader(self) -> DataLoader:
        return self._create_loader(self.train_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._create_loader(self.test_dataset)

    def val_dataloader(self) -> DataLoader:
        return self._create_loader(self.val_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._create_loader(self.test_dataset)


class GenerativeDataModule(LightningDataModule):
    def __init__(self, path, vocabulary, batch_size: int = 256, num_workers: int = 8):
        super().__init__()

        self.train_dataset = GenerativeDataset(os.path.join(path, 'train.msp'), vocabulary)
        self.test_dataset = GenerativeDataset(os.path.join(path, 'test.msp'), vocabulary)
        self.val_dataset = GenerativeDataset(os.path.join(path, 'val.msp'), vocabulary)

        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def _collate(batch):
        padded = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
        target = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True, padding_value=-100)
        return [padded, torch.LongTensor(target)]

    def _create_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=GenerativeDataModule._collate)

    def train_dataloader(self) -> DataLoader:
        return self._create_loader(self.train_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._create_loader(self.test_dataset)

    def val_dataloader(self) -> DataLoader:
        return self._create_loader(self.val_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._create_loader(self.test_dataset)
