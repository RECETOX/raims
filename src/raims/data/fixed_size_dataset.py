import os
from typing import Optional, Tuple

import numpy
import torch
from numpy.random import SeedSequence
from pytorch_lightning import LightningDataModule
from spec2vec import SpectrumDocument
from torch.utils.data import Dataset, DataLoader

from .util import load_msp_documents


def _spectrum_binning(spectrum: SpectrumDocument, size: int = 1001) -> numpy.ndarray:
    """
    From a list of peak positions and its intensities produce a vector of intensities, i.e. create a vector `v` so as
    `v[int(peak.mz)] = peak.intensity`.

    :param spectrum: a spectrum to bin
    :param size: the maximal m/z value to consider and also the length of the produced vector
    :returns: numpy array with binned spectrum
    """
    masses = numpy.asarray(spectrum.peaks.mz, dtype=numpy.int32)
    intensities = numpy.asarray(spectrum.peaks.intensities, dtype=numpy.float32)

    mask = masses < size
    masses, intensities = masses[mask], intensities[mask]

    result = numpy.zeros(size, dtype=numpy.float32)
    for mass, intensity in zip(masses, intensities):
        result[mass] = max(result[mass], intensity)
    return result


def _get_histogram_size(spectrum: SpectrumDocument, cumulative_level: float = 0.95) -> numpy.ndarray:
    """
    :param spectrum:
    :param cumulative_level:
    :returns:
    """
    sorted_intensities = numpy.sort(spectrum.peaks.intensities)
    normalized_intensities = sorted_intensities / numpy.sum(sorted_intensities)
    return numpy.argmax(numpy.cumsum(normalized_intensities[::-1]) > cumulative_level)


class FixedSizedDataset(Dataset):
    def __init__(self, filename: str, prob: float = 0.2, max_mz: int = 1001, cumulative_level: float = 0.95,
                 seed: Optional[SeedSequence] = None):
        self.spectrum_documents = load_msp_documents(filename)
        self.prob = prob
        self.max_mz = max_mz
        self.cumulative_level = cumulative_level
        self.rng = numpy.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.spectrum_documents)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spectrum_document = self.spectrum_documents[index]
        binned_spectrum = _spectrum_binning(spectrum_document, self.max_mz)

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
                 batch_size: int = 1024, seed: Optional[SeedSequence] = None, num_workers: int = 8):
        super().__init__()

        seed = SeedSequence() if seed is None else seed
        child_seeds = seed.spawn(3)

        self.prob = prob
        self.max_mz = max_mz
        self.cumulative_level = cumulative_level
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = self._create_dataset(os.path.join(path, 'train.msp'), seed=child_seeds[0])
        self.test_dataset = self._create_dataset(os.path.join(path, 'test.msp'), seed=child_seeds[1])
        self.val_dataset = self._create_dataset(os.path.join(path, 'val.msp'), seed=child_seeds[2])

    def _create_dataset(self, path: str, seed: Optional[SeedSequence]) -> Dataset:
        return FixedSizedDataset(filename=path, seed=seed, prob=self.prob, max_mz=self.max_mz,
                                 cumulative_level=self.cumulative_level)

    def _create_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def train_dataloader(self) -> DataLoader:
        return self._create_loader(self.train_dataset, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return self._create_loader(self.test_dataset, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return self._create_loader(self.val_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
