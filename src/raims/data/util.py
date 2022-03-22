from typing import Optional, List, Iterator, Dict, Tuple

import torch
from gensim.models import Word2Vec
from matchms.filtering import (select_by_mz, normalize_intensities, require_minimum_number_of_peaks,
                               reduce_to_number_of_peaks, select_by_relative_intensity)
from matchms import Spectrum
from matchms.importing import load_from_msp
from spec2vec import SpectrumDocument
from torch import Tensor


def process_spectra(spectra: Iterator[Spectrum], n_min_peaks: Optional[int] = 10, n_max_peaks: Optional[int] = None,
                    min_relative_intensity: Optional[int] = None) -> Iterator[Spectrum]:
    for spectrum in spectra:
        spectrum = normalize_intensities(spectrum)
        spectrum = select_by_mz(spectrum, mz_from=0, mz_to=1000)

        if n_min_peaks is not None:
            spectrum = require_minimum_number_of_peaks(spectrum, n_required=n_min_peaks)
        if n_max_peaks is not None:
            spectrum = reduce_to_number_of_peaks(spectrum, n_max=n_max_peaks)
        if min_relative_intensity is not None:
            spectrum = select_by_relative_intensity(spectrum, intensity_from=min_relative_intensity)

        yield spectrum


def load_msp_documents(filename: str) -> List[SpectrumDocument]:
    spectra = load_from_msp(filename=filename)
    spectra = process_spectra(spectra, n_min_peaks=10, n_max_peaks=None)
    return [SpectrumDocument(spec, n_decimals=0) for spec in spectra if spec is not None]


def load_word2vec(path: str) -> Tuple[Dict[str, int], Tensor]:
    model = Word2Vec.load(path)
    vocab = {e: i for i, e in enumerate(model.wv.index_to_key)}
    return vocab, torch.from_numpy(model.wv.vectors)
