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
    """
    This function aggregates all the common spectra processing tasks for our experiments:
    1) intensity normalisation,
    2) select only peaks in range of 0-1000 m/z, discard the others,
    3) optionally discard the spectra with less than minimal number of peaks (defaults to 10),
    4) optionally reduce the number of peaks per spectrum,
    4) optionally discard some peaks of low relative intensity.

    :param spectra: sequence to spectra to process
    :param n_min_peaks: the minimum number of peaks a spectrum must contain (unfitting spectra are discarded)
    :param n_max_peaks: the maximum number of peaks to which each spectrum is reduced
    :param min_relative_intensity: discard peaks having lower relative intensity then the specified minimum limit
    :returns: sequence of processed spectra
    """
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


def load_msp_documents(filename: str, n_decimals: int=0) -> List[SpectrumDocument]:
    """
    Using Matchms library load MSP file, process it, and produce a list of SpectrumDocuments.

    The spectra in the MSP files are processed as follows:
    1) for each spectrum normalize the intensity,
    2) select only peaks in range of 0-1000 m/z,
    3) discard the spectra with less than 10 number of peaks.

    In SpectrumDocument every peak and loss position (m/z value) is encoded into the string "peak@100.32" and
    "loss@100.32" (for n_decimals=2). By default, we use n_decimals=0. For more info on SpectrumDocuments see spec2vec
    library.

    :param filename: a path to the file to load
    :param n_decimals: the number of decimal places to round the peak and loss positions.
    :returns: list of retrieved SpectrumDocuments
    """
    spectra = load_from_msp(filename=filename)
    spectra = process_spectra(spectra, n_min_peaks=10, n_max_peaks=None, min_relative_intensity=None)
    return [SpectrumDocument(spec, n_decimals=n_decimals) for spec in spectra if spec is not None]


def load_word2vec(path: str) -> Tuple[Dict[str, int], Tensor]:
    """
    Load trained Word2Vec model.

    :param path: path to the trained model
    :returns: vocabulary, embedding
    """
    model = Word2Vec.load(path)
    vocab = {e: i for i, e in enumerate(model.wv.index_to_key)}
    return vocab, torch.from_numpy(model.wv.vectors)
