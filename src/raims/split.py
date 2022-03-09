from typing import List, Optional

import numpy as np
from matchms import Spectrum
from numpy.random import SeedSequence


def random_split(data: List[Spectrum], frac: List[float], seed: Optional[SeedSequence] = None) -> List[List[Spectrum]]:
    data = np.asarray(data)
    frac = np.asarray(frac)

    if not np.allclose(sum(frac), 1):
        raise ValueError('frac should sum to 1')

    n = len(data)
    indices = np.arange(n, dtype=np.int64)
    np.random.default_rng(seed).shuffle(indices)

    bounds = np.concatenate([[0], np.cumsum(frac) * n]).astype(np.int64)
    splits = [indices[beg:end] for beg, end in zip(bounds[:-1], bounds[1:])]

    return [list(data[indices]) for indices in splits]
