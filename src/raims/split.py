import os
from collections import defaultdict
from typing import List, Optional, Dict, Tuple

import numpy as np
from matchms import Spectrum
from matchms.exporting import save_as_msp
from numpy.random import SeedSequence


def inchikey_mapping(spectra: List[Spectrum], key: str = 'inchikey') -> Dict[str, List[Spectrum]]:
    """
    Divide the provided list of spectra into a dictionary of buckets with the same InChIKey.

    :param spectra: a list of spectra
    :param key: a key into the spectrum metadata under witch to find the InChiKey
    :returns: InChIKey mapping, a dictionary of buckets of spectra with the same InChiKey
    """
    keys = [spectrum.metadata.get(key) for spectrum in spectra]
    keys = [str.split(key, '-')[0] if key is not None and key != 'nan' else None for key in keys]

    n_dropping = sum(k is None for k in keys)
    print(f'Dropping {n_dropping} entries out of {len(spectra)} records due to missing InChIKey')

    mapping = defaultdict(list)
    for key, spectrum in zip(keys, spectra):
        if key is not None:
            mapping[key].append(spectrum)
    return mapping


def random_split(dataset: List, partitions: List[float], seed: Optional[SeedSequence] = None) -> List[List]:
    """
    Perform a random split of the dataset into partitions of a fractional size.

    :param dataset: the dataset to partition
    :param partitions: list of fractions into which to divide the original dataset (the list of fractions must sum to 1)
    :param seed: optional parameter to fix the entropy
    :returns: the partitioned dataset
    """
    dataset = np.asarray(dataset)
    partitions = np.asarray(partitions)

    if not np.allclose(sum(partitions), 1):
        raise ValueError('frac should sum to 1')

    n = len(dataset)
    indices = np.arange(n, dtype=np.int64)
    np.random.default_rng(seed).shuffle(indices)

    bounds = np.concatenate([[0], np.cumsum(partitions) * n]).astype(np.int64)
    splits = [indices[beg:end] for beg, end in zip(bounds[:-1], bounds[1:])]

    return [list(dataset[indices]) for indices in splits]


def split_a(mapping: Dict[str, List], seed: Optional[SeedSequence] = None) -> Tuple[List, List, List]:
    """
    Perform an A-split of the mapping to train, test, validation partitions.

    Each first occurrence of each unique key, e.g. InChIKey, is put into the training partition. The remaining keys are
    randomly split into testing and validation partitions so as these two partitions does not share a single key.

    :param mapping: a dataset divided into buckets of samples with the same key
    :param seed: optional parameter to fix the entropy
    :returns: training, testing, validation datasets
    """
    heads = list()
    tails = defaultdict(list)

    for key, spectra in mapping.items():
        heads.append(spectra[0])
        tails[key].extend(spectra[1:])

    tail_keys = [key for key, items in tails.items() if items]
    test_keys, val_keys = random_split(dataset=tail_keys, partitions=[.5, .5], seed=seed)

    train_partition = heads
    test_partition = [item for key in test_keys for item in tails[key]]
    val_partition = [item for key in val_keys for item in tails[key]]

    return train_partition, test_partition, val_partition


def split_b(mapping: Dict[str, List], partitions: List[float], seed: Optional[SeedSequence] = None) -> List[List]:
    """
    Perform a B-split of the mapping to multiple partitions.

    Take a random bucket (a collection of the same keys) and put it into a random non-full partition.

    :param mapping: a dataset divided into buckets of samples with the same key
    :param partitions: list of fractions into which to divide the original dataset (the list of fractions must sum to 1)
    :param seed: optional parameter to fix the entropy
    :returns: the partitioned dataset
    """
    mapping_keys = list(mapping.keys())
    partitioned_keys = random_split(dataset=mapping_keys, partitions=partitions, seed=seed)

    return [[item for key in partition for item in mapping[key]] for partition in partitioned_keys]


def save_partitions_as_msp(folder: str, filenames: List[str], partitions: List[List[Spectrum]]) -> None:
    """
    Take a list of partitions and a list of filenames and save each partition into a msp file.

    :param folder: The path to the folder where to save the partitions
    :param filenames: filenames of partitions
    :param partitions: partitions of spectra
    """
    if len(filenames) != len(partitions):
        raise ValueError('The number of given filenames does not match the number of provided partitions')

    if not os.path.exists(folder):
        os.makedirs(folder)

    for filename, partition in zip(filenames, partitions):
        save_as_msp(spectra=partition, filename=os.path.join(folder, filename))
