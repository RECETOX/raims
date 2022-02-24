import numpy as np


def get_top_k_ind(arr, k):
    arg_sorted = np.argsort(arr)[::-1]

    top_k = arg_sorted[:k]
    top_after_k = arg_sorted[k]

    return top_k, top_after_k


def get_mean_nan(data2D):
    means = np.zeros(len(data2D))
    for i, row in enumerate(data2D):
        mean_ = row[~np.isnan(row)].mean()
        means[i] = mean_
    return means


def get_mz_vector(spec, max_mz=1001):
    mz_vect = np.zeros(shape=max_mz)
    for mz, intensity in zip(spec.peaks.mz, spec.peaks.intensities):
        mz_vect[int(mz)] = intensity
    return mz_vect


def get_his_size(spec, cum_level=0.95):
    descending = np.argsort(spec.peaks.intensities)[::-1]
    normalized = spec.peaks.intensities[descending] / np.sum(spec.peaks.intensities)
    n_peaks_considered = np.argmax(np.cumsum(normalized) > cum_level)

    return n_peaks_considered


def compact_to_mz(compact, max_mz=1001):
    mz_vect = np.zeros(shape=max_mz)
    for mz, intensity in compact:
        mz_vect[int(mz)] = intensity
    return mz_vect
