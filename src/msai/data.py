from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.filtering import select_by_relative_intensity, reduce_to_number_of_peaks
from torch.utils.data import Dataset

from .utils import get_top_k_ind, get_mz_vector, get_his_size


def get_n_samples(spectra, n, seed=42):
    res = []
    nums = np.ones(len(spectra))
    nums[n:] = 0
    np.random.seed(seed)
    np.random.shuffle(nums)
    for i, sp in enumerate(spectra):
        if nums[i] == 1:
            res.append(sp)
    return res


def spectrum_processing(s, min_rel_int=None, n_required_peaks=10, max_peaks=None):
    # s = default_filters(s)
    s = select_by_mz(s, mz_from=0, mz_to=1000)
    s = normalize_intensities(s)
    if n_required_peaks is not None:
        s = require_minimum_number_of_peaks(s, n_required=n_required_peaks)

    if max_peaks is not None:
        s = reduce_to_number_of_peaks(s, n_max=max_peaks)

    if min_rel_int is not None:
        s = select_by_relative_intensity(s, intensity_from=min_rel_int)

    return s


class TopKDS(Dataset):
    def __init__(self, ref_docs, vocab, k=5, onehot=True, add_intensity=True, mode="one", onehot_y=False):
        self.ref_docs = ref_docs
        self.k = k
        self.vocab = vocab

        # drop too little peaky spectra
        self.ref_docs = [dr for dr in self.ref_docs if len(dr.words) > k]
        self.d_position = 0
        self.onehot = onehot
        self.add_intensity = add_intensity
        self.mode = mode
        self.onehot_y = onehot_y

        # get indeces mapping 
        if self.mode == "all":
            self.counts = list(map(lambda ref_doc: len(ref_doc.peaks) - self.k, self.ref_docs))
            nested_mapping = [list(zip([i] * n, range(n))) for i, n in enumerate(self.counts)]
            self.mapping = [item for sublist in nested_mapping for item in sublist]

    def __len__(self) -> int:
        if self.mode == "all":
            return sum(self.counts)
        return len(self.ref_docs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "all":
            idx, y_idx = self.mapping[idx]

        top_k, top_after_k = get_top_k_ind(self.ref_docs[idx].peaks.intensities, self.k)

        if self.add_intensity:
            top_k_intens = self.ref_docs[idx].peaks.intensities[top_k]
            X = self.encode_spec_intens(np.array(self.ref_docs[idx].words)[top_k], top_k_intens)
        else:
            X = self.encode_spec(np.array(self.ref_docs[idx].words)[top_k])

        to_predict = top_after_k
        if self.mode == "all":
            to_predict = y_idx
        if self.mode == "random":
            to_predict = np.random.randint(0, len(self.ref_docs[idx].peaks))

        y = self.encode(np.array(self.ref_docs[idx].words)[to_predict])
        if self.onehot_y:
            y = nn.functional.one_hot(y, len(self.vocab))
        return X, y

    def size(self):
        return len(self.vocab)

    def encode(self, peak):
        if peak not in self.vocab:
            return torch.tensor(-1, dtype=torch.int)
        return torch.tensor(self.vocab[peak], dtype=torch.int)

    def encode_spec(self, spec):
        res = []
        for i, peak in enumerate(spec):
            enc = self.encode(peak)
            if enc == -1:
                continue

            if self.onehot:
                res.append(nn.functional.one_hot(enc, len(self.vocab)))
            else:
                res.append(self.encode(peak))

        if self.onehot:
            return torch.stack(res).float()
        return torch.tensor(res, dtype=torch.int)

    def encode_spec_intens(self, spec, intens):
        res = []
        for i, peak in enumerate(spec):
            enc = self.encode(peak)
            if enc == -1:
                continue

            if self.onehot:
                res.append(torch.cat((nn.functional.one_hot(enc, len(self.vocab)), torch.tensor(intens[i]).reshape(-1))))
            else:
                res.append(torch.tensor((self.encode(peak), intens[i])))
        return torch.stack(res).float()


class GenDS(Dataset):
    def __init__(self, ref_docs, vocab, onehot=True, add_intensity=True):
        self.ref_docs = sorted(ref_docs, key=lambda doc: len(doc.words), reverse=True)
        self.vocab = vocab
        self.onehot = onehot
        self.add_intensity = add_intensity

    def __len__(self) -> int:
        return len(self.ref_docs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ordered_ind = np.argsort(self.ref_docs[idx].peaks.intensities)[::-1]
        X_ind = ordered_ind[:-1]
        y_ind = ordered_ind[1:]

        if self.add_intensity:
            intens = self.ref_docs[idx].peaks.intensities[X_ind]
            X = self.encode_spec_intens(np.array(self.ref_docs[idx].words)[X_ind], intens)
        else:
            X = self.encode_spec(np.array(self.ref_docs[idx].words)[X_ind])

        y = torch.tensor([self.encode(peak) for peak in np.array(self.ref_docs[idx].words)[y_ind]])
        y = y[y != -1]
        return X, y

    def size(self):
        return len(self.vocab)

    def encode(self, peak):
        if peak not in self.vocab:
            return torch.tensor(-1, dtype=torch.int)
        return torch.tensor(self.vocab[peak], dtype=torch.int)

    def encode_spec(self, spec):
        res = []
        for i, peak in enumerate(spec):
            enc = self.encode(peak)
            if enc == -1:
                continue

            if self.onehot:
                res.append(nn.functional.one_hot(enc, len(self.vocab)))
            else:
                res.append(self.encode(peak))
        if self.onehot:
            return torch.stack(res).float()
        return torch.tensor(res, dtype=torch.int)

    def encode_spec_intens(self, spec, intens):
        res = []
        for i, peak in enumerate(spec):
            enc = self.encode(peak)
            if enc == -1:
                continue

            if self.onehot:
                res.append(
                    torch.cat(
                        (
                            nn.functional.one_hot(enc, len(self.vocab)),
                            torch.tensor(intens[i]).reshape(-1)
                        )
                    )
                )
            else:
                res.append(torch.tensor((self.encode(peak), intens[i])))
        return torch.stack(res).float()


def gen_collate(batch):
    padded = nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
    target = nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True, padding_value=-100)

    target = torch.LongTensor(target)
    return [padded, target]


class HuggDS(Dataset):
    def __init__(self, ref_docs, vocab, max_len=256, add_intensity=False, quadratic=False):
        self.ref_docs = ref_docs
        self.vocab = vocab
        self.max_len = max_len
        self.add_intensity = add_intensity
        self.quadratic = quadratic
        self.quad_bins = ((np.arange(max_len) ** 2) / ((max_len - 1) ** 2))[::-1]
        self.lin_bins = (np.arange(max_len) / (max_len - 1))[::-1]

    def __len__(self) -> int:
        return len(self.ref_docs)

    @property
    def bins(self):
        if self.quadratic:
            return self.quad_bins
        return self.lin_bins

    def __getitem__(self, idx: int):
        ordered_ind = np.argsort(self.ref_docs[idx].peaks.intensities)[::-1]
        X_ind = ordered_ind[:self.max_len]

        if self.add_intensity:
            position_ids = torch.tensor(np.digitize(self.ref_docs[idx].peaks.intensities[X_ind], self.bins, right=False))
            position_ids = torch.cat([position_ids, torch.zeros(self.max_len - len(position_ids), dtype=torch.int) + self.max_len - 1])

        X = self.encode_spec(np.array(self.ref_docs[idx].words)[X_ind])
        att_mask = torch.zeros(self.max_len, dtype=torch.int)
        att_mask[:len(X)] = 1
        X_padded = torch.cat((X, torch.zeros(self.max_len - len(X), dtype=torch.int)))

        if self.add_intensity:
            return {"input_ids": X_padded, "attention_mask": att_mask, "position_ids": position_ids}
        return {"input_ids": X_padded, "attention_mask": att_mask}

    def size(self):
        return len(self.vocab)

    def encode(self, peak):
        if peak not in self.vocab:
            return torch.tensor(-1, dtype=torch.int)
        return torch.tensor(self.vocab[peak], dtype=torch.int)

    def encode_spec(self, spec):
        res = torch.empty(len(spec), dtype=torch.int)
        for i, peak in enumerate(spec):
            res[i] = self.encode(peak)
        return res[res != -1]


class FixedSizeDS(Dataset):
    def __init__(self, spectrums, p=0.2, max_mz=1001, cum_level=.95):
        self.spectrums = spectrums
        self.max_mz = max_mz
        self.cum_level = cum_level
        self.p = p

        self.reset_rng(42)

    def __len__(self) -> int:
        return len(self.spectrums)

    def __getitem__(self, idx: int):
        spec = self.spectrums[idx]
        mz_vector = get_mz_vector(spec, self.max_mz).astype(np.float32)

        his_size = get_his_size(spec, self.cum_level)

        # get to k indices in linear time
        his_ind = np.argpartition(mz_vector, -his_size)[-his_size:]
        mask_missing = self.rng.uniform(0, 1, self.max_mz) < self.p

        X = mz_vector
        X[mask_missing] = 0

        his_mask = np.zeros_like(mz_vector) == 1
        his_mask[his_ind] = True

        y = np.zeros_like(mz_vector, dtype=np.float32)
        y[mask_missing & his_mask] = 1

        return torch.from_numpy(X), torch.from_numpy(y)

    def reset_rng(self, seed=None):
        self.rng = np.random.default_rng(seed)


class IntegerMzCoder:
    def __init__(self, w2v, max_mz=1500):
        self.max_mz = max_mz
        self.mz2index = self.get_emb_indices_by_mzs(w2v)
        self.index2mz = np.array([*map(lambda x: int(x.split("@")[1]), w2v.wv.index2entity)], dtype=int)

    def get_emb_indices_by_mzs(self, w2v):
        mz2index = np.empty(self.max_mz, dtype=int)
        mz2index[:] = np.NaN
        for i, e in enumerate(w2v.wv.index2entity):
            mz = e.split("@")[1]

            mz2index[int(mz)] = i

        return mz2index

    def encode(self, integer_mzs, safe=True):
        arr = self.mz2index[integer_mzs]
        if safe:
            return arr[~np.isnan(arr)], np.argwhere(~np.isnan(arr))
        return arr

    def decode(self, indices):
        return self.index2mz[indices]

    def transform_to_mz(self, ndarray):
        if ndarray.ndim == 1:
            mz_array = np.zeros(self.max_mz)
            mz_array[self.index2mz] = ndarray
            return mz_array  # np.trim_zeros(mz_array, trim='b')

        if ndarray.ndim == 2:
            mz_array = np.zeros((len(ndarray), self.max_mz))
            mz_array[:, self.index2mz] = ndarray
            return mz_array

    def get_embedding_dimension(self):
        return len(self.index2mz)


class TextMzCoder:
    def __init__(self, w2v, max_mz=None):
        self.index2entity = np.array(w2v.wv.index2entity)
        self.vocab = {e: i for i, e in enumerate(w2v.wv.index2entity)}
        self.index2mz = np.array([*map(lambda x: int(x.split("@")[1]), w2v.wv.index2entity)], dtype=int)

        if max_mz is None:
            self.max_mz = self.index2mz.max() + 1
        else:
            self.max_mz = max_mz

    def encode(self, text_peaks, safe=True):
        arr = []
        kept = []
        for i, word in enumerate(text_peaks):
            if word in self.vocab:
                arr.append(self.vocab[word])
                kept.append(i)
        return np.array(arr, dtype=int), kept

    def decode(self, indices):
        return self.index2entity[indices]

    def transform_to_mz(self, ndarray, n_dec):
        assert n_dec == 0
        if ndarray.ndim == 1:
            mz_array = np.zeros(self.max_mz)
            mz_array[self.index2mz] = ndarray
            return mz_array  # np.trim_zeros(mz_array, trim='b')

        if ndarray.ndim == 2:
            mz_array = np.zeros((len(ndarray), self.max_mz))
            mz_array[:, self.index2mz] = ndarray
            return mz_array

    def text_peak_to_mz(self, text_peak, n_dec):
        assert n_dec == 0
        return int(text_peak.split("@")[1])

    def get_embedding_dimension(self):
        return len(self.index2entity)


class BasicCoder:
    def __init__(self, *args, max_mz=1001, **kwargs):
        self.max_mz = max_mz

    def transform_to_mz(self, ndarray, n_dec):
        return ndarray

    def text_peak_to_mz(self, text_peak, n_dec):
        assert n_dec == 0
        return int(text_peak.split("@")[1])
