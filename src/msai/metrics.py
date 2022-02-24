import numpy as np

from .utils import get_mean_nan


def set_metrics(pred_next, y_next):
    pred_next_set = set(pred_next)
    y_next_set = set(y_next)

    TP = len(y_next_set.intersection(pred_next_set))
    FP = len(pred_next_set) - TP
    FN = len(y_next_set) - TP

    precision = TP / (TP + FP) if TP + FP != 0 else np.NaN
    recall = TP / (TP + FN) if TP + FN != 0 else np.NaN

    jaccard = TP / len(y_next_set.union(pred_next_set)) if len(y_next_set.union(pred_next_set)) != 0 else np.NaN

    return precision, jaccard, recall, (TP, FP, FN, (pred_next, y_next))


def metrics_klj(l_pred_indices_per_k, y_indices, up_to_k=None, l=None, j=None):
    if up_to_k is None:
        up_to_k = len(l_pred_indices_per_k)
    else:
        up_to_k = min(up_to_k, len(l_pred_indices_per_k))
    if l is None:
        l = l_pred_indices_per_k.shape[2]
    print(f"Selected up to k={up_to_k}, l={l}, j={j}")

    precisions = np.zeros(shape=(up_to_k, l_pred_indices_per_k.shape[1]))
    jaccards = np.zeros(shape=(up_to_k, l_pred_indices_per_k.shape[1]))

    for k in range(up_to_k):
        for i in range(l_pred_indices_per_k.shape[1]):
            pred_next = l_pred_indices_per_k[k][i][:l]
            y_next = y_indices[i][k:k + j]

            # skip too short spectra indicated by -1
            if (l_pred_indices_per_k[k][i][:l] == -1).any() or len(pred_next) != l or len(y_next) != j:
                precisions[k, i] = np.NaN
                jaccards[k, i] = np.NaN
                continue
            # calculate metrics order respecting 
            # not implemented 

            # calculete metrics set
            precision, jaccard, _, _ = set_metrics(pred_next, y_next)

            precisions[k, i] = precision
            jaccards[k, i] = jaccard

    return precisions, jaccards


def metrics_klrel(l_pred_indices_per_k, y_indices, X_intens, up_to_k=None, l=None, to_rel_inten=0.2,
                  return_details=False):
    if up_to_k is None:
        up_to_k = len(l_pred_indices_per_k)
    else:
        up_to_k = min(up_to_k, len(l_pred_indices_per_k))

    print(f"Selected up to k={up_to_k}, l={l}, to_rel_inten={to_rel_inten}")

    n_records = l_pred_indices_per_k.shape[1]
    precisions = np.zeros(shape=(up_to_k, n_records))
    jaccards = np.zeros(shape=(up_to_k, n_records))
    if return_details:
        details = {
            "conf": np.zeros(shape=(3, up_to_k, l_pred_indices_per_k.shape[1])),
            "all": {
                "preds": [[None for _ in range(n_records)] for _ in range(up_to_k)],
                "ys": [[None for _ in range(n_records)] for _ in range(up_to_k)]
            }
        }

    assert len(y_indices) == len(X_intens)

    for k in range(up_to_k):
        for i in range(l_pred_indices_per_k.shape[1]):

            # skip spectrum if it contains peak unseen in training
            if len(y_indices[i]) != len(X_intens[i]):
                if k == 0:
                    print("skipped spectrum - unknown peak")
                precisions[k, i] = np.NaN
                jaccards[k, i] = np.NaN
                continue

            assert len(y_indices[i]) == len(X_intens[i])

            # skip spectrum with less than k peaks
            if len(y_indices[i]) <= k:
                precisions[k, i] = np.NaN
                jaccards[k, i] = np.NaN
                continue

            # parse various representation
            if isinstance(X_intens[i], list):
                intens = np.array(X_intens[i])
            else:
                intens = X_intens[i]

            # get number of peaks above some intensity of last seen peak
            j = min(np.argmax(intens < intens[k] * to_rel_inten) - k - 1, 20)

            # did not find any peak lower than given threshold
            if j <= -1:
                j = min(len(intens) - k, 20)

            # did not find any peak above or equal the given threshold
            if j == 0:
                precisions[k, i] = np.NaN
                jaccards[k, i] = np.NaN
                continue

            # set the number of peaks that the model should predict
            if l is None:
                curr_l = j
            else:
                curr_l = l

            # check if precomputed predictions satisfy the requested number of peaks
            # skip otherwise
            if l_pred_indices_per_k.shape[2] < curr_l:
                precisions[k, i] = np.NaN
                jaccards[k, i] = np.NaN

                print("skipped spectrum - too little pred")
                continue

            # get next l predicted and next j reference (if l was none -> j==l) 
            pred_next = l_pred_indices_per_k[k][i][:curr_l]
            y_next = y_indices[i][k:k + j]

            assert not (l_pred_indices_per_k[k][i][:l] == -1).any()

            # calculete metrics set
            precision, jaccard, _, detail = set_metrics(pred_next, y_next)

            precisions[k, i] = precision
            jaccards[k, i] = jaccard

            if return_details:
                details["conf"][:, k, i] = detail[:-1]
                details["all"]["preds"][k][i] = detail[-1][0]
                details["all"]["ys"][k][i] = detail[-1][1]

    out = {"precs": precisions, "jacs": jaccards}
    if return_details:
        return {**out, "dets": details}
    return out


def calc_mean_lj_metrics(l_pred_indices_per_k, y_indices, X_intens, up_to_k=None, \
                         l=None, j=None, l_rel=None, to_rel_inten=0.2, mask=None, \
                         return_details=False):
    print(f"Possible k up to {len(l_pred_indices_per_k)}, predict up to {l_pred_indices_per_k.shape[2]} peaks")

    precisions, jaccards = metrics_klj(l_pred_indices_per_k, y_indices, up_to_k=up_to_k, l=l, j=j)

    if mask is not None:
        precisions[~mask] = np.NaN
        jaccards[~mask] = np.NaN

    metrics_rel = metrics_klrel(l_pred_indices_per_k, y_indices, X_intens, \
                                up_to_k=up_to_k, l=l_rel, to_rel_inten=to_rel_inten,
                                return_details=return_details)

    if mask is not None:
        metrics_rel["precs"][~mask] = np.NaN
        metrics_rel["jacs"][~mask] = np.NaN

    print((~np.isnan(metrics_rel["precs"])).sum(axis=1))

    scores = {
        "mp": get_mean_nan(precisions),
        "mj": get_mean_nan(jaccards),
        "mpi": get_mean_nan(metrics_rel["precs"]),
        "mji": get_mean_nan(metrics_rel["jacs"])
    }

    if return_details:
        return {**scores, **metrics_rel}

    return scores


def calc_mean_random_metrics(some_pred_per_m, m_pred_per_m, m_y_per_m, mask=None, return_details=False):
    precs_m = np.zeros(shape=(len(m_y_per_m), len(m_y_per_m[0])))
    precs_some = np.zeros(shape=(len(m_y_per_m), len(m_y_per_m[0])))
    recs_some = np.zeros(shape=(len(m_y_per_m), len(m_y_per_m[0])))
    jacs_some = np.zeros(shape=(len(m_y_per_m), len(m_y_per_m[0])))
    f1_some = np.zeros(shape=(len(m_y_per_m), len(m_y_per_m[0])))

    n_records = len(m_y_per_m[0])
    if return_details:
        details = {"conf": np.zeros(shape=(3, len(m_y_per_m), n_records)),
                   "all": {"preds": [[None for _ in range(n_records)] for _ in range(len(m_y_per_m))],
                           "ys": [[None for _ in range(n_records)] for _ in range(len(m_y_per_m))]}
                   }
    for m in range(len(m_y_per_m)):
        for i in range(n_records):
            if mask is not None and not mask[m, i]:
                precs_some[m][i] = np.NaN
                recs_some[m][i] = np.NaN
                jacs_some[m][i] = np.NaN
                precs_m[m][i] = np.NaN
                f1_some[m][i] = np.NaN
                continue

            prec, jac, recall, detail = set_metrics(some_pred_per_m[m][i], m_y_per_m[m][i])

            precs_some[m][i] = prec
            recs_some[m][i] = recall
            jacs_some[m][i] = jac

            if np.isnan(prec) or np.isnan(recall) or prec + recall == 0:
                f1_some[m][i] = np.NaN
            else:
                f1_some[m][i] = 2 * prec * recall / (prec + recall)

            prec_m, _, _, _ = set_metrics(m_pred_per_m[m][i], m_y_per_m[m][i])
            precs_m[m][i] = prec_m

            if return_details:
                details["conf"][:, m, i] = detail[:-1]
                details["all"]["preds"][m][i] = detail[-1][0]
                details["all"]["ys"][m][i] = detail[-1][1]

    scores = {
        "mp": np.nanmean(precs_some, axis=1),
        "mr": np.nanmean(recs_some, axis=1),
        "mj": np.nanmean(jacs_some, axis=1),
        "mf1": np.nanmean(f1_some, axis=1),
        "mps": np.nanmean(precs_m, axis=1)
    }

    if return_details:
        return {**scores, **details}
    return scores


def accuracy_at_k(l_pred_indices_per_k, y_indices, up_to_k=None):
    if up_to_k is None:
        up_to_k = len(l_pred_indices_per_k)
    else:
        up_to_k = min(up_to_k, len(l_pred_indices_per_k))
    accs = np.zeros(up_to_k)

    for k in range(up_to_k):
        corr = 0
        tot = 0
        for i in range(len(y_indices)):

            # skip too short spectra
            if (l_pred_indices_per_k[k, i, :] == -1).any() or len(y_indices[i]) <= k:
                continue

            # safety check 
            assert len(y_indices[i]) > k

            if l_pred_indices_per_k[k, i, 0] == y_indices[i][k]:
                corr += 1
            tot += 1
        accs[k] = 0 if tot == 0 else corr / tot
    return accs


def accuracy_at_int(l_pred_indices_per_k, y_indices, X_intens, n_bins=10, split="uniform"):
    corr_at_int = np.zeros(n_bins)
    tot_at_int = np.zeros(n_bins)
    for k in range(len(l_pred_indices_per_k)):
        for i in range(len(y_indices)):

            # skip too short spectra
            if (l_pred_indices_per_k[k, i, :] == -1).any() or k >= len(X_intens[i]):
                continue
            # safety check 
            assert len(y_indices[i]) > k

            bin_ = get_bin(X_intens[i][k], n_bins, split)

            if l_pred_indices_per_k[k, i, 0] == y_indices[i][k]:
                corr_at_int[bin_] += 1
            tot_at_int[bin_] += 1
    return corr_at_int / tot_at_int


def get_bin(intensity, n_bins, split):
    if split == "uniform":
        bins = np.arange(n_bins - 1) / n_bins
    return np.digitize(intensity, bins)


def metrics_intlj(l_pred_indices_per_k, y_indices, X_intens, l, j, down_to_int=None, n_bins=10, split="uniform"):
    """
    compute the precision and jaccard of peaks by agregating on the intensity of last seen peak
    """

    precisions = [[] for _ in range(n_bins)]
    jaccards = [[] for _ in range(n_bins)]

    for k in range(len(l_pred_indices_per_k)):
        for i in range(l_pred_indices_per_k.shape[1]):
            pred_next = l_pred_indices_per_k[k][i][:l]
            y_next = y_indices[i][k:k + j]

            # skip too short spectra indicated by -1
            if (l_pred_indices_per_k[k][i][:l] == -1).any() or len(pred_next) != l or len(y_next) != j:
                continue

            # calculate metrics order respecting 
            # not implemented 

            # calculete metrics set
            precision, jaccard = set_metrics(pred_next, y_next)

            intensity = X_intens[i][k]

            bin_ = get_bin(intensity, n_bins, split)

            precisions[bin_].append(precision)
            jaccards[bin_].append(jaccard)

    return precisions, jaccards
