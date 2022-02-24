import os

import numpy as np

from .metrics import calc_mean_lj_metrics, calc_mean_random_metrics


def calc_predictions_random(probs, predictors, datasets, d_name, P_FOLDER, \
                            batch_size=256, device="cpu", cum_level=.95, verbose=False):
    for p_name in predictors:
        print(p_name)
        predictor = predictors[p_name]

        pe = predictor.predict_random_all(datasets[d_name], probs=probs, cum_level=cum_level, \
                                          batch_size=batch_size, device=device, verbose=verbose)

        some_pred_per_m, m_pred_per_m, m_y_per_m = pe

        # save expensive prediction 
        os.makedirs(f"{P_FOLDER}/{d_name}/{p_name}", exist_ok=True)
        np.save(f"{P_FOLDER}/{d_name}/{p_name}/some_pred_per_m", some_pred_per_m)
        np.save(f"{P_FOLDER}/{d_name}/{p_name}/m_pred_per_m", m_pred_per_m)
        np.save(f"{P_FOLDER}/{d_name}/{p_name}/m_y_per_m", m_y_per_m)


def calc_predictions(up_to_k, l, predictors, datasets, d_name, P_FOLDER, \
                     batch_size=256, device="cpu", verbose=False):
    for p_name in predictors:
        print(p_name)
        predictor = predictors[p_name]

        pe = predictor.predict_l_all(datasets[d_name], up_to_k=up_to_k, l=l, \
                                     batch_size=batch_size, device=device, verbose=verbose)

        l_pred_indices_per_k, y_indices, X_intens = pe

        # save expensive prediction 
        os.makedirs(f"{P_FOLDER}/{d_name}/{p_name}", exist_ok=True)
        np.save(f"{P_FOLDER}/{d_name}/{p_name}/l_pred_indices_per_k", l_pred_indices_per_k)
        np.save(f"{P_FOLDER}/{d_name}/{p_name}/y_indices", y_indices)
        np.save(f"{P_FOLDER}/{d_name}/{p_name}/X_intens", X_intens)


def load_predictions_random(p_name, d_name, P_FOLDER):
    some_pred_per_m = np.load(f"{P_FOLDER}/{d_name}/{p_name}/some_pred_per_m.npy", allow_pickle=True)
    m_pred_per_m = np.load(f"{P_FOLDER}/{d_name}/{p_name}/m_pred_per_m.npy", allow_pickle=True)
    m_y_per_m = np.load(f"{P_FOLDER}/{d_name}/{p_name}/m_y_per_m.npy", allow_pickle=True)

    return some_pred_per_m, m_pred_per_m, m_y_per_m


def load_predictions(p_name, d_name, P_FOLDER):
    l_pred_indices_per_k = np.load(f"{P_FOLDER}/{d_name}/{p_name}/l_pred_indices_per_k.npy")
    y_indices = np.load(f"{P_FOLDER}/{d_name}/{p_name}/y_indices.npy", allow_pickle=True)
    X_intens = np.load(f"{P_FOLDER}/{d_name}/{p_name}/X_intens.npy", allow_pickle=True)

    return l_pred_indices_per_k, y_indices, X_intens


def model_selection(P_FOLDER, d_name, up_to_k, l, j, to_rel_inten=.2, \
                    l_rel=None, predictors=None, kw=None, mask=None, return_details=False):
    best_p_name = None
    best_score = 0
    scores = dict()

    for (dirpath, dirnames, filenames) in os.walk(f"{P_FOLDER}/{d_name}/"):
        if kw is not None:
            selected = [dirname for dirname in dirnames if kw in dirname]
        if predictors is not None:
            selected = set(dirnames).intersection(set(predictors.keys()))
        for p_name in selected:
            print(p_name)
            l_pred_indices_per_k, y_indices, X_intens = load_predictions(p_name, d_name, P_FOLDER)

            met = calc_mean_lj_metrics(l_pred_indices_per_k, y_indices, X_intens, up_to_k=up_to_k, \
                                       l=l, j=j, to_rel_inten=to_rel_inten, l_rel=l_rel, mask=mask,
                                       return_details=return_details)

            scores[p_name] = met

            score = np.array(met["mpi"]).mean()

            if score > best_score:
                best_p_name = p_name
                best_score = score

    return best_p_name, scores


def model_selection_random(P_FOLDER, d_name, predictors=None, kw=None, mask=None, return_details=False):
    best_p_name = None
    best_score = 0
    scores = dict()

    for (dirpath, dirnames, filenames) in os.walk(f"{P_FOLDER}/{d_name}/"):
        if kw is not None:
            selected = [dirname for dirname in dirnames if kw in dirname]
        if predictors is not None:
            selected = set(dirnames).intersection(set(predictors.keys()))
        for p_name in selected:
            print(p_name)
            some_pred_per_m, m_pred_per_m, m_y_per_m = load_predictions_random(p_name, d_name, P_FOLDER)

            met = calc_mean_random_metrics(some_pred_per_m, m_pred_per_m, m_y_per_m, mask=mask,
                                           return_details=return_details)
            scores[p_name] = met

            score = np.nanmean(np.array(met["mf1"]))

            if score > best_score:
                best_p_name = p_name
                best_score = score

    return best_p_name, scores
