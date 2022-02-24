import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import get_top_k_ind

COLORS = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]


def plot_scores(scores, metrics="mpi", x=None, grouper_f=lambda x: "x", \
                orderer_f=lambda x: x, title=None, save_to_path=None, \
                xlabel=None, ylabel=None, hue_f=None, y_max=None):
    plt.figure(figsize=(7, 4))
    styles = ["-", "--", ":", "-."]
    map_ = {"": -1}
    # print(scores.keys())
    for p_name in sorted(scores.keys(), key=orderer_f):

        kind = grouper_f(p_name)
        # print(hue_f(p_name))
        if kind not in map_:
            map_[kind] = max(map_.values()) + 1
        kind = map_[kind]

        if x is None:
            x_ = np.arange(len(scores[p_name][metrics])) + 1
        else:
            x_ = x
        if hue_f is not None:
            sns.lineplot(y=scores[p_name][metrics], x=x_, linestyle=styles[kind], \
                         linewidth=.7, color=COLORS[hue_f(p_name)])
        else:
            sns.lineplot(y=scores[p_name][metrics], x=x_, linestyle=styles[kind], \
                         linewidth=.7)

    plt.legend(sorted(scores.keys(), key=orderer_f), loc='lower right')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if y_max is not None:
        plt.ylim(0, y_max)
    if x is None:
        plt.xticks(ticks=np.arange(1, len(scores[p_name][metrics]) + 1, 2))  # ,fontsize=8, rotation=45)
    plt.grid(color='whitesmoke', linestyle='-')

    if save_to_path is not None:
        plt.savefig(save_to_path, bbox_inches='tight')
    plt.show()


def plot_training(learner):
    plt.figure(figsize=(8, 8))
    plt.plot(learner.train_losses)
    plt.plot(learner.val_losses)
    plt.legend(["training loss", "validation loss"])
    plt.title(f"{learner.model_name} - loss over epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    # plt.ylim(1,2)
    plt.show()


def plot_stats(data1D, baseline1D=None, max_len=None, title=None, log_y=False, color="blue", \
               decreasing=False, ylim=None, ylabel=None, xlabel=None, \
               x_factor=1, disable_scientific=False):
    # sns.set()
    plt.figure(figsize=(20, 10))
    if title:
        plt.title(title)
    if max_len is None:
        max_len = len(data1D)

    x = np.arange(max_len) * x_factor

    if baseline1D is not None:
        ax = sns.scatterplot(y=baseline1D[:max_len], x=x, color="darkgrey")
        ax = sns.scatterplot(y=data1D[:max_len], x=x, color=color, ax=ax)
    else:
        ax = sns.scatterplot(y=data1D[:max_len], x=x, color=color)

    if decreasing:
        plt.gca().invert_xaxis()
    if log_y:
        ax.set(yscale="log")

    if ylim is not None:
        ax.set_ylim(ylim)
    if disable_scientific:
        ax.ticklabel_format(style='plain')

    ax.set(ylabel=ylabel, xlabel=xlabel)
    plt.xticks(ticks=np.arange(0, max_len, 10) * x_factor, fontsize=8, rotation=45)
    plt.show()


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def plot_spectrum_predictions(ref_doc, k, prob, coder, n_detail=10, plot_full=True, \
                              log_y=False, save_to_path=None, down_to=0.2):
    matplotlib.rc_file_defaults()

    # basic variables
    n_dec = ref_doc.n_decimals
    bars_len = coder.max_mz  # len(base)
    base = np.arange(bars_len)
    p_inten = ref_doc.weights

    ######################
    # reference spectrum #
    ######################

    # create an y axis array of bars
    bars = np.zeros(bars_len)
    for p, peak in enumerate(ref_doc.words):
        # loc = peakname_to_loc(peak, n_dec)
        loc = coder.text_peak_to_mz(peak, n_dec)
        bars[loc] = max(bars[loc], p_inten[p])

    # get top_k_ind
    top_k_ind, _ = get_top_k_ind(ref_doc.peaks.intensities, k)

    # distinguish top_k peaks by color
    hue = np.repeat("missing peak", bars_len)
    for ind in top_k_ind:
        hue[coder.text_peak_to_mz(ref_doc.words[ind], n_dec)] = f"regular peak"

    # distinguish filtered by color
    hue_pred = np.repeat("returned", bars_len)
    for ind in top_k_ind:
        hue_pred[coder.text_peak_to_mz(ref_doc.words[ind], n_dec)] = "filtered"
    palette = {"filtered": "rosybrown", "returned": "crimson",
               "missing peak": "tab:orange", "regular peak": "tab:blue"}

    if plot_full:
        # plot it
        f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(40, 10), sharex=True)
        ax1 = sns.barplot(x=base, y=bars, hue=hue, ax=ax1)

    # add labels on peaks from kth to 0.2kth intensity wise
    # max_l_ref = np.argsort(bars)[-n_detail]
    # big_ref = np.nonzero( bars >= max_l_ref)[0]
    kth_intens = bars[coder.text_peak_to_mz(ref_doc.words[top_k_ind[-1]], n_dec)]
    big_ref = np.nonzero([(bars < kth_intens) & (bars > kth_intens * down_to)])[1]
    if plot_full:
        plt.sca(ax1)
        for loc in big_ref:
            plt.text(loc, bars[loc] + np.max(bars) / 20, loc, ha='center', rotation=90, va='bottom')

    #########################
    # predicted distributon #
    #########################

    # convert prediction from spec2vec indexing to location on mz indexing
    bars_pred = coder.transform_to_mz(prob, n_dec)

    # add labels on top of n_details most probable peaks
    # max_l = np.sort(bars_pred)[-n_detail]
    # big = np.nonzero( bars_pred >= max_l)[0]  
    bars_pred_ = bars_pred.copy()
    bars_pred_[hue_pred == "filtered"] = 0
    big = np.argsort(bars_pred_)[-len(big_ref):]

    if plot_full:
        plt.sca(ax2)
        for loc in big:
            plt.text(loc, bars_pred[loc] + np.max(bars_pred) / 20, loc, ha='center', rotation=90, va='bottom')

        # plot it
        ax2 = sns.barplot(x=base, y=bars_pred, ax=ax2, dodge=False, hue=hue_pred, palette=palette)
        change_width(ax2, .75)

        if log_y:
            ax1.set(yscale="log")
            ax2.set(yscale="log")

            ax2.set_ylim(10e-5)
            ax1.sharey(ax2)

        ax1.title.set_text(f"Full reference spectrum of {ref_doc.metadata['name']}")
        ax2.title.set_text(f"Predicted distribution for {len(top_k_ind) + 1}. peak")
        ax1.xaxis.set_tick_params(labelbottom=True)
        ax1.set(ylabel='intensity')
        ax2.set(xlabel='m/z', ylabel='prediction')
        plt.sca(ax1)
        plt.xticks(ticks=np.arange(0, 1001, 10), fontsize=8, rotation=45)
        plt.sca(ax2)
        plt.xticks(ticks=np.arange(0, 1001, 10), fontsize=8, rotation=45)
        plt.show()

    ##################################
    # focused wiew on the dense area #
    ##################################

    # get focus area with np.quantile and nasty hack
    meta = []
    for peak, intensity in zip(ref_doc.words, p_inten):
        meta += [coder.text_peak_to_mz(peak, n_dec)] * int(intensity * 1000)
    focus = max(0, np.quantile(meta, 0.05) - 25), np.quantile(meta, 0.95) + 25

    # avoid the highest peak interfering with title
    #     if abs(sum(focus)/2 - np.argmax(bars)) < 50:
    #         big_ref = big_ref[big_ref != np.argmax(bars)]

    # plot focused
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 9), sharex=True)
    ax1 = sns.barplot(x=base, y=bars, ax=ax1, hue=hue, dodge=False, palette=palette)
    ax2 = sns.barplot(x=base, y=bars_pred, ax=ax2, dodge=False, hue=hue_pred, palette=palette)
    change_width(ax1, .55)
    change_width(ax2, .55)

    plt.sca(ax1)
    # add labels on top of peaks
    last = -1000
    for loc in sorted(big_ref):
        # skip
        if loc - last < 2 and abs(bars[loc] - bars[last]) < 0.1:
            continue
        if loc > focus[1] or loc < focus[0]:
            continue

        plt.text(loc, bars[loc] + np.max(bars) / 20, loc, ha='center', rotation=90, va='bottom')
        last = loc
    plt.sca(ax2)
    last = -1000
    for loc in sorted(big):
        # skip
        if loc - last < 2 and abs(bars_pred[loc] - bars_pred[last]) < 0.1:
            continue
        if loc > focus[0] and loc < focus[1]:
            plt.text(loc, bars_pred[loc] + np.max(bars_pred) / 20, loc, ha='center', rotation=90, va='bottom')
        last = loc
    # plot focused
    if log_y:
        ax1.set(yscale="log")
        ax2.set(yscale="log")
        # ax2.sharey(ax1)

        ax2.set_ylim(10e-5)
        ax1.sharey(ax2)

    ax1.title.set_text(f"Focused reference spectrum of {ref_doc.metadata['name']}")
    ax2.title.set_text(f"Prediction for {len(top_k_ind) + 1}. peak")
    ax1.set_xlim(*focus)
    ax2.set_xlim(*focus)
    ax1.xaxis.set_tick_params(labelbottom=True)
    ax2.xaxis.set_tick_params(labelbottom=True)
    ax1.set(ylabel='intensity')
    ax2.set(xlabel='m/z', ylabel='prediction')
    plt.sca(ax1)
    plt.legend(loc='upper right')
    plt.xticks(ticks=np.arange((focus[0] // 10) * 10, (focus[1] // 10) * 10, 5), fontsize=8, rotation=45)
    plt.sca(ax2)
    plt.legend(loc='upper right')
    plt.xticks(ticks=np.arange((focus[0] // 10) * 10, (focus[1] // 10) * 10, 5), fontsize=8, rotation=45)
    if save_to_path is not None:
        plt.savefig(save_to_path, bbox_inches='tight')
    plt.show()


def plot_spectrum_predictions_random(ref_doc, omitted_ind, prob, coder, n_detail=10, \
                                     plot_full=True, log_y=False, save_to_path=None, predicted_peaks=None):
    matplotlib.rc_file_defaults()

    # basic variables
    n_dec = ref_doc.n_decimals
    bars_len = coder.max_mz  # len(base)
    base = np.arange(bars_len)
    p_inten = ref_doc.weights

    ######################
    # reference spectrum #
    ######################

    # create an y axis array of bars
    bars = np.zeros(bars_len)
    for p, peak in enumerate(ref_doc.words):
        # loc = peakname_to_loc(peak, n_dec)
        loc = coder.text_peak_to_mz(peak, n_dec)
        bars[loc] = max(bars[loc], p_inten[p])

    # distinguish top_k peaks by color
    hue = np.repeat("regular peak", bars_len)
    for ind in omitted_ind:
        hue[ind] = "missing peak"

    # distinguish filtered by color
    hue_pred = np.repeat("filtered", bars_len)
    hue_pred[bars == 0] = "returned"
    for ind in omitted_ind:
        hue_pred[ind] = "returned"
    for ind in predicted_peaks:
        hue_pred[ind] = "returned"
    palette = {"filtered": "rosybrown", "returned": "crimson",
               "missing peak": "tab:orange", "regular peak": "tab:blue"}

    if plot_full:
        # plot it
        f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(40, 10), sharex=True)
        # ax1 = sns.barplot(x=base, y=bars, hue=hue, ax=ax1)
        ax1 = sns.barplot(x=base, y=bars, ax=ax1, dodge=False, hue=hue, palette=palette)
        change_width(ax1, .75)

        # add labels on top of n_details most intensive peaks
    max_l_ref = np.sort(bars)[-n_detail]
    big_ref = np.nonzero(bars >= max_l_ref)[0]

    if plot_full:
        plt.sca(ax1)
        for loc in big_ref:
            plt.text(loc, bars[loc] + np.max(bars) / 20, loc, ha='center', rotation=90, va='bottom')

    #########################
    # predicted distributon #
    #########################

    # convert prediction from spec2vec indexing to location on mz indexing
    bars_pred = coder.transform_to_mz(prob, n_dec)

    # add labels on top of n_details most probable peaks
    # max_l = np.sort(bars_pred)[-n_detail]
    # big = np.nonzero( bars_pred >= max_l)[0]
    bars_pred_ = bars_pred.copy()
    bars_pred_[hue_pred == "filtered"] = 0
    if predicted_peaks is None:
        big = np.argsort(bars_pred_)[-n_detail:]
    else:
        big = predicted_peaks

    if plot_full:
        plt.sca(ax2)
        for loc in big:
            plt.text(loc, bars_pred[loc] + np.max(bars_pred) / 20, loc, ha='center', rotation=90, va='bottom')

        # plot it
        # ax2 = sns.barplot(x=base, y=bars_pred, ax=ax2, color="red")
        ax2 = sns.barplot(x=base, y=bars_pred, ax=ax2, dodge=False, hue=hue_pred, palette=palette)
        change_width(ax2, .75)

        if log_y:
            ax1.set(yscale="log")
            ax2.set(yscale="log")

            ax2.set_ylim(10e-5)
            ax1.sharey(ax2)

        ax1.title.set_text(f"Full reference spectrum of {ref_doc.metadata['name']}")
        ax2.title.set_text(f"Predicted distribution for {len(omitted_ind)} peak")
        ax1.xaxis.set_tick_params(labelbottom=True)
        ax1.set(ylabel='intensity')
        ax2.set(xlabel='m/z', ylabel='prediction')
        plt.sca(ax1)
        plt.xticks(ticks=np.arange(0, bars_len, 10), fontsize=8, rotation=45)
        plt.legend(loc='upper right')
        plt.sca(ax2)
        plt.legend(loc='upper right')
        plt.xticks(ticks=np.arange(0, bars_len, 10), fontsize=8, rotation=45)
        plt.show()

    ##################################
    # focused wiew on the dense area #
    ##################################

    # get focus area with np.quantile and nasty hack
    meta = []
    for peak, intensity in zip(ref_doc.words, p_inten):
        meta += [coder.text_peak_to_mz(peak, n_dec)] * int(intensity * 1000)
    focus = max(0, np.quantile(meta, 0.05) - 25), np.quantile(meta, 0.95) + 25

    # print(big_ref)
    # avoid the highest peak interfering with title
    if abs(sum(focus) / 2 - np.argmax(bars)) < 50:
        big_ref = big_ref[big_ref != np.argmax(bars)]

    # plot focused
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 9), sharex=True)
    ax1 = sns.barplot(x=base, y=bars, ax=ax1, hue=hue, dodge=False, palette=palette)
    ax2 = sns.barplot(x=base, y=bars_pred, ax=ax2, dodge=False, hue=hue_pred, palette=palette)
    change_width(ax1, .55)
    change_width(ax2, .55)

    # add labels on top of peaks
    plt.sca(ax1)
    last = -1000
    for loc in sorted(omitted_ind):
        # skip
        if loc - last < 2 and abs(bars[loc] - bars[last]) < 0.1:
            continue
        if loc > focus[1] or loc < focus[0]:
            continue

        plt.text(loc, bars[loc] + np.max(bars) / 20, loc, ha='center', rotation=90, va='bottom')
        last = loc
    plt.sca(ax2)
    last = -1000
    for loc in sorted(big):
        # skip
        if loc - last < 2 and abs(bars_pred[loc] - bars_pred[last]) < 0.1:
            continue
        if loc > focus[0] and loc < focus[1]:
            plt.text(loc, bars_pred[loc] + np.max(bars_pred) / 20, loc, ha='center', rotation=90, va='bottom')
        last = loc
    # plot focused
    if log_y:
        ax1.set(yscale="log")
        ax2.set(yscale="log")
        # ax2.sharey(ax1)

        ax2.set_ylim(10e-5)
        ax1.sharey(ax2)

    ax1.title.set_text(f"Focused reference spectrum of {ref_doc.metadata['name']}")
    ax2.title.set_text(f"Prediction for {len(omitted_ind)} omitted peaks")
    ax1.set_xlim(*focus)
    ax2.set_xlim(*focus)
    ax1.xaxis.set_tick_params(labelbottom=True)
    ax2.xaxis.set_tick_params(labelbottom=True)
    ax1.set(ylabel='intensity')
    ax2.set(xlabel='m/z', ylabel='prediction')
    plt.sca(ax1)
    plt.legend(loc='upper right')
    plt.xticks(ticks=np.arange((focus[0] // 10) * 10, (focus[1] // 10) * 10, 5), fontsize=8, rotation=45)
    plt.sca(ax2)
    plt.legend(loc='upper right')
    plt.xticks(ticks=np.arange((focus[0] // 10) * 10, (focus[1] // 10) * 10, 5), fontsize=8, rotation=45)
    if save_to_path is not None:
        plt.savefig(save_to_path, bbox_inches='tight')

    plt.show()


def plot_spectrum_sample_missing(proc_spec, missing_mz, max_mz=1001, save_to_path=None):
    # basic variables
    base = np.arange(max_mz)
    p_inten = proc_spec.peaks.intensities

    # create an y axis array of bars
    bars = np.zeros(max_mz)
    for p, mz in enumerate(proc_spec.peaks.mz):
        bars[int(mz)] = p_inten[p]

        # distinguish top_k peaks by color
    hue = np.repeat("detected peak", max_mz)
    for mz in missing_mz:
        hue[int(mz)] = f"missing peak"

    palette = {"filtered": "rosybrown", "returned": "crimson",
               "missing peak": "tab:orange", "detected peak": "tab:blue"}

    # get focus area with np.quantile and nasty hack
    meta = []
    for mz, intensity in proc_spec.peaks.to_numpy:
        meta += [mz] * int(intensity * 1000)
    focus = max(0, np.quantile(meta, 0.05) - 25), np.quantile(meta, 0.95) + 25

    # plot focused
    f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
    ax1 = sns.barplot(x=base, y=bars, ax=ax1, hue=hue, dodge=False, palette=palette)

    change_width(ax1, .55)

    plt.sca(ax1)
    ax1.set_xlim(*focus)
    ax1.xaxis.set_tick_params(labelbottom=True)
    ax1.set(ylabel='intensity')
    ax1.set(xlabel='m/z')

    plt.sca(ax1)
    plt.legend(loc='upper left')
    plt.xticks(ticks=np.arange((focus[0] // 10) * 10, (focus[1] // 10) * 10, 20))
    if save_to_path is not None:
        plt.savefig(save_to_path, bbox_inches='tight')
    plt.show()


def plot_spectrum_sample_metric(proc_spec, missing_mz, little_mz, max_mz=1001, save_to_path=None):
    # basic variables
    base = np.arange(max_mz)
    p_inten = proc_spec.peaks.intensities

    # create an y axis array of bars
    bars = np.zeros(max_mz)
    for p, mz in enumerate(proc_spec.peaks.mz):
        bars[int(mz)] = p_inten[p]

        # distinguish top_k peaks by color
    hue = np.repeat("detected peak", max_mz)
    for mz in missing_mz:
        hue[int(mz)] = f"target peak"

    for mz in little_mz:
        hue[int(mz)] = f"little peak"

    palette = {"little peak": "gainsboro", "returned": "crimson",
               "target peak": "tab:orange", "detected peak": "tab:blue"}

    # get focus area with np.quantile and nasty hack
    meta = []
    for mz, intensity in proc_spec.peaks.to_numpy:
        meta += [mz] * int(intensity * 1000)
    focus = max(0, np.quantile(meta, 0.05) - 25), np.quantile(meta, 0.95) + 25

    # plot focused
    f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
    ax1 = sns.barplot(x=base, y=bars, ax=ax1, hue=hue, dodge=False, palette=palette)

    change_width(ax1, .55)

    plt.sca(ax1)

    # ax1.title.set_text(f"Focused reference spectrum of {proc_spec.metadata['name']}")
    ax1.set_xlim(*focus)
    ax1.xaxis.set_tick_params(labelbottom=True)
    ax1.set(ylabel='intensity')
    ax1.set(xlabel='m/z')

    plt.sca(ax1)
    plt.legend(loc='upper left')
    plt.xticks(ticks=np.arange((focus[0] // 10) * 10, (focus[1] // 10) * 10, 20))
    if save_to_path is not None:
        plt.savefig(save_to_path, bbox_inches='tight')
    plt.show()


def plot_spectrum_sample(proc_spec, max_mz=1001, save_to_path=None):
    # basic variables
    base = np.arange(max_mz)
    p_inten = proc_spec.peaks.intensities

    # create an y axis array of bars
    bars = np.zeros(max_mz)
    for p, mz in enumerate(proc_spec.peaks.mz):
        bars[int(mz)] = p_inten[p]

    hue = np.repeat("detected peak", max_mz)

    palette = {"filtered": "rosybrown", "returned": "crimson",
               "missing peak": "tab:orange", "detected peak": "tab:blue"}
    # get focus area with np.quantile and nasty hack
    meta = []
    for mz, intensity in proc_spec.peaks.to_numpy:
        meta += [mz] * int(intensity * 1000)
    focus = max(0, np.quantile(meta, 0.05) - 25), np.quantile(meta, 0.95) + 25

    # plot focused
    f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
    ax1 = sns.barplot(x=base, y=bars, ax=ax1, hue=hue, dodge=False, palette=palette)

    change_width(ax1, .55)

    plt.sca(ax1)
    ax1.title.set_text(f"Focused reference spectrum of {proc_spec.metadata['name']}")
    ax1.set_xlim(*focus)
    ax1.xaxis.set_tick_params(labelbottom=True)
    ax1.set(ylabel='intensity')
    ax1.set(xlabel='m/z')
    ax1.get_legend().remove()
    plt.sca(ax1)
    # plt.legend(None)
    plt.xticks(ticks=np.arange((focus[0] // 10) * 10, (focus[1] // 10) * 10, 20))
    if save_to_path is not None:
        plt.savefig(save_to_path, bbox_inches='tight')
    plt.show()


def plot_spectrum_sample_prediction(proc_spec, missing_mz, TP_mz, FP_mz, FN_mz, max_mz=1001, save_to_path=None):
    # basic variables
    base = np.arange(max_mz)
    p_inten = proc_spec.peaks.intensities

    # create an y axis array of bars
    bars = np.zeros(max_mz)
    for p, mz in enumerate(proc_spec.peaks.mz):
        bars[int(mz)] = p_inten[p]

        # distinguish top_k peaks by color
    hue = np.repeat("detected", max_mz)
    for mz in missing_mz:
        hue[int(mz)] = f"little"
    for mz in TP_mz:
        hue[int(mz)] = f"TP"
    for mz in FP_mz:
        hue[int(mz)] = f"FP"
    for mz in FN_mz:
        hue[int(mz)] = f"FN"

    palette = {"FP": "darkorange", "FN": "crimson", "TP": "limegreen",
               "little": "gainsboro", "detected": "tab:blue"}

    # get focus area with np.quantile and nasty hack
    meta = []
    for mz, intensity in proc_spec.peaks.to_numpy:
        meta += [mz] * int(intensity * 1000)
    focus = max(0, np.quantile(meta, 0.05) - 25), np.quantile(meta, 0.95) + 25

    # plot focused
    f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
    ax1 = sns.barplot(x=base, y=bars, ax=ax1, hue=hue, dodge=False, palette=palette)

    change_width(ax1, .75)

    # dots on top of FP bars
    dots = np.zeros_like(bars[hue == "FP"])
    dots = np.repeat(np.arange(5), len(bars[hue == "FP"])) / 40
    # np.zeros_like(bars[hue=="FP"])

    # dots[hue=="FP"] 
    mz_FP = np.tile(np.nonzero([hue == "FP"])[1], 5)
    print(len(dots))
    print(len(mz_FP))
    print(len(np.tile(hue[hue == "FP"], 5)))
    sns.scatterplot(x=mz_FP, y=dots, ax=ax1, hue=np.tile(hue[hue == "FP"], 5), palette=palette, legend=False,
                    marker="x", s=10 * 200 / (focus[1] - focus[0]))

    plt.sca(ax1)

    ax1.title.set_text(f"{proc_spec.metadata['name']}")
    ax1.set_xlim(*focus)
    ax1.xaxis.set_tick_params(labelbottom=True)
    ax1.set(ylabel='intensity')
    ax1.set(xlabel='m/z')

    plt.sca(ax1)
    plt.legend(loc='upper left')
    plt.xticks(ticks=np.arange((focus[0] // 10) * 10, (focus[1] // 10) * 10, 20))
    if save_to_path is not None:
        plt.savefig(save_to_path, bbox_inches='tight')
    plt.show()
