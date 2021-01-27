#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

from collections import OrderedDict as odict
from copy import deepcopy
from typing import Union, Type, List, Dict, Iterable, Tuple

import numpy as np
import numpy_groupies as npg
import torch
from matplotlib import pyplot as plt

from dtb import dtb_2D_sim as sim2d
from data_2d import consts
from lib.pylabyk import localfile, np2, plt2

locfile = localfile.LocalFile(
    pth_root='../../data/Data_2D_Py/dtb/RTRecover',
    cache_dir=''
)

subj_parad_bis0 = consts.SUBJ_PARAD_BI


def get_subj_parad_bi_str(subj_parad_bis
                          : Iterable[Tuple[str, str, bool]] = None):
    """

    :param subj_parad_bis: [('subj', 'parad', is_bimanual), ...]
    :return:
    """
    ss = []
    for subj, parad, bimanual in subj_parad_bis:
        if parad in ['RT', 'eye']:
            s = 'eye, %s' % subj
        elif parad == 'bimanual' or bimanual:
            s = 'bimanual, %s' % subj
        elif parad == 'unibimanual' and not bimanual:
            s = 'unimanual, %s' % subj
        else:
            s = '%s, %s' % (parad, subj)

        ss.append(s)
    return ss


parad_bis, ix_parad_bi = np.unique(
    np.stack([v[1:] for v in subj_parad_bis0]), axis=0,
    return_inverse=True)
colors_parad = {
    ('RT', 'False'): 'tab:orange',
    ('unibimanual', 'False'): 'tab:blue',
    ('unibimanual', 'True'): 'tab:cyan',
    ('binary', 'False'): 'plum',
}
labels_parad = {
    ('RT', 'False'): 'eye',
    ('unibimanual', 'False'): 'unimanual',
    ('unibimanual', 'True'): 'bimanual',
    ('binary', 'False'): 'binary',
}


def plot_bar_dloss_across_subjs(
        dlosses, elosses=None, ix_datas=None,
        subj_parad_bis: Iterable[Tuple[str, str, bool]] = None,
        axs: Union[plt2.GridAxes, plt2.AxesArray] = None,
        vmax=None,
        add_scale=True,
        base=10.,
):
    """

    :param dlosses: [ix_data]
    :param ix_datas:
    :param axs:
    :param subj_parad_bis: [('subj', 'parad', is_bimanual), ...]
    :return: axs
    """

    if subj_parad_bis is None:
        subj_parad_bis = subj_parad_bis0
    if vmax is None:
        vmax = np.amax(np.abs(dlosses))

    # order: eye S1-S3, hand by ID, paired uni-bimanual
    subjs, parads, bis = zip(*subj_parad_bis)
    subjs = np.array(['ID0' + v[-1] if v[:2] == 'ID' and len(v) == 3 else v
             for v in subjs])
    parads = np.array(parads)
    bis = np.array(bis)

    is_eye = parads == 'RT'
    is_bin = parads == 'binary'
    ix = np.arange(len(subjs))

    def filt_sort(filt):
        ind = [int(subj[1:]) for subj in subjs[filt]]
        return ix[filt][np.argsort(ind)]

    ix = np.concatenate([
        filt_sort(is_eye & ~is_bin),
        np.stack([filt_sort(~is_eye & ~bis & ~is_bin),
                  filt_sort(~is_eye & bis & ~is_bin)
                  ], -1).flatten('C'),
        filt_sort(is_bin)
    ])
    subjs = subjs[ix]
    parads = parads[ix]
    bis = bis[ix]
    is_eye = is_eye[ix]
    dlosses = dlosses[ix]
    subj_parad_bis = subj_parad_bis[ix]

    n_eye = int(np.sum(is_eye))
    n_hand = int(np.sum(~is_eye))

    y = np.empty([n_eye + n_hand])
    y[is_eye] = 1.5 + np.arange(n_eye)
    y[~is_eye] = n_eye - 1 + 1.5 + np.cumsum([1.5, 1.] * (n_hand // 2))
    y_max = np.amax(y) + 1.5

    if axs is None:
        axs = plt2.GridAxes(
            nrows=1, ncols=1,
            heights=y_max * 0.2,
            widths=2,
            left=1.5, right=0.25,
            bottom=0.85
        )
    ax = axs[0, 0]
    plt.sca(ax)

    m = dlosses
    if elosses is None:
        e = np.zeros_like(m)
    else:
        e = elosses

    for y1, m1, e1, parad1, bi1 in zip(y, m, e, parads, bis):
        plt.barh(y1, m1, xerr=e1,
                 color=colors_parad[(parad1, '%s' % bi1)],
                 edgecolor='None')

    if add_scale:
        dy = y[1] - y[0]

    axvline_dcost()

    x_lim = [-vmax * 1.2, vmax * 1.2]
    for ix_big in range(len(y)):
        if np.abs(m[ix_big]) > vmax:
            for i_sign, sign in enumerate([1, -1]):
                plt2.patch_wave(y[ix_big], x_lim[i_sign] * 1.01,
                                ax=ax, color='w',
                                wave_margin=0.15,
                                wave_amplitude=sign * 0.025,
                                )

    plt.xlim(x_lim)
    xticks_serial_vs_parallel(vmax, base)
    subj_parad_bi_str = get_subj_parad_bi_str(subj_parad_bis)
    plt.yticks(y, subj_parad_bi_str)
    plt2.detach_axis('y', y[0], y[-1])
    plt2.detach_axis('x', -vmax, vmax)
    plt.ylim([y_max - 1, 1.])

    return axs


def xticks_serial_vs_parallel(vmax, base):
    plt.xticks([-vmax, 0, vmax])
    plt.xlabel('support for parallel model\n'
               + r'($\mathrm{log}_{%g}\mathrm{BF}$)' % base)


def axvline_dcost(BF=100., base=10., style='patch'):
    thres = np.log(BF) / np.log(base)
    if style == 'line':
        plt.axvline(0, color='k', linewidth=0.5, linestyle='--', zorder=1)
        for sign in [-1, 1]:
            plt.axvline(sign * thres,
                        color='silver',
                        linewidth=0.5,
                        linestyle='--', zorder=1)
    elif style == 'patch':
        import matplotlib.patches as patches
        ax = plt.gca()
        ax.add_patch(
            patches.Rectangle(
                (-thres, -100),
                thres * 2,
                200,
                edgecolor='None',
                facecolor=[0., 0., 0., 0.4],
                fill=True,
            ))
    else:
        raise ValueError()
    plt2.box_off()


def ____Main____():
    pass


if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.double)