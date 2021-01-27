#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

from typing import Union, Iterable
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
from copy import deepcopy
from scipy.io import loadmat
import os

import torch

from lib.pylabyk.numpytorch import npy, npys
from lib.pylabyk import localfile, argsutil, np2, plt2

from data_2d import consts, load_data
from dtb import dtb_1D_sim as sim1d, dtb_2D_sim as sim2d
from dtb.RT import dtb_2D_fit_RT as fit2d
from dtb.dtb_1D_sim import save_fit_results


locfile = localfile.LocalFile(
    '../../data/Data_2D_Py/dtb/RTNonparamMATLAB')


def main():
    for to_use_easiest_only in [0, 1]:
        main_plot(to_use_easiest_only=to_use_easiest_only)


def main_plot(to_use_easiest_only=0):
    # === Mean RT
    plot_kind = 'rt_mean_row_per_model'
    axs = plt2.GridAxes(
        4, 4,
        top=0.6, left=0.7, right=0.7, bottom=0.6,
        wspace=[0.8, 1.6, 0.8],
        hspace=[0.1, 1.2, 0.1],
        heights=1.5, widths=1.7
    )
    inds = ['A', 'B', 'C']
    parads = ['RT', 'unimanual', 'bimanual']
    rowcol_toplefts = [(0, 0), (0, 2), (2, 0)]
    fit_kinds = ['Fit all', 'Fit strongest,\npredict rest']
    models = ['Ser', 'Par']
    model_names = ['serial', 'parallel']
    n_dim = 2

    for rowcol_topleft, parad, ind in zip(rowcol_toplefts, parads, inds):
        for row_shift, model in enumerate(models):
            row = rowcol_topleft[0] + row_shift
            col = rowcol_topleft[1]
            axs1 = axs[row:(row+1), col:(col+n_dim)]
            plot_rt(
                parad,
                to_cumsum=False,
                corners_only=False,
                plot_kind='rt_mean',
                model_names=[model],
                use_easiest_only=to_use_easiest_only,
                axs=axs1,
                to_add_legend=row_shift == 0
            )

            if row_shift == 0:
                for ax in axs1.flatten():
                    ax.set_xticklabels([])
                    ax.set_xlabel('')
                axs1.suptitle(
                    get_title_parad(parad), pad=0.05, va='bottom',
                    xprop=0.55
                )
                axs1.suptitle(ind, pad=0.05, xprop=-0.05, va='bottom',
                              fontweight='bold')

            axs1[0, 0].set_ylabel('Response time (s)\n(%s model)' %
                                  model_names[row_shift])

    for ax in axs[2:, 2:].flatten():
        ax.set_visible(False)

    for ext in ['.png', '.pdf']:
        file = locfile.get_file_fig(plot_kind, {
            'parad': parads,
            'easiest_only': to_use_easiest_only,
        }, ext=ext)
        localfile.mkdir4file(file)
        plt.savefig(file)
        print('Saved to %s' % file)

    # === RT distrib
    plot_kind = 'rt_distrib'
    to_normalize_max = False
    for to_cumsum in [True]:  # [False, True]:
        for corners_only in [True]: # , False]:
            for parad in ['RT', 'unimanual', 'bimanual']:
                axs = plot_rt(
                    parad,
                    to_cumsum=to_cumsum,
                    corners_only=corners_only,
                    plot_kind=plot_kind,
                    use_easiest_only=to_use_easiest_only,
                    to_add_legend=parad == 'RT',
                )
                axs.suptitle(get_title_parad(parad), pad=0.25)
                for ext in ['.png', '.pdf']:
                    file = locfile.get_file_fig(plot_kind, {
                        'parad': parad,
                        'cumsum': to_cumsum,
                        'corner': corners_only,
                        'easiest_only': to_use_easiest_only,
                        'nrmmax': to_normalize_max,
                    }, ext=ext)
                    localfile.mkdir4file(file)
                    plt.savefig(file)
                    print('Saved to %s' % file)

    print('--')


def plot_rt(
        parad, to_cumsum=False,
        corners_only=False,
        to_add_legend=True,
        use_easiest_only=0,
        plot_kind='rt_mean',
        model_names=('Ser', 'Par'),
        to_normalize_max=False,
        axs=None,
):
    """

    :param parad:
    :param to_cumsum:
    :param corners_only:
    :param to_add_legend:
    :param use_easiest_only:
    :param plot_kind: 'mean'|'distrib'
    :return:
    """
    subjs = consts.SUBJS[parad]
    n_models = len(model_names)

    p_preds = np.empty([n_models, len(subjs)], dtype=np.object)
    p_datas = np.empty([n_models, len(subjs)], dtype=np.object)
    condss = np.empty([n_models, len(subjs)], dtype=np.object)
    ts = np.empty([n_models, len(subjs)], dtype=np.object)

    p_pred_trains = np.empty([n_models, len(subjs)], dtype=np.object)
    p_data_trains = np.empty([n_models, len(subjs)], dtype=np.object)
    p_pred_tests = np.empty([n_models, len(subjs)], dtype=np.object)
    p_data_tests = np.empty([n_models, len(subjs)], dtype=np.object)

    colors = ('red', 'blue')
    normalize_ev = parad == 'RT'

    for i_model, model_name in enumerate(model_names):
        for i_subj, subj in enumerate(subjs):
            (
                p_pred1, p_data1, conds1, t1, cond_ch_incl1,
                p_pred_train, p_data_train, p_pred_test, p_data_test,
            ) = load_fit(
                    subj, parad, model_name,
                    corners_only=corners_only,
                    use_easiest_only=use_easiest_only
                )
            ix_abs_cond = np.unique(np.abs(conds1[1]), return_inverse=True)[1]
            sign_cond = np.sign(conds1[1])

            if normalize_ev:
                conds1 = conds1 / np.amax(conds1, axis=1, keepdims=True)

            p_preds[i_model, i_subj], p_datas[i_model, i_subj], \
                condss[i_model, i_subj], ts[i_model, i_subj] \
                = p_pred1, p_data1, conds1, t1

            (
                p_pred_trains[i_model, i_subj], p_data_trains[i_model, i_subj],
                p_pred_tests[i_model, i_subj], p_data_tests[i_model, i_subj],
            ) = p_pred_train, p_data_train, p_pred_test, p_data_test

    siz0 = list(p_preds[0, 0].shape)

    def cell2array(p):
        return np.stack(p.flatten()).reshape(
            [len(model_names), len(subjs)] + siz0)

    p_preds = cell2array(p_preds)
    p_datas = cell2array(p_datas)

    p_pred_trains = cell2array(p_pred_trains)
    p_data_trains = cell2array(p_data_trains)

    p_pred_tests = cell2array(p_pred_tests)
    p_data_tests = cell2array(p_data_tests)

    def pool_subjs(p_datas, p_preds):
        n_data_subj = np.sum(p_datas, (2, 3, 4), keepdims=True)

        # P(ch, rt | cond, subj, model)
        p_preds = np2.nan2v(p_preds / p_preds.sum((3, 4), keepdims=True))
        p_pred_avg = np.sum(p_preds * n_data_subj, 1)
        n_data_sum = np.sum(p_datas, 1)
        return p_pred_avg, n_data_sum

    p_pred_avg, n_data_sum = pool_subjs(p_datas, p_preds)
    p_pred_avg_train, n_data_sum_train = pool_subjs(p_data_trains,
                                                    p_pred_trains)
    p_pred_avg_test, n_data_sum_test = pool_subjs(p_data_tests, p_pred_tests)

    ev_cond_dim = np.stack(condss[0, 0], -1)
    dt = 1 / 75

    if plot_kind == 'rt_mean':
        if axs is None:
            axs = plt2.GridAxes(
                1, 2,
                top=0.4, left=0.6, right=0.1, bottom=0.65,
                wspace=0.35, heights=1.7, widths=2.2
            )
        for i_model, model_name in enumerate(model_names):
            hs = fit2d.plot_fit_combined(
                pAll_cond_rt_chFlat=n_data_sum[i_model],
                evAll_cond_dim=ev_cond_dim,
                pTrain_cond_rt_chFlat=n_data_sum_train[i_model],
                pTest_cond_rt_chFlat=n_data_sum_test[i_model],
                pModel_cond_rt_chFlat=p_pred_avg[i_model],
                dt=dt,
                to_plot_params=False,
                to_plot_internals=False,
                to_plot_choice=False,
                group_dcond_irr=None,
                kw_plot_pred={
                    # 'linestyle': ':',
                    # 'alpha': 0.7,
                    # 'linewidth': 2,
                    # 'linestyle': '--' if model_name == 'Par' else '-',
                },
                kw_plot_data={
                    'markersize': 4,
                },
                axs=axs,
            )[2]

            if to_add_legend:
                n_dim = 2
                for dim in range(n_dim):
                    plt.sca(axs[0, dim])
                    odim = consts.get_odim(dim)
                    conds_irr = np.unique(np.abs(ev_cond_dim[:, odim]))
                    hs1 = [v[0][0] for v in hs['rt'][dim]]
                    hs1 = hs1[len(conds_irr):(len(conds_irr) * 2)]

                    h = fit2d.legend_odim(
                        [np.round(v, 3) for v in conds_irr], hs1,
                        '',
                        loc='lower left',
                        bbox_to_anchor=[1., 0.]
                    )
                    h.set_title(
                        consts.DIM_NAMES_LONG[odim] + '\nstrength'
                    )
                    plt.setp(h.get_title(), multialignment='center')

        ev_max = np.amax(condss[0, 0], -1)
        for col in [0, 1]:
            plt.sca(axs[-1, col])
            txt = '%s strength' % consts.DIM_NAMES_LONG[col].lower()
            if normalize_ev:
                txt += '\n(a.u.)'
            plt.xlabel(txt)
            xticks = [-ev_max[col], 0, ev_max[col]]
            plt.xticks(xticks, ['%g' % v for v in xticks])

        from matplotlib.ticker import MultipleLocator
        axs[0, 0].yaxis.set_major_locator(MultipleLocator(1))
        print('--')

    elif plot_kind == 'rt_distrib':
        axs, hs = fit2d.plot_rt_distrib_pred_data(
            p_pred_avg, n_data_sum[0], ev_cond_dim, dt,
            to_normalize_max=to_normalize_max,
            to_cumsum=to_cumsum,
            to_skip_zero_trials=True,
            xlim=[0., 5.],
            kw_plot_data={
                'linewidth': 2,
                'linestyle': ':',
                'alpha': 0.75,
            },
            labels=['serial', 'parallel', 'data'],
            colors=colors,
        )
        if to_add_legend:
            plt.sca(axs[0, 0])
            locs = {
                'loc': 'center right',
                'bbox_to_anchor': (1.05, 0., 0., 1.)
            } if to_cumsum else {
                'loc': 'upper right',
                'bbox_to_anchor': (1.05, 1.01)
            }
            plt.legend(
                **locs,
                handlelength=0.8,
                handletextpad=0.5,
                frameon=False,
                borderpad=0.
            )
        print('--')

    else:
        raise ValueError()

    return axs


def get_title_parad(parad):
    return 'Eye' if parad == 'RT' else parad.capitalize()


def load_fit(subj: str, parad: str, model_name: str,
             corners_only=False,
             use_easiest_only=0,
             ):
    """

    :param subj:
    :param parad:
    :param model_name:
    :return: p_pred[cond, rt, ch], p_data[cond, rt, ch], conds[dim][cond], t
    """
    pth_in = '../../data/Fit.D2.RT.Td2Tnd.Main'
    nam = (
        'export=pred_data+sbj=%s+prd=%s+tsk=A+dtk=2+dmr=1'
        '+trm=1+eor=0+msf=1+ef=%d+ec=-%d+lf=0+eb=10'
        '+td=%s+smr=NaN+um=2+us=2'
        % (subj, parad, use_easiest_only, use_easiest_only, model_name)
    )
    file = os.path.join(pth_in, nam + '.mat')
    dat = loadmat(file)
    print('Loaded %s' % file)

    t = dat['t'].flatten()
    nt = len(t)

    conds = [v.flatten() for v in dat['conds'].flatten()]
    cond0, cond1 = list(np.meshgrid(*conds, indexing='ij'))
    conds = np.stack([cond0.flatten(), cond1.flatten()])
    n_cond = len(conds[0])
    n_ch = 4

    p_pred = dat['pred'].reshape([nt, n_cond, n_ch]).transpose([1, 0, 2])
    p_data = dat['data'].reshape([nt, n_cond, n_ch]).transpose([1, 0, 2])
    cond_ch_to_incl = dat['cond_ch_to_incl'].reshape([n_cond, n_ch])

    try:
        cond_ch_to_incl_train = dat['cond_ch_to_incl_train'].reshape([n_cond, n_ch])
        cond_ch_to_incl_test = dat['cond_ch_to_incl_test'].reshape([n_cond, n_ch])
    except KeyError:
        cond_ch_to_incl_train = cond_ch_to_incl
        cond_ch_to_incl_test = cond_ch_to_incl

    # if to_filter_cond_ch:
    p_pred = p_pred * cond_ch_to_incl[:, None, :]
    p_data = p_data * cond_ch_to_incl[:, None, :]

    p_pred_train = p_pred * cond_ch_to_incl_train[:, None, :]
    p_data_train = p_data * cond_ch_to_incl_train[:, None, :]

    p_pred_test = p_pred * cond_ch_to_incl_test[:, None, :]
    p_data_test = p_data * cond_ch_to_incl_test[:, None, :]

    if corners_only:
        incl = np.all(
            (np.abs(conds) == np.amax(conds, -1, keepdims=True))
            | (conds == 0), 0)
        conds = conds[:, incl]
        cond_ch_to_incl = cond_ch_to_incl[incl]
        p_pred = p_pred[incl]
        p_data = p_data[incl]

        p_pred_train = p_pred_train[incl]
        p_data_train = p_data_train[incl]

        p_pred_test = p_pred_test[incl]
        p_data_test = p_data_test[incl]

    return (
        p_pred, p_data, conds, t, cond_ch_to_incl,
        p_pred_train, p_data_train, p_pred_test, p_data_test
    )


if __name__ == '__main__':
    main()
