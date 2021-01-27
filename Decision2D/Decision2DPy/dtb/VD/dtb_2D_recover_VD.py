"""
Recover models (of different buf_dur, in particular)
"""

#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

from typing import Dict, Union, Any, Tuple, Iterable, Sequence
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
from collections import OrderedDict as odict
from importlib import reload
import shutil, os
from copy import copy, deepcopy
from brokenaxes import brokenaxes

import torch

import dtb.dtb_1D_sim
from lib.pylabyk import localfile, np2, plt2, argsutil, yktorch as ykt
import lib.pylabyk.numpytorch as npt
from lib.pylabyk.numpytorch import npy

npt.device0 = torch.device('cpu')  # enforce cpu
ykt.default_device = torch.device('cpu')

from data_2d import load_data, consts
from dtb.VD import dtb_2D_sim_VD as vd2d, dtb_2D_fit_VD as vdfit

locfile = localfile.LocalFile(
    pth_root='../../data/Data_2D_Py/dtb/VD_model_recovery',
    cache_dir=''
)

bufdurs0 = vdfit.bufdurs0

n_seed = 12

def main():
    fix_post = vdfit.fix0_post

    # === Fit
    for seed in np.arange(n_seed):
        for bufdur in bufdurs0:
            for subj in ['S4', 'S5']:
                losses = main_fit(
                    subjs=[subj],
                    bufdur=bufdur,
                    bufdur_best=get_bufdur_best(subj),
                    seed_sims=[seed],
                    skip_fit_if_absent=False,
                    fix_post=fix_post,
                    plot_kind='None'
                )

    # === For plotting
    losses = main_fit(
        seed_sims=np.arange(n_seed),
        subjs=['S4', 'S5'],
        fix_post=fix_post,
        skip_fit_if_absent=True,
        bufdurs_sim=bufdurs0,
        bufdurs_fit=bufdurs0,
    )


def get_bufdur_best(subj: str) -> float:
    return 0.08


def main_fit(
        subjs=None,
        skip_fit_if_absent=False,
        bufdur=None,
        bufdur_best=0.12,
        bufdurs_sim=None,
        bufdurs_fit=None,
        loss_kind='NLL',
        plot_kind='line_sim_fit',
        fix_post=None,
        seed_sims=(0,),
        err_kind='std',
        base=10,
):
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.double)

    dict_res_add = {}

    parad = 'VD'
    seed_sims = np.array(seed_sims)

    if subjs is None:
        subjs = consts.SUBJS_VD

    if bufdur is not None:
        bufdurs_sim = list({bufdur_best}.union({bufdur}))
        bufdurs_fit = list({bufdur_best}.union({bufdur}))
    else:
        if bufdurs_sim is None:
            bufdurs_sim = bufdurs0  #
            if bufdurs_fit is None:
                bufdurs_fit = bufdurs0
        else:
            if bufdurs_fit is None:
                bufdurs_fit0 = [0., bufdur_best, 1.2]
                if bufdurs_sim[0] in bufdurs_fit0:
                    bufdurs_fit = copy(bufdurs_fit0)
                else:
                    bufdurs_fit = bufdurs_fit0[:2] + bufdurs_sim + [1.2]

    n_dur_sim = len(bufdurs_sim)
    n_dur_fit = len(bufdurs_fit)

    # DEF: losses[seed, subj, simSerDurPar, fitDurs]
    size_all = [
        len(seed_sims), len(subjs), n_dur_sim, n_dur_fit
    ]
    losses0 = np.zeros(size_all) + np.nan
    ns = np.zeros(size_all) + np.nan
    ks0 = np.zeros(size_all) + np.nan

    for i_seed, seed_sim in enumerate(seed_sims):
        for i_subj, subj in enumerate(subjs):
            for i_fit, bufdur_fit in enumerate(bufdurs_fit):
                for i_sim, bufdur_sim in enumerate(bufdurs_sim):
                    d, dict_fit_sim, dict_subdir_sim = get_fit_sim(
                        subj, seed_sim, bufdur_sim, bufdur_fit,
                        parad=parad,
                        skip_fit_if_absent=skip_fit_if_absent,
                        fix_post=fix_post,
                    )
                    if d is not None:
                        losses0[i_seed, i_subj, i_sim, i_fit] = d[
                            'loss_NLL_test']
                        ns[i_seed, i_subj, i_sim, i_fit] = d['loss_ndata_test']
                        ks0[i_seed, i_subj, i_sim, i_fit] = d['loss_nparam']
    ks = copy(ks0)

    if loss_kind == 'BIC':
        losses = ks * np.log(ns) + 2 * losses0

        # REF: Kass, Raftery 1995 https://doi.org/10.2307%2F2291091
        #   https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
        thres_strong = 10. / np.log(base)
    elif loss_kind == 'NLL':
        losses = copy(losses0)
        thres_strong = np.log(100) / np.log(base)
    else:
        raise ValueError('Unsupported loss_kind: %s' % loss_kind)

    #%%
    losses = losses / np.log(base)

    if plot_kind == 'loss_serpar':
        n_row = len(subjs)
        n_col = 1
        axs = plt2.GridAxes(
            n_row, n_col,
            left=1, widths=2, bottom=0.75
        )
        for i_subj, subj in enumerate(subjs):
            plt.sca(axs[i_subj, 0])

            loss_dur_sim_ser_fit = losses[:, i_subj, :, 0].mean(0)
            loss_dur_sim_par_fit = losses[:, i_subj, :, -1].mean(0)

            dloss_serpar = loss_dur_sim_par_fit - loss_dur_sim_ser_fit

            plt.bar(bufdurs_sim, dloss_serpar, color='k')
            plt.axhline(0, color='k', linestyle='-', linewidth=0.5)

            if i_subj < len(subjs) - 1:
                plt2.box_off(['top', 'right', 'bottom'])
                plt.xticks([])
            else:
                plt2.box_off()
                plt.xticks(
                    np.arange(0, 1.4, 0.2),
                    ['0\nser'] + [''] * 2 + ['0.6'] + [''] * 2 + ['1.2\npar']
                )
                plt2.detach_axis('x', 0, 1.2)
            vdfit.patch_chance_level()

        plt.sca(axs[0, 0])
        plt.title('Support for Serial\n'
                  '($\mathcal{L}_\mathrm{ser} - \mathcal{L}_\mathrm{par}$)')

        plt.sca(axs[-1, 0])
        plt.xlabel('true buffer duration (s)')

        plt2.rowtitle(consts.SUBJS_VD, axs)
    elif plot_kind == 'imshow_ser_buf_par':
        n_row = len(subjs)
        n_col = 1
        axs = plt2.GridAxes(
            n_row, n_col,
            left=1, widths=2, heights=2, bottom=0.75
        )
        for i_subj, subj in enumerate(subjs):
            plt.sca(axs[i_subj, 0])

            dloss = losses[:, i_subj, :, :].mean(0)
            dloss = dloss - np.diag(dloss)[:, None]

            plt.imshow(dloss, cmap='bwr')

            cmax = np.amax(np.abs(dloss))
            plt.clim(-cmax, +cmax)

        plt.sca(axs[0, 0])

        plt.sca(axs[-1, 0])
        plt.xlabel('true buffer duration (s)')
        plt.ylabel('model buffer duration (s)')

        plt2.rowtitle(
            consts.SUBJS_VD,
            axs)

    elif plot_kind == 'line_sim_fit':
        bufdur_bests = np.array([get_bufdur_best(subj)
                                 for subj in subjs])
        i_bests = np.array([
            list(bufdurs_fit).index(bufdur_best)
            for bufdur_best in bufdur_bests
        ])
        i_subjs = np.arange(len(subjs))
        loss_sim_best_fit_best = losses[:, i_subjs, i_bests, i_bests]
        loss_sim_best_fit_rest = (
            losses[:, i_subjs, i_bests, :] - loss_sim_best_fit_best[:, :, None])
        mean_loss_sim_best_fit_rest = np.mean(loss_sim_best_fit_rest, 0)
        if err_kind == 'std':
            err_loss_sim_best_fit_rest = np.std(loss_sim_best_fit_rest, 0)
        elif err_kind == 'sem':
            err_loss_sim_best_fit_rest = np2.sem(loss_sim_best_fit_rest, 0)

        n_dur = len(bufdurs_fit)
        loss_sim_rest_fit_best = np.swapaxes(np.stack([
            losses[:, i_subj, np.arange(n_dur), i_best]
            - losses[:, i_subj, np.arange(n_dur), np.arange(n_dur)]
            for i_subj, i_best in zip(i_subjs, i_bests)
        ]), 0, 1)
        mean_loss_sim_rest_fit_best = np.mean(loss_sim_rest_fit_best, 0)
        if err_kind == 'std':
            err_loss_sim_rest_fit_best = np.std(loss_sim_rest_fit_best, 0)
        elif err_kind == 'sem':
            err_loss_sim_rest_fit_best = np2.sem(loss_sim_rest_fit_best, 0)

        dict_res_add['err'] = err_kind

        n_row = 2
        n_col = len(subjs)

        axs = plt2.GridAxes(
            n_row, n_col,
            left=1.15, right=0.1,
            widths=2,
            heights=1.5, wspace=0.35, hspace=0.5,
            top=0.25, bottom=0.6
        )

        for row, (m, s, xlabel, ylabel) in enumerate([(
                mean_loss_sim_best_fit_rest,
                err_loss_sim_best_fit_rest,
                'model buffer capacity (s)',
                (r'$-\mathrm{log}_{%g}\mathrm{BF}$ given simulated'
                 + '\nbest duration data'
                 ) % base
            ), (
                mean_loss_sim_rest_fit_best,
                err_loss_sim_rest_fit_best,
                'simulated data buffer capacity (s)',
                (r'$-\mathrm{log}_{%g}\mathrm{BF}$ of the'
                 + '\nbest duration model'
                ) % base
        )]):
            for i_subj, subj in enumerate(subjs):
                ax = axs[row, i_subj]
                plt.sca(ax)
                gs1 = axs.gs[row * 2 + 1, i_subj * 2 + 1]

                bax = vdfit.breakaxis(gs1)
                bax.axs[1].errorbar(
                    bufdurs0, m[i_subj, :], yerr=s[i_subj, :],
                    color='k', marker='.',
                    linewidth=0.75, elinewidth=0.5,
                    markersize=3
                )
                m1 = copy(m[i_subj, :])
                m1[3:] = np.nan
                s1 = copy(s[i_subj, :])
                s1[3:] = np.nan
                bax.axs[0].errorbar(
                    bufdurs0, m1, yerr=s1,
                    color='k', marker='.',
                    linewidth=0.75, elinewidth=0.5,
                    markersize=3
                )

                ax1 = bax.axs[1]  # type: plt.Axes
                plt.sca(ax1)
                vdfit.patch_chance_level(
                    level=thres_strong, signs=[-1, 1])
                plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
                vdfit.beautify_ticks(
                    ax1,
                )
                vdfit.beautify(ax1)
                ax1.set_yticks([0, 20])
                if i_subj > 0:
                    ax1.set_yticklabels([])
                if row == 0:
                    ax1.set_xticklabels([])

                ax1 = bax.axs[0]   # type: plt.Axes
                plt.sca(ax1)
                ax1.set_yticks([40, 200])
                if i_subj > 0:
                    ax1.set_yticklabels([])

                plt.sca(ax)
                plt2.box_off('all')
                if row == 0:
                    plt.title(consts.SUBJS['VD'][i_subj])

                if i_subj == 0:
                    plt.xlabel(xlabel, labelpad=8 if row == 0 else 20)
                    plt.ylabel(ylabel, labelpad=30)

                plt2.sameaxes(bax.axs, xy='x')

    elif plot_kind == 'bar_sim_fit':
        i_best = bufdurs_fit.index(bufdur_best)
        loss_sim_best_fit_best = losses[0, :, i_best, i_best]
        loss_sim_best_fit_rest = np2.nan2v(
                losses[0, :, i_best, :] - loss_sim_best_fit_best[:, None])
        n_dur = len(bufdurs_fit)
        loss_sim_rest_fit_best = np2.nan2v(
                losses[0, :, np.arange(n_dur), i_best]
                - losses[0, :, np.arange(n_dur), np.arange(n_dur)]).T

        n_row = 2
        n_col = len(subjs)

        axs = plt2.GridAxes(
            n_row, n_col,
            left=1, widths=3, right=0.25,
            heights=1, wspace=0.3, hspace=0.6,
            top=0.25, bottom=0.6
        )
        for i_subj, subj in enumerate(subjs):
            def plot_bars(gs1, bufdurs, losses1, add_xticklabel=True):
                i_break = np.amax(np.nonzero(np.array(bufdurs) < 0.3)[0])
                bax = brokenaxes(
                    subplot_spec=gs1,
                    xlims=((-1, i_break + 0.5),
                           (i_break + 0.5, len(bufdurs) - 0.5)),
                    ylims=((-3, 20), (20, 1250/5)),
                    height_ratios=(50 / 100, 500 / (1250 - 100)),
                    hspace=0.15,
                    wspace=0.075,
                    d=0.005,
                )
                bax.bar(np.arange(len(bufdurs)),
                        losses1[i_subj, :], color='k')

                ax11 = bax.axs[3]  # type: plt.Axes
                ax11.set_xticks([bufdurs.index(0.6), bufdurs.index(1.2)])
                if i_subj == 0 and add_xticklabel:
                    ax11.set_xticklabels(['0.6', '1.2'])
                else:
                    ax11.set_xticklabels([])

                ax00 = bax.axs[0]  # type: plt.Axes
                ax00.set_yticks([500, 1000])

                ax10 = bax.axs[2]  # type: plt.Axes
                ax10.set_yticks([0, 50])
                plt.sca(ax10)
                plt2.detach_axis('x', amin=-0.4, amax=i_break + 0.5)
                for ax in [ax10, ax11]:
                    plt.sca(ax)
                    plt.axhline(0,
                                linewidth=0.5, color='k',
                                linestyle='--')
                    for sign in [-1, 1]:
                        plt.axhline(sign * thres_strong,
                                    linewidth=0.5, color='silver',
                                    linestyle='--')
                ax10.set_xticks([bufdurs.index(0.), bufdurs.index(0.2)])
                if i_subj == 0:
                    if add_xticklabel:
                        ax10.set_xticklabels(['0', '0.2'])
                    else:
                        ax10.set_xticklabels([])
                else:
                    ax10.set_yticklabels([])
                    ax10.set_xticklabels([])
                    ax00.set_yticklabels([])
                return bax
            bax = plot_bars(
                axs.gs[1, i_subj * 2 + 1],
                bufdurs_fit, loss_sim_best_fit_rest,
                add_xticklabel=False
            )

            ax = axs[0, i_subj]
            plt.sca(ax)
            plt2.box_off('all')
            plt.title(
                consts.SUBJS['VD'][consts.SUBJS['VD'].index(subj)])
            if i_subj == 0:
                ax.set_ylabel('misfit to simulated\nbest duration data\n'
                              r'($\Delta$BIC)',
                              labelpad=35)
                ax.set_xlabel('model buffer duration (s)',
                              labelpad=8)

            plot_bars(
                axs.gs[3, i_subj * 2 + 1],
                bufdurs_sim, loss_sim_rest_fit_best,
                add_xticklabel=True
            )
            ax = axs[1, i_subj]
            plt.sca(ax)
            plt2.box_off('all')
            if i_subj == 0:
                ax.set_ylabel('misfit of\nbest duration model\n'
                              r'($\Delta$BIC)',
                              labelpad=35)
                ax.set_xlabel('simulated data buffer duration (s)',
                              labelpad=20)

    elif plot_kind == 'bar_ser_buf_par':
        n_row = len(subjs)
        n_col = 1
        axs = plt2.GridAxes(
            n_row * n_dur_sim, n_col,
            left=1.25, widths=1.5, right=0.25,
            heights=0.4,
            hspace=[0.15] * (n_dur_sim - 1) +  [0.5]
                   + [0.15] * (n_dur_sim - 1),
            top=0.6, bottom=0.5
        )
        row = -1
        for i_subj, subj in enumerate(subjs):
            for i_sim in range(n_dur_sim):
                row += 1
                plt.sca(axs[row, 0])

                loss1 = losses[:, i_subj, i_sim, :].mean(0)
                dloss = loss1 - loss1[i_sim]

                x = np.arange(n_dur_fit)
                for x1, dloss1 in zip(x, dloss):
                    plt.bar(x1, dloss1,
                            color='r' if dloss1 > 0 else 'b')
                plt.axhline(0, color='k', linewidth=0.5)
                vdfit.patch_chance_level(6.)
                plt.ylim([-100, 100])
                plt.yticks([-100, 0, 100], [''] * 2)
                plt.ylabel(
                    '%g' % bufdurs_sim[i_sim],
                    rotation=0,
                    va='center',
                    ha='right'
                )

                if i_sim == 0:
                    plt.title(subj)

                if i_sim < n_dur_sim - 1 or i_subj < len(subjs) - 1:
                    plt2.box_off(['top', 'bottom', 'right'])
                    plt.xticks([])
                else:
                    plt2.box_off(['top', 'right'])

        plt.sca(axs[-1, 0])
        plt.xticks(np.arange(n_dur_sim), ['%g' % v for v in bufdurs_sim])
        plt2.detach_axis('x', 0, n_dur_sim - 1)
        plt.xlabel('model buffer duration (s)', fontsize=10)

        c = axs[-1, 0].get_position().corners()
        plt.figtext(
            x=0.15, y=np.mean(c[:2, 1]), fontsize=10,
            s='true\nbuffer\nduration\n(s)',
            rotation=0, va='center', ha='center')

        plt.figtext(
            x=(1.25 + 1.5 / 2) / (1.25 + 1.5 + 0.25),
            y=0.98,
            s=r'$\mathrm{BIC} - \mathrm{BIC}_\mathrm{true}$',
            ha='center', va='top',
            fontsize=12,
        )

    if plot_kind != 'None':
        dict_res = deepcopy(dict_fit_sim)  # noqa
        for k in ['sbj', 'prd']:
            dict_res.pop(k)
        dict_res = {**dict_res, 'los': loss_kind}
        dict_res.update(dict_res_add)
        for ext in ['.png', '.pdf']:
            file = locfile.get_file_fig(plot_kind, dict_res,
                                        subdir=dict_subdir_sim,
                                        ext=ext
                                        )  # noqa
            plt.savefig(file, dpi=300)
            print('Saved to %s' % file)

    #%%
    print('--')
    return losses


def subj2subjnew(subj):
    return consts.SUBJS['VD'][consts.SUBJS['VD'].index(subj)]


def get_fit_sim(
        subj: str,
        seed_sim: int, bufdur_sim: float, bufdur_fit: float,
        parad='VD',
        skip_fit_if_absent=False,
        fix_post=None,
) -> (dict, dict, dict):
    """

    :param subj:
    :param seed_sim:
    :param bufdur_sim:
    :param bufdur_fit:
    :param parad:
    :param fix_post:
    :return: d, dict_fit_sim, dict_subdir_sim
    """

    if fix_post is None:
        fix_post = (
            'basym0_fix',
            'bhalf042_lb01',
            'diffix',
            'lps0'
        )

    # --- Load fit to simulated data
    def remove_buffix(fix_strs):
        return [s for s in fix_strs
                if not s.startswith('(buffix')]

    _, dict_cache, dict_subdir = vdfit.init_model(
        subj=subj, bufdur=bufdur_sim, parad=parad,
        fix_post=fix_post
    )
    _, dict_subdir_sim = vdfit.get_subdir(
        fix_strs=remove_buffix(dict_subdir['fix']),
        **dict_subdir
    )
    dict_sim = {
        **dict_cache,
        'fix': remove_buffix(dict_cache['fix']),
        'bufdur_sim': bufdur_sim,
        'seed_sim': seed_sim
    }
    dict_fit_sim = {
        **dict_sim,
        'bufdur_fit': bufdur_fit
    }
    cache_fit_sim = locfile.get_cache(
        'fit_sim', dict_fit_sim, subdir=dict_subdir_sim)
    if cache_fit_sim.exists():
        best_state, d = cache_fit_sim.getdict([
            'best_state', 'd'
        ])
    elif skip_fit_if_absent:
        return None, dict_fit_sim, dict_subdir_sim
    else:
        # --- Load model fit to real data
        _, data, dict_cache, d, subdir = vdfit.load_fit(
            subj=subj, bufdur=bufdur_sim,
            fix_post=fix_post,
            skip_fit_if_absent=skip_fit_if_absent
        )

        # --- Simulate new data and save
        data_sim = deepcopy(data)  # type: vd2d.Data2DVD
        cache_data_sim = locfile.get_cache(
            'data_sim', dict_sim, subdir=dict_subdir_sim)
        if cache_data_sim.exists():
            data_sim.update_data(
                ch_tr_dim=
                cache_data_sim.getdict(['chSim_tr_dim'])[0])
        else:
            ch_tr_dim_bef = data_sim.ch_tr_dim.copy()
            data_bef = npy(data_sim.n_cond_dur_ch).copy()

            data_sim.simulate_data(
                pPred_cond_dur_ch=d['out_all'],
                seed=seed_sim)

            ch_tr_dim_aft = data_sim.ch_tr_dim.copy()
            data_aft = npy(data_sim.n_cond_dur_ch).copy()

            print('Proportion of trials with the same choice:')
            print(np.mean(ch_tr_dim_bef == ch_tr_dim_aft))

            cache_data_sim.set({
                'chSim_tr_dim': data_sim.ch_tr_dim
            })
            cache_data_sim.save()
        del cache_data_sim

        # --- Fit simulated data
        model, dict_cache, dict_subdir = vdfit.init_model(
            subj=subj, bufdur=bufdur_fit,
            fix_post=fix_post
        )
        _, best_state, d, plotfuns = vd2d.fit_dtb(
            model, data_sim,
            comment='+' + argsutil.dict2fname(dict_fit_sim),
            max_epoch=vdfit.max_epoch0,
        )
        dtb.dtb_1D_sim.save_fit_results(
            model=model, best_state=best_state,
            d=d, plotfuns=plotfuns,
            locfile=locfile,
            dict_cache=dict_fit_sim, subdir=dict_subdir_sim
        )
        cache_fit_sim.set({
            'best_state': best_state,
            'd': {k: v for k, v in d.items()
                  if k.startswith('loss_')}
        })
        cache_fit_sim.save()
    del cache_fit_sim
    return d, dict_fit_sim, dict_subdir_sim


def ____Main____():
    pass


if __name__ == '__main__':
    main()