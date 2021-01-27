"""
"""

#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.


from typing import Dict, Union, Any, Tuple, Iterable, List
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pprint import pprint
from collections import OrderedDict as odict
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import torch
from brokenaxes import brokenaxes

from dtb.dtb_1D_sim import save_fit_results
from lib.pylabyk import localfile, np2, plt2, argsutil, yktorch as ykt
from lib.pylabyk import numpytorch as npt
from lib.pylabyk.numpytorch import npy

npt.device0 = torch.device('cpu')
ykt.default_device = torch.device('cpu')

from data_2d import load_data, consts
from dtb.VD import dtb_2D_sim_VD as vd2d
from dtb import dtb_1D_sim as rt1d

locfile = localfile.LocalFile(
    pth_root='../../data/Data_2D_Py/dtb/VD',
    cache_dir=''
)

i_subjs = (0, 1)
fix0_pre = [
    'tdfix0',
    'ssq0',
]
fix0_post = [
    'basym0_fix',
    'bhalf042_lb01',
    'diffix',
    'lps0',
    'noixn',
    'y0'
]
fix0 = fix0_pre + fix0_post

bufdurs0 = [0., 0.04, 0.08, 0.12, 0.16,
            0.20, 0.24, 0.36, 0.48, 0.60,
            0.72, 0.84, 0.96, 1.08, 1.2]
bufdur_best = 0.08

FixType = Iterable[Union[str, Tuple[str, Any]]]

max_epoch0 = 500


def main():
    i_subjs = [0, 1]

    # === Fit
    for buffix in bufdurs0:
        for i_subj in i_subjs:
            model, data, dict_cache, d, subdir = main_fit(
                i_subj=i_subj,
                n_fold_valid=1,
                fix=fix0_pre
                    + [('buffix', buffix)]
                    + fix0_post
            )

    # === Plot
    main_plot_across_models(
        i_subjs=i_subjs
    )

    # === Plot buffer, parallel, and serial models
    for coef in ['slope']:
        main_plot_models_on_columns(coef=coef)


def main_fit(
        i_subj=0, fit_mode='auto',
        parad='VD',
        fix: FixType = None,
        n_fold_valid=1,
        mode_train='all',
        **kwargs
):
    """
    Load data and fit if needed
    :param i_subj:
    :param fit_mode: 'auto'|'new'|'continue'|'load'|'replicate'|'init_model'
      'auto': load if cache exists, new fit otherwise
      'new': new fit ignoring cache
      'continue': continue fitting from cache
      'load': quickly check the bias-slope plot only
      'replicate': reproduce all plots from saved cache
    :param fix:
    :param n_fold_valid:
    :param kwargs: added to kw_optim
    :return: model, data, dict_cache, d, subdir
    """

    assert fit_mode in ['auto', 'new', 'continue', 'load', 'replicate',
                        'dict_cache_only', 'init_model']

    # torch.autograd.detect_anomaly()

    state_dict, fix_params, fix_strs = specify_fixs(fix)
    subdir, dict_subdir = get_subdir(fix_strs, n_fold_valid,
                                     )

    # --- Build model
    subsample_factor = 3
    max_t = 1.5
    nt0 = int(max_t * consts.REFRESH_RATE)

    dt = consts.DT * subsample_factor
    nt = nt0 // subsample_factor

    model = vd2d.FitVD2D(
        dtb2d=vd2d.Dtb2DVDBufSerial,
        bound=rt1d.BoundExp,
        dt=dt, nt=nt,
    )

    state_dict = odict(state_dict)
    state_dict = odict([(k, torch.tensor(v)) for k, v in state_dict.items()])
    model.load_state_dict(state_dict, strict=False)

    for k, param in model.named_parameters():
        if k in fix_params:
            param.requires_grad = False

    # --- Get data by subj and parad
    dict_cache = get_dict_cache(
        subj=consts.SUBJS_VD[i_subj],
        parad=parad,
        ndim=2,
        model=model,
        dict_subdir=dict_subdir,
        mode_train=mode_train,
        n_fold_valid=n_fold_valid
    )

    if fit_mode == 'init_model':
        return model, dict_cache, dict_subdir

    # --- Optimization options
    kw_optim0 = {
        'show_progress_every': 5,
        'patience': 100,
        'max_epoch': 0 if fit_mode in ['load', 'replicate'] else max_epoch0,
        'skip_fit_if_absent': fit_mode in ['load'],
        'learning_rate': 0.05 if fit_mode in ['continue'] else 0.05,
        'ignore_cache': fit_mode in ['new'],
        'to_plot_progress': fit_mode in ['new', 'continue', 'auto'],
        'save_results': None,  # if fit_mode not in ['load'],
        'continue_fit': fit_mode in ['continue'],
        **kwargs
    }

    data = get_data_2D(
        i_subj=i_subj,
        parad='VD',
        trial_st=0,
        subsample_factor=subsample_factor,
        max_t=max_t,
    )

    # --- Run fitting
    out0 = fit_model_internal(
        model=model, data=data,
        dict_cache=dict_cache,
        subdir=subdir,
        n_fold_valid=n_fold_valid,
        mode_train=mode_train,
        dict_cache_only=fit_mode in ['dict_cache_only'],
        **kw_optim0
    )  # type: (vd2d.FitVD2D, ...)
    if fit_mode == 'dict_cache_only':
        dict_cache = out0
        return dict_cache, subdir
    else:
        best_loss, dict_cache, inp, out, targ, d, best_state = out0

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, data, dict_cache, d, dict_subdir


def get_subdir(fix_strs, n_fold_valid=1,
               **kwargs
               ) -> (str, dict):
    """
    for backward compatibility
    :return: subdir, dict_subdir
    """
    dict_subdir = {'fix': fix_strs}
    if n_fold_valid > 1:
        dict_subdir.update({'nfv': n_fold_valid})
    dict_subdir_str = {
        **dict_subdir,
        'fix': ','.join([''] + fix_strs)}
    subdir = argsutil.dict2fname(dict_subdir_str)
    return subdir, dict_subdir


def specify_fixs(
        fix: FixType = (
            'tdfix0',
            'ssq0',
            ('buffix', 0.12),
            'basym0_fix',
            'bhalf042_lb01',
            'diffix',
            'lps0',
            'noixn',
            'y0'
        )
) -> (dict, List[str], List[str]):
    """

    :param fix:
    :return: state_dict, fix_params, fix_strs
    """
    state_dict = {}
    fix_params = []
    fix_strs = []
    for fix1 in fix:
        if type(fix1) is str:
            fix_str1 = fix1
        else:
            fix_str1 = fix1[0]

        if fix1 == 'None':
            pass
        elif fix1 == 'tdfix0':
            state_dict.update({
                'dtb.td_offset._data': [0., 0.],
            })
            fix_params += [
                'dtb.td_offset._param'
            ]
        elif fix1 == 'ssq0':
            state_dict.update({
                'dtb.dtb.dtb1ds.0.ssq0._data': [0.0001],
                'dtb.dtb.dtb1ds.1.ssq0._data': [0.0001],
            })
            fix_params += [
                'dtb.dtb.dtb1ds.0.ssq0._param',
                'dtb.dtb.dtb1ds.1.ssq0._param'
            ]
        elif fix1[0] == 'buffix':
            fix_str1 = '(%s,%1.2f)' % (fix1[0], fix1[1])
            state_dict.update({
                'dtb.dur_buffer._data': [fix1[1]],
                'dtb.dur_buffer._ub': 2.,
            })
            fix_params += [
                'dtb.dur_buffer._param'
            ]
        elif fix1 == 'basym0_fix':
            state_dict.update({
                'dtb.dtb.dtb1ds.0.bound.b_asymptote._data': [0.],
                'dtb.dtb.dtb1ds.1.bound.b_asymptote._data': [0.],
                'dtb.dtb.dtb1ds.1.bound.b_asymptote._lb': 0.,
            })
            fix_params += [
                'dtb.dtb.dtb1ds.0.bound.b_asymptote._param',
                'dtb.dtb.dtb1ds.1.bound.b_asymptote._param',
            ]
        elif fix1 == 'bhalf042_lb01':
            state_dict.update({
                'dtb.dtb.dtb1ds.0.bound.b_t_half._data': [0.4],
                'dtb.dtb.dtb1ds.1.bound.b_t_half._data': [0.4],
                'dtb.dtb.dtb1ds.0.bound.b_t_half._ub': 2.,
                'dtb.dtb.dtb1ds.1.bound.b_t_half._ub': 2.,
                'dtb.dtb.dtb1ds.0.bound.b_t_half._lb': .1,
                'dtb.dtb.dtb1ds.1.bound.b_t_half._lb': .1,
                'dtb.dtb.dtb1ds.0.bound.b_t_st._lb': .1,
                'dtb.dtb.dtb1ds.1.bound.b_t_st._lb': .1,
            })
        elif fix1 == 'diffix':
            state_dict.update({
                'dtb.dtb.dtb1ds.0.diffusion._data': [1.],
                'dtb.dtb.dtb1ds.1.diffusion._data': [1.],
            })
            fix_params += [
                'dtb.dtb.dtb1ds.0.diffusion._param',
                'dtb.dtb.dtb1ds.1.diffusion._param',
            ]
        elif fix1 == 'lps0':
            state_dict.update({
                'lapse.p_lapse._data': [1e-4],
            })
            fix_params += [
                'lapse.p_lapse._param'
            ]
        elif fix1 == 'noixn':
            state_dict.update({
                'dtb.dtb.kappa_rel_odim._data': [0., 0.],
                'dtb.dtb.kappa_rel_abs_odim._data': [0., 0.],
            })
            fix_params += [
                'dtb.dtb.kappa_rel_odim._param',
                'dtb.dtb.kappa_rel_abs_odim._param'
            ]
        elif fix1 == 'y0':
            state_dict.update({
                'dtb.dtb.dtb1ds.0.bias_t0._data': [0.],
                'dtb.dtb.dtb1ds.1.bias_t0._data': [0.],
            })
            fix_params += [
                'dtb.dtb.dtb1ds.0.bias_t0._param',
                'dtb.dtb.dtb1ds.1.bias_t0._param',
            ]
        else:
            raise ValueError('Unknown fix=%s' % fix1)

        fix_strs.append(fix_str1)

    return state_dict, fix_params, fix_strs


def main_plot_across_models(
        i_subjs=i_subjs,
        axs=None,
        models_incl=('buffer+serial',),
        base=10,
        **kwargs,
):
    n_subj = len(i_subjs)

    # ---- Load goodness-of-fit
    def load_gof(i_subj, buffix_dur):
        fix = fix0_pre + [('buffix', buffix_dur)] + fix0_post
        dict_cache, subdir = main_fit(
            i_subj, fit_mode='dict_cache_only', fix=fix, **kwargs)

        file = locfile.get_file('tab', 'best_loss', dict_cache, ext='.csv',
                                subdir=subdir)
        import csv, os
        rows = None
        if not os.path.exists(file):
            print('File absent - returning NaN: %s' % file)
            gof = np.nan
            return gof
        with open(file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if rows is None:
                    rows = {k: [row[k]] for k in row.keys()}
                else:
                    for k in row.keys():
                        rows[k].append(row[k])
        gof_kind = 'loss_NLL_test'
        igof = rows['name'].index(gof_kind)
        gof = float(rows[' value'][igof][3:])
        return gof

    nbufdurs = len(bufdurs0)
    gofs = np.zeros([nbufdurs, n_subj])

    for i_subj in range(n_subj):
        for idur, bufdur in enumerate(bufdurs0):
            gofs[idur, i_subj] = load_gof(i_subj, bufdur)

    gofs = gofs - np.nanmin(gofs, 0, keepdims=True)
    gofs = gofs / np.log(base)

    # ---- Load slopes
    fixs = [
        (
        'buffer+serial', [
            ('buffix', bufdur_best),
        ]),
        # (
        # 'parallel', [
        #     ('buffix', 1.2),
        # ]),
        # (
        # 'serial', [
        #     ('buffix', 0.),
        # ]),
    ]
    model_names = [f[0] for f in fixs]
    n_models = len(model_names)

    models = np.empty([n_models, n_subj], dtype=np.object)
    dict_caches = np.empty([n_models, n_subj], dtype=np.object)
    ds = np.empty([n_models, n_subj], dtype=np.object)
    datas = np.empty([n_subj], dtype=np.object)

    kw_plots = {
        'serial': {'linestyle': '--'},
        'buffer+serial': {'linestyle': '-'},
        'parallel': {'linestyle': ':'},
    }

    subdir = ','.join(fix0) + '+buffix=' + ','.join([
        ('%1.2f' % v) for v in [0., bufdur_best, 1.2]
    ])

    for i_subj in range(n_subj):
        for i_model, (name, fix1) in enumerate(fixs):
            fix = fix0_pre + fix1 + fix0_post
            # try:
            model, data, dict_cache, d, _ = main_fit(
                i_subj, fit_mode='load', fix=fix, **kwargs
            )
            # except RuntimeError:
            #     model = None
            #     data = None
            #     dict_cache = None
            #     d = None

            models[i_model, i_subj] = model
            datas[i_subj] = data
            dict_caches[i_model, i_subj] = dict_cache
            ds[i_model, i_subj] = d

    # ---- Plot goodness-of-fit
    if axs is None:
        n_row = 2 + len(models_incl)
        n_row_gs = n_row + 2
        n_col = n_subj
        axs = plt2.GridAxes(
            3, 2,
            top=.3,
            left=1.15,
            right=0.1,
            bottom=.5,
            widths=[2],
            wspace=0.35,
            heights=[1.5, 1.5, 1.5],
            hspace=[0.2, 0.9]
        )

    for i_subj in range(n_subj):
        ax = axs[-1, i_subj]
        plt.sca(ax)
        plt2.box_off('all')

        gs1 = axs.gs[-2, i_subj * 2 + 1]
        bax = breakaxis(gs1)
        ax0 = bax.axs[0]  # type: plt.Axes
        ax1 = bax.axs[1]  # type: plt.Axes

        ax0.plot(bufdurs0[:3], gofs[:3, i_subj], 'k.-',
                 linewidth=0.75, markersize=4.5)
        ax1.plot(bufdurs0, gofs[:, i_subj], 'k.-',
                 linewidth=0.75, markersize=4.5)

        plt.sca(ax1)
        patch_chance_level(level=np.log(100.) / np.log(base), signs=[-1, 1])
        plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
        beautify(ax1)
        beautify_ticks(ax1, add_ticklabel=True) # i_subj == 0)

        plt2.sameaxes([ax0, ax1], xy='x')

        ax1.set_yticks([0, 20])
        ax0.set_yticks([40, 200])

        if i_subj == 0:
            plt.sca(ax1)
            plt.xlabel('buffer capacity (s)')

            plt.sca(ax)
            plt.ylabel(r'$-\mathrm{log}_{10}\mathrm{BF}$', labelpad=27)
        else:
            ax0.set_yticklabels([])
            ax1.set_yticklabels([])

        plt.sca(axs[0, i_subj])
        beautify_ticks(axs[0, i_subj], add_ticklabel=False)
        beautify_ticks(axs[1, i_subj], add_ticklabel=True)

    plt.sca(axs[-2, 0])
    plt.xlabel('stimulus duration (s)')

    # ---- Plot slopes
    for i_subj in range(n_subj):
        hss = []
        for model_name in models_incl:
            i_model = model_names.index(model_name)
            fix1 = fixs[i_model]
            model = models[i_model, i_subj]
            data = datas[i_subj]
            name = model_names[i_model]

            if model is not None:
                _, hs = plot_coefs_dur_odif_pred_data(
                    data, model,
                    axs=axs[:2, [i_subj]],
                    # to_plot_data=True,
                    to_plot_data=True,
                    kw_plot_model=kw_plots[model_name],
                    coefs_to_plot=[1],
                    add_rowtitle=False
                )
                hss.append(hs)

        if i_subj == 0:
            # hss[0] = hs['pred'|'data'][coef, dim, dif]
            hs1 = hss[0]['data']  # type: np.ndarray
            for dim in range(hs1.shape[1] - 1, -1, -1):
                hs = []
                for dif in range(hs1.shape[2]):
                    hs.append(hs1[0, dim, dif])
                odim = 1 - dim
                odim_name = consts.DIM_NAMES_LONG[odim]
                plt.sca(axs[dim, 0])

                plt.legend(hs, ['weak ' + odim_name.lower(),
                                'strong ' + odim_name.lower()],
                           bbox_to_anchor=[0., 0., 1., 1.],
                           handletextpad=0.35,
                           handlelength=0.5,
                           labelspacing=0.3,
                           borderpad=0.,
                           frameon=False,
                           loc='upper left')

        for row in range(axs.shape[0]):
            beautify(axs[row, i_subj])
            if i_subj > 0:
                plt.ylabel('')

    plt2.sameaxes(axs[-1, :])
    ax = axs[-1, -1]
    ax.set_yticklabels([])

    for i_subj in range(n_subj):
        plt.sca(axs[0, i_subj])
        plt.title(consts.SUBJS['VD'][i_subj])

        for row in range(1, n_row):
            plt.sca(axs[row, i_subj])
            plt.title('')

    for row, title in enumerate([
        r'Motion sensitivity ($\beta$)',
        r'Color sensitivity ($\beta$)',
    ]):
        plt.sca(axs[row, 0])
        plt.ylabel(title)

    dict_cache = dict_caches[0, 0]
    for k in ['fix', 'sbj']:
        if k in dict_cache:
            dict_cache.pop(k)

    for ext in ['.pdf', '.png']:
        file = locfile.get_file_fig('coefs_dur_odif_sbjs', dict_cache,
                                    subdir=subdir, ext=ext)
        from lib.pylabyk.cacheutil import mkdir4file
        mkdir4file(file)
        plt.savefig(file, dpi=300)
        print('Saved to %s' % file)

    print('--')

    return gofs, models, data


def breakaxis(gs1):
    return brokenaxes(
        subplot_spec=gs1,
        ylims=((-15/5*2, 90/5*2), (90/5*2, 750/5*2)),
        height_ratios=(50 / (90 + 15), 400 / (750 - 90) * 3),
        hspace=0.15,
        wspace=0.075,
        d=0.005,
    )


def beautify_ticks(ax, add_ticklabel=True):
    plt.sca(ax)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    if add_ticklabel:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    else:
        ax.set_xticklabels([])

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    ax.tick_params(which='minor', length=1.5)


def beautify(ax):
    xticks = np.arange(0, 1.2 + 0.1, 0.1)

    plt.sca(ax)
    plt.xlim(xmax=1.2 * 1.05)
    plt2.detach_axis('x', 0, 1.2)
    plt2.box_off()


def patch_chance_level(level=3., signs=(-1, 1,), ax=None,
                       style='line'):
    """

    :param level: 3 for NLL, 6 for BIC
    :param signs:
    :param ax:
    :return:
    """
    if ax is None:
        ax = plt.gca()

    hs = []
    x_lim = ax.get_xlim()

    for sign in signs:
        if style == 'patch':
            rect = mpl.patches.Rectangle(
                [x_lim[0], 0.], x_lim[1] - x_lim[0], level * sign,
                linewidth=0,
                fc=np.zeros(3) + 0.7,
                zorder=-1
            )
            ax.add_patch(rect)
            hs.append(rect)
        elif style == 'line':
            plt.axhline(level * sign, color=np.zeros(3) + 0.7, linestyle='--',
                        linewidth=0.5)
        else:
            raise ValueError()

    return hs
    

def main_plot_models_on_columns(
        i_subjs=(0, 1), axs=None,
        coef='slope'
):
    if coef not in ['slope', 'bias']:
        raise ValueError()

    n_subj = len(i_subjs)

    # ---- Load goodness-of-fit
    def load_gof(i_subj, buffix_dur):
        fix = fix0_pre + [('buffix', buffix_dur)] + fix0_post
        dict_cache, subdir = main_fit(
            i_subj, fit_mode='dict_cache_only', fix=fix)

        file = locfile.get_file('tab', 'best_loss', dict_cache, ext='.csv',
                                subdir=subdir)
        import csv, os
        rows = None
        if not os.path.exists(file):
            print('File absent - returning NaN: %s' % file)
            gof = np.nan
            return gof
        with open(file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if rows is None:
                    rows = {k: [row[k]] for k in row.keys()}
                else:
                    for k in row.keys():
                        rows[k].append(row[k])
        gof_kind = 'loss_NLL_test'
        igof = rows['name'].index(gof_kind)
        gof = float(rows[' value'][igof][3:])
        return gof

    bufdurs = [0., bufdur_best, 1.2]
    nbufdurs = len(bufdurs)
    gofs = np.zeros([nbufdurs, n_subj])

    for i_subj in range(n_subj):
        for idur, bufdur in enumerate(bufdurs):
            gofs[idur, i_subj] = load_gof(i_subj, bufdur)

    gofs = gofs - np.nanmin(gofs, 0, keepdims=True)

    # ---- Load coefs
    fixs = [
        (
        'buffer+serial', [
            ('buffix', bufdur_best),
        ]),
        (
        'parallel', [
            ('buffix', 1.2),
        ]),
        (
        'serial', [
            ('buffix', 0.),
        ]),
    ]
    model_names = [f[0] for f in fixs]
    n_models = len(model_names)

    models = np.empty([n_models, n_subj], dtype=np.object)
    dict_caches = np.empty([n_models, n_subj], dtype=np.object)
    ds = np.empty([n_models, n_subj], dtype=np.object)
    datas = np.empty([n_subj], dtype=np.object)

    subdir = ','.join(fix0) + '+buffix=' + ','.join([
        ('%1.2f' % v) for v in [0., bufdur_best, 1.2]
    ])

    for i_subj in range(n_subj):
        for i_model, (name, fix1) in enumerate(fixs):
            fix = fix0_pre + fix1 + fix0_post
            model, data, dict_cache, d, _ = main_fit(
                i_subj, fit_mode='load', fix=fix)

            models[i_model, i_subj] = model
            datas[i_subj] = data
            dict_caches[i_model, i_subj] = dict_cache
            ds[i_model, i_subj] = d

    # ---- Plot coefs
    kw_plots = {
        'serial': {'linestyle': '-'},
        'buffer+serial': {'linestyle': '-'},
        'parallel': {'linestyle': '-'},
    }

    def beautify(row, col, axs):
        plt.sca(axs[row, col])
        plt.xlim(xmax=1.2 * 1.05)
        plt2.detach_axis('x', 0, 1.2)
        plt2.box_off()

        beautify_ticks(
            axs[row, col],
            add_ticklabel=(row == n_row - 1)
                          and (col == 0))

        if col > 0:
            plt.ylabel('')

    import matplotlib as mpl

    for i_subj in range(n_subj):
        fig = plt.figure(figsize=[2.25 * n_models, 5.25])
        n_row = 3
        n_col = n_models

        n_col = len(fixs)
        n_row = 2
        axs = plt2.GridAxes(
            n_row, n_col,
            top=.7,
            left=1.15,
            right=0.1,
            bottom=.5,
            widths=[2],
            wspace=0.35,
            heights=[1.5],
            hspace=[0.2]
        )

        # ---- Plot slopes
        title_model = {
            'buffer+serial': 'parallel followed\nby serial',
            'parallel': 'strictly parallel',
            'serial': 'strictly serial'
        }
        for ii_model, model_name in enumerate(
                # ['buffer+serial', 'serial']):
                ['buffer+serial', 'parallel', 'serial']):
            i_model = model_names.index(model_name)
            fix1 = fixs[i_model]
            model = models[i_model, i_subj]
            data = datas[i_subj]
            name = model_names[i_model]

            plot_coefs_dur_odif_pred_data(
                data, model,
                axs=axs[:2, :][:, [ii_model]],
                to_plot_data=True,
                kw_plot_model=kw_plots[model_name],
                coefs_to_plot=[
                    1 if coef == 'slope' else 0
                ],
                add_rowtitle=False
            )

            for row in range(2):
                plt.sca(axs[row, ii_model])
                plt2.sameaxes(axs[row, :], xy='y')
                beautify(row, ii_model, axs)
                if ii_model > 0:
                    plt.gca().set_yticklabels([])

                if row == 0:
                    plt.title(title_model[model_name])
                    plt.xlabel('')
                else:
                    plt.title('')
                    if ii_model == 0:
                        plt.xlabel('stimulus duration (s)')
                    else:
                        plt.xlabel('')


        bnd = axs[0,1].get_position().bounds
        subj = consts.SUBJS['VD'][i_subj]
        plt.suptitle(
            subj,
            x=bnd[0] + bnd[2] / 2
        )

        rowtitles = [
            r'Motion sensitivity ($\beta$)',
            r'Color sensitivity ($\beta$)',
        ] if coef == 'slope' else [
            r'Motion bias ($\beta$)',
            r'Color bias ($\beta$)',
        ]
        for row, title in enumerate(rowtitles):
            plt.sca(axs[row, 0])
            plt.ylabel(title)

        dict_cache = dict_caches[0, 0]
        if 'fix' in dict_cache:
            dict_cache.pop('fix')

        dict_cache['sbj'] = subj
        dict_cache['coef'] = coef

        for ext in ['.png', '.pdf']:
            file = locfile.get_file_fig(
                'coefs_dur_odif', dict_cache, subdir=subdir,
                ext=ext
            )
            plt.savefig(file, dpi=300)
        print('Saved to %s' % file)
    print('--')

    return models, data


def get_data_2D(i_subj=0, parad='VD',
                trial_st=0,
                max_t=1.5,
                **kwargs
                ) -> (vd2d.Data2DVD, str, dict):
    """
    :param i_subj:
    :param parad:
    :return: dat, ch_by_dim[tr, dim], rt[tr], cond_by_dim[tr, dim], subj
    """

    nt0 = int(max_t * consts.REFRESH_RATE)

    # Choose by dim_rel and parad
    dat0 = load_data.load_data_combined()
    dat = np2.filt_dict(dat0, (
        np.all(dat0['dim_rel'], 1)
        & (dat0['id_parad'] == dat0['parads'].index(parad))
    ))

    # Choose subject
    id_subjs = np.unique(dat['id_subj'])
    id_subj = id_subjs[i_subj]
    subj = dat['subjs'][id_subj]
    dat = np2.filt_dict(dat, dat['id_subj'] == id_subj)

    print('Subject: %s (%d/%d) out of: ' %
          (subj, i_subj, len(id_subjs)), end='')
    print(np.array(dat['subjs'])[id_subjs]) # to check

    # ev and ch
    incl = np.all(~np.isnan(dat['ch']), 1) \
           & ((dat['id_parad'] == dat['parads'].index('VD'))
              | (~np.isnan(dat['RT'])))

    # don't subtract 1 from ch here - already done
    ch_tr_dim = (dat['ch'][incl, :]).astype(np.long)  # [tr, dim]

    ev_tr_dim = dat['cond'][incl, :]  # [tr, dim]
    dur_tr = dat['t_RDK_dur'][incl]  # type: np.ndarray

    ch_tr_dim = ch_tr_dim[trial_st:]
    ev_tr_dim = ev_tr_dim[trial_st:]
    dur_tr = dur_tr[trial_st:]

    data = vd2d.Data2DVD(
        ev_tr_dim, ch_tr_dim, dur_tr, nt0=nt0,
        subj=subj,
        parad=parad,
        ndim=2,
        dat=dat,
        **kwargs
    )
    return data


def get_dict_cache(
        subj: str,
        model=None,
        parad='VD', ndim=2,
        n_fold_valid=1,
        mode_train='all',
        dict_subdir=(),
) -> dict:
    """

    :param data:
    :param model:
    :param mode_train:
    :param dict_subdir:
    :return: dict_cache, dict_subdir
    """
    dict_subdir = dict(dict_subdir)

    dict_cache = argsutil.kwdef(
        dict_subdir, {
            'sbj': subj,
            'prd': parad,
            'ndim': ndim,
            'bnd': model.dtb.dtb1ds[0].bound.kind,
            'td': model.dtb.kind,
            'lps': model.lapse.kind,
            'mdtrn': mode_train,
        }
    )
    if n_fold_valid > 1:
        dict_cache.update({'nfv': n_fold_valid})

    return dict_cache


def fit_model_internal(
        model: vd2d.FitVD2D,
        data: vd2d.Data2DVD,
        dict_cache: dict,
        # --- Training
        mode_train='all',
        n_fold_valid=1,
        # --- Misc options
        ignore_cache=False,
        to_debug=False,
        continue_fit=False,
        save_results=None,
        subdir='',
        dict_cache_only=False,
        to_compute_grad=True,
        skip_fit_if_absent=False,
        **kwargs
) -> (vd2d.FitVD2D, vd2d.Data2DVD, dict, float, dict, tuple, torch.Tensor,
      torch.Tensor, dict):
    """

    :param model:
    :param data:
    :param i_subj:
    :param parad:
    :param n_fold_valid:
    :param kw_name:
    :param ignore_cache:
    :param subsample_factor:
    :param mode_train:
    :param to_debug:
    :param continue_fit:
    :param subdir: if not None or empty, copy files into the subdir
    :param kwargs:fed to initializing model and to sim2d.fit_dtb
    :return: best_loss, dict_cache, inp, out, targ, d, best_state
    """
    # == Get cache
    if dict_cache_only:
        return dict_cache

    cache = locfile.get_cache('fit', dict_cache, subdir=subdir)
    cache_exists = cache.exists()
    if cache_exists and not ignore_cache:
        best_loss, best_state = cache.getdict(['best_loss', 'best_state'])
        loaded_cache = True

        # Always load best_state
        model.load_state_dict(best_state, strict=False)
    else:
        loaded_cache = False
        best_loss = None

    best_state = model.state_dict()  # missing params filled in by model

    def save_cache(cache, d, best_loss, best_state, subdir):
        d_cache = {k: v for k, v in d.items() if k.startswith('loss_')}
        d_cache.update({
            'best_loss': best_loss,
            'best_state': best_state
        })
        cache.set(d_cache)
        cache.save()
        return cache.fullpath

    if ((best_loss is None) and (not skip_fit_if_absent)) or continue_fit:
        # == Fit and get best_loss and best_state
        if cache_exists:
            if continue_fit:
                reason = 'User chose to continue from'
            elif ignore_cache:
                reason = 'User chose to ignore'
            else:
                reason = 'For an unspecified reason, ignoring'
            print('%s cache at %s\n= %s'
                  % (reason, cache.fullpath, cache.fullpath_orig))
        print('Fitting model..')

        best_loss, best_state, d, plotfuns = vd2d.fit_dtb(
            model, data,
            n_fold_valid=n_fold_valid,
            mode_train=mode_train,
            to_debug=to_debug,
            comment='+' + argsutil.dict2fname(dict_cache),
            **kwargs
        )
        # if save_results:
        save_cache(cache, d, best_loss, best_state, subdir)

    elif skip_fit_if_absent:
        return None, dict_cache, None, None, None, None, None

    else:
        # Otherwise, just get plotfuns
        model.load_state_dict(best_state, strict=False)
        best_state = model.state_dict()
        kw = {
            **kwargs,
            'n_fold_valid': n_fold_valid,
            'mode_train': mode_train,
            'max_epoch': 0,
            'learning_rate': 0,
            'to_debug': to_debug,
            'comment': argsutil.dict2fname(dict_cache),
        }
        _, _, d, plotfuns = vd2d.fit_dtb(
            model, data,
            **kw
        )

    print('model (fit):')
    print(model.__str__())

    # == Compute gradient for plotting
    model.load_state_dict(best_state)
    ev_cond_fr_dim_meanvar, targ, durs = data.get_data_by_cond(
        'all')[:3]
    inp = (ev_cond_fr_dim_meanvar.clone(), durs.clone())

    if to_compute_grad:
        out = model(ev_cond_fr_dim_meanvar, durs)
        cost = vd2d.fun_loss_VD(out, targ)
        cost.backward()
    else:
        out = None

    if save_results is None:
        save_results = not (loaded_cache or skip_fit_if_absent)
    if save_results:
        save_fit_results(model, best_state, d, plotfuns,
                         locfile, dict_cache, subdir)

    pprint([(v[0], v[1].data, v[1].grad) for v in
            model.named_parameters()])

    if skip_fit_if_absent and not loaded_cache:
        best_loss = None
        d = None
        best_state = None

    return best_loss, dict_cache, inp, out, targ, d, best_state


def plot_coefs_dur_odif_pred_data(
        data, model, kw_plot_model=(),
        to_plot_data=True,
        axs=None,
        coefs_to_plot=(0, 1),
        dim_incl=(0, 1),
        **kwargs
):
    with torch.no_grad():
        ev_cond_fr_dim_meanvar, n_cond_dur_ch, durs0 = data.get_data_by_cond(
            'all')[:3]

        dt1 = model.dt
        durs = torch.arange(durs0[0] - dt1 * 0. - 1e-2 * 0, durs0[-1] + dt1,
                            dt1 * 3
                            ).clone()
        out0 = model(ev_cond_fr_dim_meanvar, durs)
        out1 = npy(out0)
        n_cond_dur_ch = npy(data.get_data_by_cond('all')[1])

        if axs is None:
            fig = plt.figure(figsize=[6, 4])

        kw_plot_model = {
            'zorder': -1,
            **kw_plot_model
        }

        axs, hs_pred = vd2d.plot_coefs_dur_odif(
            out1, data=data, durs=npy(durs), kw_plot=kw_plot_model,
            style='pred', axs=axs,  # fig=fig,
            coefs_to_plot=coefs_to_plot,
            dim_incl=dim_incl,
            jitter0=0.,
            **kwargs
        )[2:4]

        if to_plot_data:
            axs, hs_data = vd2d.plot_coefs_dur_odif(
                npy(n_cond_dur_ch), data=data,
                style='data', axs=axs,  # fig=fig,
                jitter0=0.,
                coefs_to_plot=coefs_to_plot,
                dim_incl=dim_incl,
                **kwargs
            )[2:4]
        else:
            hs_data = None
        hs = {
            'pred': hs_pred,
            'data': hs_data
        }
    return axs, hs


def plot_coefs_dur_irr_ixn_pred_data(
        data, model, kw_plot_model=(),
        to_plot_data=True,
        axs=None,
        coefs_to_plot=(2,),
):
    with torch.no_grad():
        ev_cond_fr_dim_meanvar, n_cond_dur_ch, durs0 = data.get_data_by_cond(
            'all')[:3]

        dt1 = model.dt
        durs = torch.arange(durs0[0] - dt1 * 0. - 1e-2, durs0[-1] + dt1,
                            dt1 * 1
                            ).clone()
        out0 = model(ev_cond_fr_dim_meanvar, durs)
        out1 = npy(out0)

        if axs is None:
            fig = plt.figure(figsize=[6, 4])

        axs = vd2d.plot_coefs_dur_irrixn(
            out1, data=data, durs=npy(durs), kw_plot=kw_plot_model,
            style='pred', axs=axs,  # fig=fig,
            jitter0=0.,
            coefs_to_plot=coefs_to_plot
        )[2]

        if to_plot_data:
            axs = vd2d.plot_coefs_dur_irrixn(
                npy(n_cond_dur_ch), data=data,
                style='data', axs=axs,  # fig=fig,
                jitter0=0.,
                coefs_to_plot=coefs_to_plot
            )[2]
    return axs  #, fig


def ____External_Interface____():
    pass


def load_fit(
        subj: str, bufdur: float,
        parad='VD',
        fix_post=None,
        skip_fit_if_absent=False
) -> (vd2d.FitVD2D, vd2d.Data2DVD, Dict[str, Any],
      Dict[str, torch.Tensor], str):
    """

    :param subj:
    :param bufdur:
    :return: model, data, dict_cache, d, subdir
    """
    if fix_post is None:
        fix_post = fix0_post
    i_subj = consts.SUBJS_VD.index(subj)
    fix = fix0_pre + [('buffix', bufdur)] + list(fix_post)

    if skip_fit_if_absent:
        fit_mode = 'load'
    else:
        fit_mode = 'auto'

    return main_fit(i_subj, parad=parad, fit_mode=fit_mode, fix=fix)


def init_model(
        subj: str, bufdur: float,
        parad='VD',
        fix_post=None
) -> (vd2d.FitVD2D, dict, dict):

    if fix_post is None:
        fix_post = fix0_post
    i_subj = consts.SUBJS_VD.index(subj)
    fix = fix0_pre + [('buffix', bufdur)] + list(fix_post)
    return main_fit(i_subj, fit_mode='init_model', fix=fix,
                    parad=parad)


def ____Main____():
    pass


if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.double)

    main()