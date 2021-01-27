#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

from typing import Union, Iterable
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
from copy import deepcopy

import torch

from lib.pylabyk.numpytorch import npy, npys
from lib.pylabyk import localfile, argsutil, np2, plt2

from data_2d import consts, load_data
from dtb import dtb_1D_sim as sim1d, dtb_2D_sim as sim2d
from dtb.dtb_1D_sim import save_fit_results

locfile = localfile.LocalFile(
    pth_root='../../data/Data_2D_Py/dtb/RT',
    cache_dir=''
)

# kw_plot_preds[(serial, parallel)]
kw_plot_preds = [
    {'linestyle': '-', 'linewidth': .5},
    {'linestyle': '--', 'linewidth': .5}
]
kw_plot_data = {
    'markersize': 4.5,
    'mew': 0.5
}

# cmaps[dim_rel]
cmaps = consts.CMAP_DIM


def get_title_str(parad):
    if parad == 'RT':
        title_str = 'Eye'
    elif parad == 'unimanual':
        title_str = 'Hand'
    elif parad == 'bimanual':
        title_str = 'Bimanual'
    else:
        raise ValueError()
    return title_str


def get_data_2D(i_subj: Union[str, int] = 0, parad='RT',
                bimanual: bool = None,
                trial_st=0,
                subsample_factor=1,
                ) -> (dict, np.ndarray, np.ndarray, np.ndarray, str):
    """
    :param i_subj:
    :param parad:
    :return: dat, ch_by_dim[tr, dim], rt[tr], cond_by_dim[tr, dim], subj
    """

    if parad == 'unimanual':
        assert (bimanual is None) or (not bimanual)
        bimanual = False
        parad = 'unibimanual'

    elif parad == 'bimanual':
        assert (bimanual is None) or bimanual
        bimanual = True
        parad = 'unibimanual'

    # Choose by dim_rel and parad
    dat0 = load_data.load_data_combined()
    dat = np2.filt_dict(dat0, (
        np.all(dat0['dim_rel'], 1)
        & (dat0['id_parad'] == dat0['parads'].index(parad))
    ))

    if parad == 'unibimanual':
        if bimanual is not None:
            dat = np2.filt_dict(dat, 
                                dat['bimanual'].astype(np.bool) == bimanual)



    # Choose subject
    id_subjs = np.unique(dat['id_subj'])
    if type(i_subj) is str:
        subj = i_subj
        id_subj = dat['subjs'].index(subj)
        i_subj = list(id_subjs).index(id_subj)
    else:
        id_subj = id_subjs[i_subj]
        subj = dat['subjs'][id_subj]
    dat = np2.filt_dict(dat, dat['id_subj'] == id_subj)

    print('Subject: %s (%d/%d) out of: ' %
          (subj, i_subj, len(id_subjs)), end='')
    print(np.array(dat['subjs'])[id_subjs]) # to check

    # ev and ch
    incl = np.all(~np.isnan(dat['ch']), 1) & ~np.isnan(dat['RT'])
    ch_tr_dim = (dat['ch'][incl, :] - 1).astype(np.long)  # [tr, dim]
    rt_tr = dat['RT'][incl]  # type: np.ndarray
    ev_tr_dim = dat['cond'][incl, :]  # [tr, dim]

    ch_tr_dim = ch_tr_dim[trial_st:]
    rt_tr = rt_tr[trial_st:]
    ev_tr_dim = ev_tr_dim[trial_st:]

    data = sim2d.Data2DRT(
        ev_tr_dim, ch_tr_dim, rt_tr,
        subsample_factor=subsample_factor
    )
    return data, subj, dat


def plot_fit_combined(
        data: Union[sim2d.Data2DRT, dict] = None,
        pModel_cond_rt_chFlat=None, model=None,
        pModel_dimRel_condDense_chFlat=None,
        # --- in place of data:
        pAll_cond_rt_chFlat=None,
        evAll_cond_dim=None,
        pTrain_cond_rt_chFlat=None,
        evTrain_cond_dim=None,
        pTest_cond_rt_chFlat=None,
        evTest_cond_dim=None,
        dt=None,
        # --- optional
        ev_dimRel_condDense_fr_dim_meanvar=None,
        dt_model=None,
        to_plot_internals=True,
        to_plot_params=True,
        to_plot_choice=True,
        # to_group_irr=False,
        group_dcond_irr=None,
        to_combine_ch_irr_cond=True,
        kw_plot_pred=(),
        kw_plot_pred_ch=(),
        kw_plot_data=(),
        axs=None,
):
    """

    :param data:
    :param pModel_cond_rt_chFlat:
    :param model:
    :param pModel_dimRel_condDense_chFlat:
    :param ev_dimRel_condDense_fr_dim_meanvar:
    :param to_plot_internals:
    :param to_plot_params:
    :param to_group_irr:
    :param to_combine_ch_irr_cond:
    :param kw_plot_pred:
    :param kw_plot_data:
    :param axs:
    :return:
    """
    if data is None:
        if pTrain_cond_rt_chFlat is None:
            pTrain_cond_rt_chFlat = pAll_cond_rt_chFlat
        if evTrain_cond_dim is None:
            evTrain_cond_dim = evAll_cond_dim
        if pTest_cond_rt_chFlat is None:
            pTest_cond_rt_chFlat = pAll_cond_rt_chFlat
        if evTest_cond_dim is None:
            evTest_cond_dim = evAll_cond_dim
    else:
        _, pAll_cond_rt_chFlat, _, _, evAll_cond_dim = \
            data.get_data_by_cond('all')
        _, pTrain_cond_rt_chFlat, _, _, evTrain_cond_dim = data.get_data_by_cond(
            'train_valid', mode_train='easiest')
        _, pTest_cond_rt_chFlat, _, _, evTest_cond_dim = data.get_data_by_cond(
            'test', mode_train='easiest')
        dt = data.dt
    hs = {}

    if model is None:
        assert not to_plot_internals
        assert not to_plot_params

    if dt_model is None:
        if model is None:
            dt_model = dt
        else:
            dt_model = model.dt

    if axs is None:
        if to_plot_params:
            axs = plt2.GridAxes(3, 3)
        else:
            if to_plot_internals:
                axs = plt2.GridAxes(3, 3)
            else:
                if to_plot_choice:
                    axs = plt2.GridAxes(2, 2)
                else:
                    axs = plt2.GridAxes(1, 2)  # TODO: beautify ratios

    rts = []
    hs['rt'] = []
    for dim_rel in range(consts.N_DIM):
        # --- data_pred may not have all conditions, so concatenate the rest
        #  of the conditions so that the color scale is correct. Then also
        #  concatenate p_rt_ch_data_pred1 with zeros so that nothing is
        #  plotted in the concatenated.
        evTest_cond_dim1 = np.concatenate([
            evTest_cond_dim, evAll_cond_dim
        ], axis=0)
        pTest_cond_rt_chFlat1 = np.concatenate([
            pTest_cond_rt_chFlat, np.zeros_like(pAll_cond_rt_chFlat)
        ], axis=0)

        if ev_dimRel_condDense_fr_dim_meanvar is None:
            evModel_cond_dim = evAll_cond_dim
        else:
            if ev_dimRel_condDense_fr_dim_meanvar.ndim == 5:
                evModel_cond_dim = npy(ev_dimRel_condDense_fr_dim_meanvar[
                                           dim_rel][:, 0, :, 0])
            else:
                assert ev_dimRel_condDense_fr_dim_meanvar.ndim == 4
                evModel_cond_dim = npy(ev_dimRel_condDense_fr_dim_meanvar[
                                           dim_rel][:, 0, :])
            pModel_cond_rt_chFlat = npy(pModel_dimRel_condDense_chFlat[dim_rel])

        if to_plot_choice:
            # --- Plot choice
            ax = axs[0, dim_rel]
            plt.sca(ax)

            if to_combine_ch_irr_cond:
                ev_cond_model1, p_rt_ch_model1 = combine_irr_cond(
                    dim_rel, evModel_cond_dim, pModel_cond_rt_chFlat
                )

                sim2d.plot_p_ch_vs_ev(ev_cond_model1, p_rt_ch_model1,
                                      dim_rel=dim_rel, style='pred',
                                      group_dcond_irr=None,
                                      kw_plot=kw_plot_pred_ch,
                                      cmap=lambda n: lambda v: [0., 0., 0.],
                                      )
            else:
                sim2d.plot_p_ch_vs_ev(evModel_cond_dim, pModel_cond_rt_chFlat,
                                      dim_rel=dim_rel, style='pred',
                                      group_dcond_irr=group_dcond_irr,
                                      kw_plot=kw_plot_pred,
                                      cmap=cmaps[dim_rel]
                                      )
            hs, conds_irr = sim2d.plot_p_ch_vs_ev(
                evTest_cond_dim1, pTest_cond_rt_chFlat1,
                dim_rel=dim_rel, style='data_pred',
                group_dcond_irr=group_dcond_irr,
                cmap=cmaps[dim_rel],
                kw_plot=kw_plot_data,
            )
            hs1 = [h[0] for h in hs]
            odim = 1 - dim_rel
            odim_name = consts.DIM_NAMES_LONG[odim]
            legend_odim(conds_irr, hs1, odim_name)
            sim2d.plot_p_ch_vs_ev(evTrain_cond_dim, pTrain_cond_rt_chFlat,
                                  dim_rel=dim_rel, style='data_fit',
                                  group_dcond_irr=group_dcond_irr,
                                  cmap=cmaps[dim_rel],
                                  kw_plot=kw_plot_data
                                  )
            plt2.detach_axis('x', np.amin(evTrain_cond_dim[:, dim_rel]),
                             np.amax(evTrain_cond_dim[:, dim_rel]))
            ax.set_xlabel('')
            ax.set_xticklabels([])
            if dim_rel != 0:
                plt2.box_off(['left'])
                plt.yticks([])

            ax.set_ylabel('P(%s choice)' % consts.CH_NAMES[dim_rel][1])

        # --- Plot RT
        ax = axs[int(to_plot_choice) + 0, dim_rel]
        plt.sca(ax)
        hs1, rts1 = sim2d.plot_rt_vs_ev(
            evModel_cond_dim,
            pModel_cond_rt_chFlat,
            dim_rel=dim_rel, style='pred',
            group_dcond_irr=group_dcond_irr,
            dt=dt_model,
            kw_plot=kw_plot_pred,
            cmap=cmaps[dim_rel]
        )
        hs['rt'].append(hs1)
        rts.append(rts1)

        sim2d.plot_rt_vs_ev(evTest_cond_dim1, pTest_cond_rt_chFlat1,
                            dim_rel=dim_rel, style='data_pred',
                            group_dcond_irr=group_dcond_irr,
                            dt=dt,
                            cmap=cmaps[dim_rel],
                            kw_plot=kw_plot_data
                            )
        sim2d.plot_rt_vs_ev(evTrain_cond_dim, pTrain_cond_rt_chFlat,
                            dim_rel=dim_rel, style='data_fit',
                            group_dcond_irr=group_dcond_irr,
                            dt=dt,
                            cmap=cmaps[dim_rel],
                            kw_plot=kw_plot_data
                            )
        plt2.detach_axis('x', np.amin(evTrain_cond_dim[:, dim_rel]),
                         np.amax(evTrain_cond_dim[:, dim_rel]))
        if dim_rel != 0:
            ax.set_ylabel('')
            plt2.box_off(['left'])
            plt.yticks([])

        ax.set_xlabel(consts.DIM_NAMES_LONG[dim_rel].lower() + ' strength')

        if dim_rel == 0:
            ax.set_ylabel('RT (s)')

        if to_plot_internals:
            for ch1 in range(consts.N_CH):
                ch0 = dim_rel
                ax = axs[3 + ch1, dim_rel]
                plt.sca(ax)

                ch_flat = consts.ch_by_dim2ch_flat(np.array([ch0, ch1]))
                model.tnds[ch_flat].plot_p_tnd()
                ax.set_xlabel('')
                ax.set_xticklabels([])
                ax.set_yticks([0, 1])
                if ch0 > 0:
                    ax.set_yticklabels([])

                ax.set_ylabel(r"$\mathrm{P}(T^\mathrm{n} \mid"
                              " \mathbf{z}=[%d,%d])$"
                              % (ch0, ch1))

            ax = axs[5, dim_rel]
            plt.sca(ax)
            if hasattr(model.dtb, 'dtb1ds'):
                model.dtb.dtb1ds[dim_rel].plot_bound(color='k')

    plt2.sameaxes(axs[-1, :consts.N_DIM], xy='y')

    if to_plot_params:
        ax = axs[0, -1]
        plt.sca(ax)
        model.plot_params()

    return axs, rts, hs


def legend_odim(conds_irr, hs1, odim_name, **kwargs):
    return plt.legend(
        hs1, ['%g' % v for v in conds_irr],
        title=r'$\left|\mathrm{%s~str}\right|$' % odim_name, **{
            'loc': 'lower right',
            'handlelength': 0.5,
            'handletextpad': 0.35,
            'labelspacing': 0.15,
            'borderpad': 0.,
            'bbox_to_anchor': [0., -0.02, 1., 1.],
            'frameon': False,
            **kwargs
        }
    )


def combine_irr_cond(dim_rel, evAll_cond_dim, pAll_cond_rt_chFlat):
    conds_rel = np.unique(evAll_cond_dim[:, dim_rel])
    n_conds_rel = len(conds_rel)
    p_rt_ch_dat1 = np.zeros((n_conds_rel,) + pAll_cond_rt_chFlat.shape[1:])
    ev_cond1 = np.zeros((n_conds_rel,) + evAll_cond_dim.shape[1:])
    for i_cond, cond in enumerate(conds_rel):
        incl = evAll_cond_dim[:, dim_rel] == cond
        p_rt_ch_dat1[i_cond] = pAll_cond_rt_chFlat[incl].sum(0)
        ev_cond1[i_cond, dim_rel] = cond
    return ev_cond1, p_rt_ch_dat1


def summarize_dict_fig(dict_fig):
    return {
        k: v
        for k, v in dict_fig.items()
        if k in ['lps', 'mdtrn', 'prd', 'sbj', 'td', 'tnd', 'grir']
    }


def get_data_combined(datas, normalize_ev=True):
    data0 = deepcopy(datas.flatten()[0])  # type: sim2d.Data2DRT
    evs = []
    chs = []
    rts = []

    def normalize(v):
        return v / np.amax(v, 0, keepdims=True)

    for data in datas.flatten():
        ev1 = deepcopy(data.ev_tr_dim)
        if normalize_ev:
            ev1 = normalize(ev1)
        evs.append(ev1)
        chs.append(data.ch_tr_dim)
        rts.append(data.rt_tr)
    data = sim2d.Data2DRT(
        np.concatenate(evs, axis=0),
        np.concatenate(chs, axis=0),
        np.concatenate(rts, axis=0),
        subsample_factor=data0.subsample_factor
    )
    return data


def get_suptitle_str(data, dict_cache, loss_CE=None,
                     to_show_n=True):
    """

    :param data:
    :param dict_cache: keys 'prd' (paradigm), 'sbj' (subj)
    :param loss_CE:
    :return:
    """
    parad = dict_cache['prd']
    # if parad == 'unibimanual':
    #     if dict_cache['bim']:
    #         parad_title = 'bimanual'
    #     else:
    #         parad_title = 'unimanual'
    if parad == 'RT':
        parad_title = 'RT eye'
    else:
        parad_title = parad
        # raise ValueError()
    _, p_rt_ch_dat, _, _, ev_cond = data.get_data_by_cond('all')
    n = p_rt_ch_dat.sum()

    subj = dict_cache['sbj']

    title_str = 'Subj %s - 2D, %s' % (
        subj, parad_title)
    if to_show_n or loss_CE is not None:
        title_str += '\n'
    if to_show_n:
        title_str += 'N=%d' % n
    if loss_CE is not None:
        if to_show_n:
            title_str += ', '
        title_str += 'CE=%1.3f, NLL=%1.1f' % npys(
            loss_CE,
            loss_CE * n * np.prod(np.array(p_rt_ch_dat.shape[1:])))
    return title_str


def plot_rt_distrib_pred_data(
        p_pred_cond_rt_ch,
        n_cond_rt_ch, ev_cond_dim, dt_model, dt_data=None,
        smooth_sigma_sec=0.1,
        to_plot_scale=False,
        to_cumsum=False,
        to_normalize_max=True,
        xlim=None,
        colors=('magenta', 'cyan'),
        kw_plot_pred=(),
        kw_plot_data=(),
        to_skip_zero_trials=False,
        labels=None,
        **kwargs
):
    """

    :param n_cond_rt_ch: [cond, rt, ch] = n_tr(cond, rt, ch)
    :param p_pred_cond_rt_ch: [model, cond, rt, ch] = P(rt, ch | cond, model)
    :param ev_cond_dim:
    :param dt_model:
    :param dt_data:
    :param smooth_sigma_sec:
    :param to_plot_scale:
    :param to_cumsum:
    :param xlim:
    :param kwargs:
    :return:
    """

    axs = None
    ps = []
    ps0 = []
    hss = []

    p_pred_cond_rt_ch = p_pred_cond_rt_ch / np.sum(
        p_pred_cond_rt_ch, (-1, -2), keepdims=True)
    n_preds1 = p_pred_cond_rt_ch * np.sum(
        n_cond_rt_ch, (-1, -2))[None, :, None, None]
    nt = p_pred_cond_rt_ch.shape[-2]
    if dt_data is None:
        dt_data = dt_model
    if labels is None:
        labels = [''] * (len(n_preds1) + 1)

    for i_pred, n_pred in enumerate(n_preds1):
        color = colors[i_pred]
        axs, p0, p1, hs = sim2d.plot_rt_distrib(
            n_pred, ev_cond_dim,
            dt=dt_model,
            axs=axs,
            alpha=1.,
            smooth_sigma_sec=smooth_sigma_sec,
            to_skip_zero_trials=to_skip_zero_trials,
            colors=color,
            alpha_face=0,
            to_normalize_max=to_normalize_max,
            to_cumsum=to_cumsum,
            to_use_sameaxes=False,
            kw_plot={
                'linewidth': 1.5,
                **dict(kw_plot_pred),
            },
            label=labels[i_pred],
            **kwargs,
        )[:4]
        ps.append(p1)
        ps0.append(p0)
        hss.append(hs)

    axs, p0, p1, hs = sim2d.plot_rt_distrib(
        n_cond_rt_ch, ev_cond_dim,
        dt=dt_data,
        axs=axs,
        smooth_sigma_sec=smooth_sigma_sec,
        colors='k',
        alpha_face=0.,
        to_normalize_max=to_normalize_max,  # normalize across preds and data instead
        to_cumsum=to_cumsum,
        to_skip_zero_trials=to_skip_zero_trials,
        kw_plot={
            'linewidth': 0.5,
            **dict(kw_plot_data),
        },
        label=labels[-1],
        **kwargs,
    )
    ps.append(p1)
    ps0.append(p0)
    hss.append(hs)

    ps = np.stack(ps)
    ps0 = np.stack(ps0)

    ps_flat = np.swapaxes(ps, 0, 2).reshape([ps.shape[1] * ps.shape[2], -1])

    for ax in axs.flatten():
        if xlim is None:
            if to_cumsum:
                xlim = [0.5, 4.5]
            else:
                xlim = [0.5, 4.5]

        plt2.detach_axis('x', *xlim, ax=ax)
        ax.set_xlim(xlim[0] - 0.1, xlim[1] + 0.1)

    axs[-1, 0].set_xticks(xlim)
    axs[-1, 0].set_xticklabels(['%g' % v for v in xlim])

    from lib.pylabyk import numpytorch as npt
    t = torch.arange(nt) * dt_model

    mean_rts = []
    for p1 in ps0:
        p11 = npt.sumto1(torch.tensor(p1).sum([-1, -2])[0, 0, :])
        mean_rts.append(npy((torch.tensor(t) * p11).sum()))
    print('mean_rts:')
    print(mean_rts)
    print(mean_rts[1] - mean_rts[0])

    conds = [np.unique(ev_cond_dim[:, i]) for i in [0, 1]]
    p_preds = torch.tensor(n_preds1).reshape([
        2, len(conds[0]), len(conds[1]), nt, 2, 2
    ]) + 1e-12

    if to_plot_scale:
        y = 0.8
        axs[-1, -1].plot(mean_rts[:2], y + np.zeros(2), 'k-', linewidth=0.5)
        x = np.mean(mean_rts[:2])
        plt.text(x, y + 0.1,
                 '%1.0f ms' % (np.abs(mean_rts[1] - mean_rts[0]) * 1e3),
                 ha='center', va='bottom')
    return axs, hss


def ____Main____():
    pass


if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.double)
