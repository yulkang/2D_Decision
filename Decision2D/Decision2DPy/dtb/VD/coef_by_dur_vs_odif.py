#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import matplotlib as mpl
import numpy as np
import numpy_groupies as npg
import statsmodels.api as sm
from matplotlib import pyplot as plt

from data_2d import consts, load_data
from lib.pylabyk import plt2, np2


def get_coefs(
        dim, dif_other,
        dur, ch, cond, t_RDK_dur,
        correct_only=True
):
    """

    :param dim:
    :param dif_other:
    :param dur: [tr]
    :param ch: [tr, dim]
    :param cond: [tr, dim]
    :param t_RDK_dur:
    :param correct_only:
    :return: glmres.params, glmres.bse, glmres, glmmodel
    """
    id_dif = np.empty_like(cond)
    for dim1 in range(consts.N_DIM):
        out = np.unique(np.abs(cond[:,dim1]),
                        return_inverse=True)
        _, id_dif[:, dim1] = out

    odim = consts.N_DIM - 1 - dim
    incl = (
            (t_RDK_dur == dur)
            & (np.isin(id_dif[:, odim], dif_other))
    )
    if correct_only:
        incl = (
                incl
                & (np.sign(ch[:, odim] - 0.5)
                   == np.sign(cond[:, odim]))
        )
    ch1 = ch[incl, dim]
    coh1 = cond[incl, dim]

    cohs, id_cohs = np.unique(coh1, return_inverse=True)
    if np.issubdtype(ch1.dtype, np.floating):
        # p_ch=1 is given
        ch11 = np.stack([
            npg.aggregate(id_cohs, ch1),
            npg.aggregate(id_cohs, 1 - ch1)
        ], -1)
    else:
        ch11 = npg.aggregate(np.vstack((id_cohs, 1 - ch1)), 1)

    glmmodel = sm.GLM(
        ch11, sm.add_constant(cohs), family=sm.families.Binomial())
    glmres = glmmodel.fit()
    return glmres.params, glmres.bse, glmres, glmmodel


def get_coefs_mesh(cond, ch, t_RDK_dur,
                   dif_irrs=(2, (0, 1)),
                   correct_only=False
                   ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """

    :param cond:
    :param ch:
    :param t_RDK_dur:
    :param dif_irrs:
    :param correct_only:
    :return: (coef, se_coef, glmres, glmmodel)
       coef[(bias, slope), dim, dif, dur]
    """
    dims = [0, 1]
    t_RDK_durs, id_durs = np.unique(t_RDK_dur, return_inverse=True)
    coef, se_coef, glmres, glmmodel = np2.meshfun(
        lambda *args: get_coefs(
            *args,
            ch=ch, cond=cond, t_RDK_dur=t_RDK_dur,
            correct_only=correct_only),
        [dims, dif_irrs, t_RDK_durs],
        n_out=4,
        outshape_first=True
    )
    return coef, se_coef, glmres, glmmodel


def get_coefs_from_histogram(cond, p_cond_ch):
    glmmodel = sm.GLM(p_cond_ch, sm.add_constant(cond),
                      family=sm.families.Binomial())
    glmres = glmmodel.fit()
    return glmres.params, glmres.bse, glmres, glmmodel


def get_coefs_mesh_from_histogram(
        p_cond_dur_ch: np.ndarray,
        ev_cond_dim: np.ndarray,
        dif_irrs=((2,), (0, 1))
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """

    :param p_cond_dur_ch:
    :param ev_cond_dim: [cond, dim]
    :param dif_irrs:
    return: (coef, se_coef, glmres, glmmodel)
      coef[(bias, slope), dim, dif, dur]
    """
    n_dim = ev_cond_dim.shape[1]
    n_dif = len(dif_irrs)
    n_dur = p_cond_dur_ch.shape[1]
    siz = [n_dim, n_dif, n_dur]

    n_coef = 4

    coef = np.zeros([n_coef] + siz) + np.nan
    se_coef = np.zeros([n_coef] + siz) + np.nan
    glmress = np.empty(siz, dtype=np.object)
    glmmodels = np.empty(siz, dtype=np.object)

    p_cond_dur_ch = p_cond_dur_ch.reshape([-1] + [n_dur]
                                          + [consts.N_CH] * 2)

    for dim_rel in range(n_dim):
        for idif, dif_irr in enumerate(dif_irrs):
            for idur in range(n_dur):
                dim_irr = consts.get_odim(dim_rel)

                cond_irr = ev_cond_dim[:, dim_irr]
                adcond_irr = np.unique(np.abs(cond_irr), return_inverse=True)[1]
                incl = np.isin(adcond_irr, dif_irr)

                ev_cond_dim1 = ev_cond_dim[incl]
                reg = [ev_cond_dim1[:, dim_rel]]

                reg += [
                    ev_cond_dim1[:, dim_irr],
                ]
                if len(dif_irr) > 1:
                    # otherwise np.abs(cond_irr) would be constant
                    reg.append(np.abs(ev_cond_dim1[:, dim_irr]))

                reg = np.stack(reg, -1)
                reg = sm.add_constant(reg)
                n_coef1 = reg.shape[1]

                if dim_rel == 0:
                    p_cond_ch = p_cond_dur_ch[incl, idur, :, :].sum(-1)
                else:
                    p_cond_ch = p_cond_dur_ch[incl, idur, :, :].sum(-2)
                glmmodel = sm.GLM(np.flip(p_cond_ch, -1),
                                  reg,
                                  family=sm.families.Binomial())
                glmres = glmmodel.fit()
                coef[:n_coef1, dim_rel, idif, idur] = glmres.params
                se_coef[:n_coef1, dim_rel, idif, idur] = glmres.bse
                glmress[dim_rel, idif, idur] = glmres
                glmmodels[dim_rel, idif, idur] = glmmodel
    return coef, se_coef, glmress, glmmodels


def get_coefs_irr_ixn_from_histogram(
        p_cond_dur_ch: np.ndarray,
        ev_cond_dim: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """

    :param p_cond_dur_ch:
    :param ev_cond_dim: [cond, dim]
    :param dif_irrs:
    return: (coef, se_coef, glmres, glmmodel)
      coef[(bias, slope), dim, dif, dur]
    """
    n_dim = ev_cond_dim.shape[1]
    n_dur = p_cond_dur_ch.shape[1]
    siz = [n_dim, n_dur]

    n_coef = 6  # constant, rel, rel x abs(irr), rel x irr, abs(irr), irr,

    coef = np.zeros([n_coef] + siz) + np.nan
    se_coef = np.zeros([n_coef] + siz) + np.nan
    glmress = np.empty(siz, dtype=np.object)
    glmmodels = np.empty(siz, dtype=np.object)

    p_cond_dur_ch = p_cond_dur_ch.reshape([-1] + [n_dur]
                                          + [consts.N_CH] * 2)

    for dim_rel in range(n_dim):
        for idur in range(n_dur):
            dim_irr = consts.get_odim(dim_rel)

            reg = [
                ev_cond_dim[:, dim_rel],
                ev_cond_dim[:, dim_rel] * np.abs(ev_cond_dim[:, dim_irr]),
                ev_cond_dim[:, dim_rel] * ev_cond_dim[:, dim_irr],
                np.abs(ev_cond_dim[:, dim_irr]),
                ev_cond_dim[:, dim_irr]
            ]

            reg = np.stack(reg, -1)
            reg = sm.add_constant(reg)
            n_coef1 = reg.shape[1]

            if dim_rel == 0:
                p_cond_ch = p_cond_dur_ch[:, idur, :, :].sum(-1)
            else:
                p_cond_ch = p_cond_dur_ch[:, idur, :, :].sum(-2)
            glmmodel = sm.GLM(np.flip(p_cond_ch, -1),
                              reg,
                              family=sm.families.Binomial())
            glmres = glmmodel.fit()
            coef[:n_coef1, dim_rel, idur] = glmres.params
            se_coef[:n_coef1, dim_rel, idur] = glmres.bse
            glmress[dim_rel, idur] = glmres
            glmmodels[dim_rel, idur] = glmmodel
    return coef, se_coef, glmress, glmmodels


def get_subj_id(dat0=None, i_subj=0, parad='VD'):
    if dat0 is None:
        dat0 = load_data.load_data_combined()
    dat = load_data.filt_dict(
        dat0,
        dat0['id_parad'] == dat0['parads'].index(parad)
    )
    return np.unique(dat['id_subj'])[i_subj]


def get_data_VD(dat0=None, i_subj=0, parad='VD', return_dict=True):
    if dat0 is None:
        dat0 = load_data.load_data_combined()
    dat = load_data.filt_dict(
        dat0,
        dat0['id_parad'] == dat0['parads'].index(parad)
    )

    #%%
    id_subj = np.unique(dat['id_subj'])[i_subj]
    subj = dat['subjs'][id_subj]
    dat1 = load_data.filt_dict(
        dat,
        (dat['n_dim_task'] == 2) &
        (dat['id_subj'] == id_subj)
    )

    #%%
    dat1['ch'] = dat1['ch'].astype(np.long)

    if return_dict:
        return dat1, subj, dat0
    else:
        return [
            dat1[k] for k in ['cond', 'ch', 't_RDK_dur', 'RT', 'en']
        ] + [subj, dat0]


def plot_coef_by_dur_vs_odif(
        coef, se_coef,
        normalize_bias=True,
        coef_name='slope',
        savefig=True,
        difs=[[2, 1], [0]],
        t_RDK_durs=np.arange(1, 11) * 0.12,
        fig=None,
        horizontal_panels=False
):
    """

    @param dat0:
    @param parad:
    @param use_data:
    @param correct_only:
    @param plot_slope: if False, plot bias
    @param i_subj:
    @param savefig:
    @param fig:
    @return:
    """

    if fig is None:
        if horizontal_panels:
            fig = plt.figure(figsize=(7, 2))
        else:
            fig = plt.figure(figsize=(4, 3))
    if horizontal_panels:
        gs = plt.GridSpec(
            figure=fig,
            nrows=1, ncols=2,
            hspace=0.3,
            left=0.11,
            right=0.98,
            top=0.9,
            bottom=0.15
        )
    else:
        gs = mpl.gridspec.GridSpec(
            figure=fig,
            nrows=2, ncols=1,
            left=0.22,
            right=0.98,
            top=0.9,
            bottom=0.15
        )

    coef_names = ['bias', 'slope']
    i_coef = coef_names.index(coef_name)

    units = ['(logit)', '(logit/coh)']
    if normalize_bias:
        coef[0] = -coef[0] / coef[1]
        se_coef[0] = -se_coef[0] / coef[1]
        units[0] = '(coh)'
    unit = units[i_coef]

    y = coef[i_coef]
    se = se_coef[i_coef]
    if len(difs) == 1:
        labels = ['all']
    elif len(difs) == 2:
        labels = ['easy', 'hard']
    elif len(difs) == 3:
        labels = ['easy', 'medium', 'hard']
    else:
        raise ValueError()

    for dim, (slope1, se_slope1) in enumerate(zip(y, se)):
        odim = consts.N_DIM - 1 - dim
        if horizontal_panels:
            plt.subplot(gs[0, dim])
        else:
            plt.subplot(gs[dim, 0])

        # plt.plot(t_RDK_durs, slope[dim])
        for odif, (slope2, se_slope2) in enumerate(zip(slope1, se_slope1)):
            label = labels[odif] + ' ' + consts.DIM_NAMES_SHORT[odim]
            plt.errorbar(
                t_RDK_durs, slope2, se_slope2, marker='o',
                label=label
            )
            plt.ylabel(consts.DIM_NAMES_LONG[dim].lower()
                       + ' ' + coef_name + '\n' + unit)
        plt.axis('auto')
        plt.xlim(xmin=-0.02)
        if coef_name == 'slope':
            plt.ylim(ymin=-0.05)
        plt2.box_off()
        plt2.detach_axis('x')
        if dim == 0 and not horizontal_panels:
            plt.gca().set_xticklabels([])
        plt.legend(
            handlelength=0.4,
            frameon=False, loc='upper left'
        )
    plt.xlabel('stimulus duration (s)')
    return gs


def plot_ch_vs_coh_by_dur(ch, coh, dur):
    cohs, i_coh = np.unique(coh, return_inverse=True)
    durs, i_dur = np.unique(dur, return_inverse=True)
    ch_by_coh_dur = npg.aggregate(
        np.stack([i_dur, i_coh]),
        ch.astype(np.double),
        'mean'
    )
    return plt2.plotmulti(
        cohs, ch_by_coh_dur, cmap='coolwarm'
    ), ch_by_coh_dur, cohs, durs

