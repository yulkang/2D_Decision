#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.


import numpy as np
from matplotlib import pyplot as plt
import numpy_groupies as npg
from collections import OrderedDict as odict
from typing import Union, Iterable, Sequence, Tuple, Any

import torch

from dtb.VD import coef_by_dur_vs_odif as coefdur
from lib.pylabyk import numpytorch as npt, yktorch as ykt
from lib.pylabyk.numpytorch import npy
from lib.pylabyk import argsutil, plt2

from dtb import dtb_1D_sim as sim1d, dtb_2D_sim as sim2d
from dtb.dtb_1D_sim \
    import fun_loss
from data_2d import consts

npt.device0 = torch.device('cpu')
ykt.default_device = torch.device('cpu')

def ____Utils____():
    pass


def fun_loss_VD(p_cond_dur__ch_pred: torch.Tensor,
                p_cond_dur__ch_data: torch.Tensor,
                to_average=True, base_n_bin=True) -> torch.Tensor:
    """
    Different from RT in that durations are treated as a
    condition, and doesn't count into n_bin.
    :param p_cond_dur__ch_pred: [condition, rt_frame, choice] = number of
    trials in that bin
    :param p_cond_dur__ch_data: [condition, rt_frame, choice] = P(rt,ch|cond)
    :return: cross entropy (if to_average=True) or NLL (otherwise)
    """
    n_bin = np.prod(p_cond_dur__ch_pred.shape[2:]).astype(np.float)
    loss = -torch.sum(
        p_cond_dur__ch_data
        * torch.log(p_cond_dur__ch_pred + 1e-12 / n_bin)  # noqa
    )
    if to_average:
        loss = loss / torch.sum(p_cond_dur__ch_data)
    if base_n_bin:
        loss = loss / np.log(n_bin)
    return loss


def get_n_conds_dur_chs(n_cond_dur_ch):
    """

    :param n_cond_dur_ch[cond_flat, dur, ch_flat]:
    :return: n_conds_dur_chs[cond_dim0, cond_dim1, dur, ch_dim0, ch_dim1]
    """
    n_dur = n_cond_dur_ch.shape[1]
    n_chs = [2, 2]
    n_conds = [6, 6]
    n_conds_dur_chs = n_cond_dur_ch.reshape(
        n_conds + [n_dur] + n_chs
    )
    return n_conds_dur_chs


def ____Model_Classes____():
    pass


class Data2DVD(sim2d.Data2D):
    def __init__(
            self,
            ev_tr_dim: np.ndarray,
            ch_tr_dim: np.ndarray,
            dur_tr: np.ndarray,
            subj='None',
            parad='VD',
            ndim=-1,
            dim_rel=-1,
            desc_dict: Union[dict, Iterable[Tuple[str, Any]]] = (),
            dat: dict = None,
            **kwargs
    ):
        """

        :param ev_tr_dim:
        :param ch_tr_dim:
        :param dur_tr: stimulus duration in seconds
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.ev_tr_dim = ev_tr_dim
        self.ch_tr_dim = ch_tr_dim
        self.dur_tr = dur_tr
        self.n_ch = consts.N_CH_FLAT

        self._desc_dict = odict(desc_dict)
        self.subj = subj
        self.parad = parad
        self.ndim = ndim
        self.dim_rel = dim_rel

        if dat is None:
            self.dat = dict()
        else:
            self.dat = dat

        # Placeholders
        self.n_cond_dur_ch = np.empty(())
        self.ev_cond_dim = np.empty(())
        self.dcond_tr = np.empty(())
        self.durs = np.empty(())
        self.ddur_tr = np.empty(())

        self.update_data(self.ch_tr_dim, self.dur_tr, self.ev_tr_dim)

    def update_data(
            self,
            ch_tr_dim: np.ndarray = None,
            dur_tr: np.ndarray = None,
            ev_tr_dim: np.ndarray = None
    ):
        if ch_tr_dim is None:
            ch_tr_dim = self.ch_tr_dim
        else:
            self.ch_tr_dim = ch_tr_dim

        if dur_tr is None:
            dur_tr = self.dur_tr
        else:
            self.dur_tr = dur_tr

        if ev_tr_dim is None:
            ev_tr_dim = self.ev_tr_dim
        else:
            self.ev_tr_dim = ev_tr_dim

        self.n_cond_dur_ch, _, self.ev_cond_dim, self.dcond_tr, self.durs, \
            self.ddur_tr = \
                self.dat2p_dat(ch_tr_dim, dur_tr, ev_tr_dim)[:6]

    def get_data_by_cond(
            self, mode='all', i_fold_valid=0, epoch=0,
            mode_train='all', n_fold_valid=5,
            to_debug=False
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray,
          np.ndarray, np.ndarray, np.ndarray):
        """

        :param mode: 'all'|'train'|'valid'|'train_valid'|'test'
        :param i_fold_valid:
        :param epoch:
        :param mode_train: 'all'|'easiest'
        :param n_fold_valid:
        :param to_debug:
        :return: ev_cond_fr_dim_meanvar, n_cond_dur_ch, durs, \
            in_tr, in_dcond, ev_cond_dim
        """
        in_dcond, in_tr, in_tr_train_valid = self.get_incl(
            mode, i_fold_valid, mode_train, n_fold_valid)

        n_cond_dur_ch, ev_cond_fr_dim_meanvar, ev_cond_dim, dcond_tr, durs, \
            ddur_tr = self.dat2p_dat(
                self.ch_tr_dim[in_tr],
                self.dur_tr[in_tr],
                self.ev_tr_dim[in_tr])

        ev_cond_dim = self.ev_cond_dim[in_dcond]
        return ev_cond_fr_dim_meanvar, n_cond_dur_ch, durs, \
            in_tr, in_dcond, ev_cond_dim

    def dat2p_dat(
            self,
            ch_tr_dim: np.ndarray,
            dur_tr: np.ndarray,
            ev_tr_dim: np.ndarray
    ) -> (torch.Tensor, torch.Tensor, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray):
        """
        :param ch_tr_dim: [tr, dim]
        :param dur_tr: [tr]
        :param ev_tr_dim: [tr, dim]
        :return: n_cond_dur_ch[cond, dur, ch],
        ev_cond_fr_dim_meanvar[dcond, fr, dim, (mean, var)],
        ev_cond_dim[dcond, dim], dcond_tr[tr],
        durs[dur], ddur_tr[tr]
        """
        nt0 = self.nt0
        dt0 = self.dt0
        n_ch_flat = self.n_ch
        subsample_factor = self.subsample_factor

        nt = int(nt0 // subsample_factor)

        durs, ddur_tr = np.unique(dur_tr, return_inverse=True)
        ddur_tr = ddur_tr.astype(np.int)
        n_dur = len(durs)
        durs = torch.tensor(durs)
        ddur_tr = torch.tensor(ddur_tr, dtype=torch.long)

        ch_tr_flat = consts.ch_by_dim2ch_flat(ch_tr_dim)

        ev_cond_dim, dcond_tr = np.unique(ev_tr_dim, return_inverse=True,
                                          axis=0)
        n_cond_flat = len(ev_cond_dim)
        ev_cond_fr_dim = torch.tensor(ev_cond_dim)[:, None, :].expand(
            [-1, nt, -1])

        ev_cond_fr_dim_meanvar = torch.stack([
            ev_cond_fr_dim, torch.zeros_like(ev_cond_fr_dim)
        ], -1)

        n_cond_dur_ch = npt.tensor(npg.aggregate(np.stack([
            dcond_tr, npy(ddur_tr), ch_tr_flat
        ]), 1., 'sum', [n_cond_flat, n_dur, n_ch_flat]))

        return n_cond_dur_ch, ev_cond_fr_dim_meanvar, ev_cond_dim, dcond_tr, \
            durs, ddur_tr

    def simulate_data(self, pPred_cond_dur_ch: torch.Tensor, seed=0):
        dcond_tr = self.dcond_tr
        ddur_tr = self.ddur_tr

        pPred_tr_ch = pPred_cond_dur_ch[dcond_tr, ddur_tr, :]

        torch.random.manual_seed(seed)
        chSim_tr_ch = npt.categrnd(probs=pPred_tr_ch)
        chs = np.array(consts.CHS)
        chSim_tr_dim = np.stack([
            chs[dim][npy(chSim_tr_ch)]
            for dim in range(consts.N_DIM)
        ], -1)
        self.update_data(ch_tr_dim=chSim_tr_dim)


class Dtb2DVD(sim1d.TimedModule):
    kind = 'None'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, ev: torch.Tensor, durs: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class Dtb2DVDParallel(Dtb2DVD):
    kind = 'parallel'

    def __init__(self,
                 dtb2d: sim2d.Dtb2DRTDimensionWise
                 = sim2d.Dtb2DRTDimensionWise,
                 **kwargs):
        super().__init__(**kwargs)

        if type(dtb2d) is type:
            kw = argsutil.kwdef(kwargs, {
                'timer': self.timer
            })
            dtb2d = dtb2d(**kw)
        self.dtb = dtb2d  # type: sim2d.Dtb2DRTDimensionWise
        self.expand_ev = dtb2d.expand_ev
        self.get_out_dtb1ds = dtb2d.get_out_dtb1ds

        self.td_offset = ykt.BoundedParameter([0.] * consts.N_DIM, -8/75, 0.3)

    @property
    def dtb1ds(self):
        """Delegate - for compatibility with plotting functions in sim2d (RT)"""
        return self.dtb.dtb1ds

    def forward(self, ev: torch.Tensor, durs: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param ev: [cond, fr, dim, (mean, var)]
        :param durs: [ddur] (in fr, dtype int for now)
        :return: p_cond_dur_ch [cond, dur, ch],
        unabs_dim_td_cond_ev[dim, td, cond, ev]
        """

        ev = self.expand_ev(ev)
        p_dim_cond_td_ch, unabs_dim_td_cond_ev = self.get_out_dtb1ds(
            ev, return_unabs=True)

        p_cond_dur_ch = self.get_parallel_choice(durs, p_dim_cond_td_ch,
                                                 unabs_dim_td_cond_ev)

        return p_cond_dur_ch, unabs_dim_td_cond_ev

    def get_parallel_choice(
            self, durs: torch.Tensor,
            p_dim_cond_td_ch: torch.Tensor,
            unabs_dim_td_cond_ev: torch.Tensor
    ) -> torch.Tensor:
        dim_ev = torch.stack([
            dtb.ev_bin for dtb in self.dtb.dtb1ds
        ])
        cum_p_dim_cond_td_ch = p_dim_cond_td_ch.cumsum(-2)
        chs = torch.tensor([0., 1.])
        # dim_ch_ev[dim, ch, ev]: weight given to each ch
        dim_ch_ev = (chs[None, :, None]
                     - (-dim_ev.unsqueeze(1).sign() / 2 + 0.5)).abs()
        n_dim = p_dim_cond_td_ch.shape[0]
        n_dur = len(durs)
        n_cond = p_dim_cond_td_ch.shape[1]
        p_dim_cond_dur_ch0 = torch.zeros(n_dim, n_cond, n_dur, consts.N_CH)
        unabs_dim_dur_cond_ch = torch.zeros(n_dim, n_dur, n_cond, consts.N_CH)
        for dim in range(consts.N_DIM):
            td_offset = self.td_offset[dim]

            # NOTE: when dur == 0, it must remain 0 regardless of td_offset
            #   but this is moot since no condition has dur == 0
            # itdurs[durs_pos] = (durs[durs_pos] + td_offset) / self.dt
            itdurs = (durs + td_offset) / self.dt
            durs_rem = (itdurs % self.dt) / self.dt
            itdurs_lb = itdurs.floor().long()
            itdurs_ub = itdurs_lb + 1
            # durs_rem = (durs - itdurs_lb * self.dt) / self.dt
            w_lbub_dur = torch.stack([1. - durs_rem, durs_rem])

            # p_dim_cond_dur_ch0[dim, cond, dur, ch]
            p_dim_cond_dur_ch0[dim] = (
                    torch.stack([
                        cum_p_dim_cond_td_ch[dim, :, itdurs_lb, :],
                        cum_p_dim_cond_td_ch[dim, :, itdurs_ub, :]
                    ]) * w_lbub_dur[:, None, :, None]
            ).sum(0)

            for i, durs1 in enumerate([itdurs_lb, itdurs_ub]):
                unabs_dur_cond_ev1 = unabs_dim_td_cond_ev[dim, durs1, :, :]
                ch1 = (
                        dim_ch_ev[dim, None, None, :, :]
                        @ unabs_dur_cond_ev1.unsqueeze(-1)
                ).squeeze(-1)
                unabs_dim_dur_cond_ch[dim] = (
                        unabs_dim_dur_cond_ch[dim]
                        + ch1 * w_lbub_dur[i, :, None, None]
                )
        # Sum unabs and abs to get ch
        p_cond_dur_ch = torch.empty(p_dim_cond_dur_ch0.shape[1:-1]
                                    + (consts.N_CH_FLAT,))
        for i, ch1 in enumerate(torch.tensor(consts.CHS, dtype=torch.long).T):
            # NOTE: sums each dim's choice (absorbed OR unabsorbed)
            #  and then multiply between dims.
            p_ch1_given_cond_dur = (
                    (p_dim_cond_dur_ch0[0, :, :, ch1[0]]
                     + unabs_dim_dur_cond_ch[0, :, :, ch1[0]].permute([1, 0]))
                    * (p_dim_cond_dur_ch0[1, :, :, ch1[1]]
                       + unabs_dim_dur_cond_ch[1, :, :, ch1[1]].permute([1, 0]))
            )
            p_cond_dur_ch[:, :, i] = p_ch1_given_cond_dur
        if torch.any(p_cond_dur_ch < 0):
            print('Negative p_cond_dur_ch!')
            print('--')
        p_cond_dur_ch = p_cond_dur_ch / p_cond_dur_ch.sum(-1, keepdim=True)
        return p_cond_dur_ch

    def plot_p_ch_by_dur(self, p_dim_cond_dur_ch0, cond_irr=2, ch=1):
        p_dim_conds_dur_ch = npy(p_dim_cond_dur_ch0.reshape(
            p_dim_cond_dur_ch0.shape[:1] + (6, 6,) + (10, 2)
        ))
        n_cond1 = p_dim_conds_dur_ch.shape[1]
        cmap = plt2.cool2_rev(n_cond1)
        for i in range(n_cond1):
            plt.plot(p_dim_conds_dur_ch[0, i, cond_irr, :, ch].T,
                     color=cmap(i))


def ev2ch(p_ev: torch.Tensor, dim=-1) -> torch.Tensor:
    if dim != 0:
        p_ev = p_ev.transpose(0, dim)

    n = p_ev.shape[0]
    p_ch = torch.zeros((2,) + p_ev.shape[1:])
    if n % 2 == 0:
        n2 = n // 2
        p_ch[0] = p_ev[:n2].sum(0)
        p_ch[1] = p_ev[n2:].sum(0)
    else:
        n2 = n // 2
        p_ch[0] = p_ev[:n2].sum(0) + p_ev[n2] / 2.
        p_ch[1] = p_ev[(n2 + 1):].sum(0) + p_ev[n2] / 2.

    if dim != 0:
        p_ch = p_ch.transpose(0, dim)
    return p_ch


class Dtb2DVDBufSerial(Dtb2DVDParallel):
    kind = 'buffered_serial'

    def __init__(
            self, *args,
            dur_buffer0=0.2,
            p1st_dim0_0=0.5,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        # dur_buffer: in seconds.
        self.dur_buffer = ykt.BoundedParameter([dur_buffer0], 0.001, .5)
        # self.dur_buffer = ykt.BoundedParameter([dur_buffer0], 0., 1e-4)
        self.p1st_dim0 = ykt.BoundedParameter([p1st_dim0_0], 1e-3, 1 - 1e-3)

    def forward(self, ev: torch.Tensor, durs: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param ev: [cond, fr, dim, (mean, var)]
        :param durs: [ddur] (in fr, dtype int for now)
        :return: p_cond_dur_chFlat [cond, dur, chFlat],
        unabs_dim_td_cond_ev[dim, td, cond, ev]
        """

        ev = self.expand_ev(ev)
        p_dim_cond_td_chDim0, unabs_dim_td_cond_ev = self.get_out_dtb1ds(
            ev, return_unabs=True)
        unabs_dim_td_cond_chDim0 = ev2ch(unabs_dim_td_cond_ev, -1)

        # ==== Adjust Td using td_offset
        p_dim_cond_td_chDim1 = torch.empty_like(p_dim_cond_td_chDim0)
        unabs_dim_td_cond_chDim1 = torch.empty_like(unabs_dim_td_cond_chDim0)

        for dim, (p_cond_td_chDim0, unabs_td_cond_chDim0) in enumerate(zip(
                p_dim_cond_td_chDim0, unabs_dim_td_cond_chDim0)):
            td_offset_fr = -self.td_offset[dim] / self.dt
            p_dim_cond_td_chDim1[dim] = npt.shiftdim(
                p_cond_td_chDim0, td_offset_fr, -2, pad=0
            )
            unabs_dim_td_cond_chDim1[dim] = npt.shiftdim(
                unabs_td_cond_chDim0, td_offset_fr, -3, pad='repeat'
            )

        # Parallel up to buffer duration
        p_cond_dur_chFlat = self.get_p_cond_dur_chFlat(
            p_dim_cond_td_chDim1,
            unabs_dim_td_cond_chDim1,
            dur_buffer_fr=self.dur_buffer[0] / self.dt,
            dur_stim_frs=(durs / self.dt).long(),
            p1st_dim0=self.p1st_dim0[0],
        )
        return p_cond_dur_chFlat, unabs_dim_td_cond_ev

    @staticmethod
    def get_p_cond_dur_chFlat(
            p_dim_cond_td_chDim: torch.Tensor,
            unabs_dim_td_cond_chDim: torch.Tensor,
            dur_buffer_fr: torch.Tensor, dur_stim_frs: torch.Tensor,
            p1st_dim0: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param p_dim_cond_td_chDim: [dim, cond, td, chDim]
        :param unabs_dim_td_cond_chDim: [dim, td, cond, chDim]
        :param dur_buffer_fr: scalar
        :param dur_stim_frs: [idur]
        :return: p_cond_dur_chFlat[cond, dur, chFlat]
        """
        if torch.is_floating_point(dur_buffer_fr):
            bufs = torch.cat([
                dur_buffer_fr.floor().long().reshape([1]),
                dur_buffer_fr.floor().long().reshape([1]) + 1
            ], 0)
            prop_buf = torch.tensor(1.) - torch.abs(dur_buffer_fr - bufs)
            ps = []
            for buf in bufs:
                ps.append(Dtb2DVDBufSerial.get_p_cond_dur_chFlat(
                    p_dim_cond_td_chDim, unabs_dim_td_cond_chDim,
                    buf.long(), dur_stim_frs=dur_stim_frs,
                    p1st_dim0=p1st_dim0
                ))
            ps = torch.stack(ps)
            p_cond_dur_chFlat = (ps * prop_buf[:, None, None, None]).sum(0)
            return p_cond_dur_chFlat

        p1st_dim = [p1st_dim0, torch.tensor(1.) - p1st_dim0]

        n_cond = p_dim_cond_td_chDim.shape[1]
        ndur = len(dur_stim_frs)
        p_cond_dur_chFlat = torch.zeros([n_cond, ndur, consts.N_CH_FLAT])

        cumP_dim_cond_td_chDim = p_dim_cond_td_chDim.cumsum(-2)

        for dim1st in range(consts.N_DIM):
            dim2nd = consts.get_odim(dim1st)
            for idur, dur_stim in enumerate(dur_stim_frs):
                p0 = torch.zeros([n_cond, consts.N_CH_FLAT])
                for ich, chs in enumerate(consts.CHS_TENSOR.T):
                    ch1st = chs[dim1st]
                    ch2nd = chs[dim2nd]

                    for td1st in torch.arange(dur_stim):
                        max_td2nd = dur_stim - max([td1st - dur_buffer_fr, 0])
                        # ==== When both dims are absorbed
                        p0[:, ich] = p0[:, ich] + (
                            p_dim_cond_td_chDim[dim1st, :, td1st, ch1st]
                            * cumP_dim_cond_td_chDim[
                                dim2nd, :, max_td2nd, ch2nd]
                        )

                        # ==== When only dim1st is absorbed
                        p0[:, ich] = p0[:, ich] + (
                            p_dim_cond_td_chDim[dim1st, :, td1st, ch1st]
                            * unabs_dim_td_cond_chDim[
                                dim2nd, max_td2nd, :, ch2nd]
                        )
                    # ==== When dim1st is not absorbed
                    t1st = dur_stim
                    t2nd = dur_stim - max([t1st - dur_buffer_fr, 0])

                    # ==== When only dim2nd is absorbed: this can happen when
                    #   dim2nd is absorbed within the buffer duration
                    p0[:, ich] = p0[:, ich] + (
                        unabs_dim_td_cond_chDim[
                            dim1st, t1st, :, ch1st]
                        * cumP_dim_cond_td_chDim[
                            dim2nd,:, t2nd, ch2nd]
                    )

                    # ==== When neither dim is absorbed
                    p0[:, ich] = p0[:, ich] + (
                        unabs_dim_td_cond_chDim[
                            dim1st, t1st, :, ch1st]
                        * unabs_dim_td_cond_chDim[
                            dim2nd, t2nd, :, ch2nd]
                    )

                p0 = p0 / p0.sum(1, keepdim=True)
                p_cond_dur_chFlat[:, idur, :] = (
                    p_cond_dur_chFlat[:, idur, :]
                    + p1st_dim[dim1st] * p0
                )
        return p_cond_dur_chFlat

    @staticmethod
    def get_p_cond_dur_chFlat_vectorized(
                p_dim_cond_td_chDim: torch.Tensor,
                unabs_dim_td_cond_chDim: torch.Tensor,
                dur_buffer_fr: torch.Tensor, dur_stim_frs: torch.Tensor,
                p1st_dim0: torch.Tensor,
        ) -> torch.Tensor:

        if torch.is_floating_point(dur_buffer_fr):
            bufs = torch.cat([
                dur_buffer_fr.floor().long().reshape([1]),
                dur_buffer_fr.floor().long().reshape([1]) + 1
            ], 0)
            prop_buf = torch.tensor(1.) - torch.abs(dur_buffer_fr - bufs)
            ps = []
            for buf in bufs:
                ps.append(Dtb2DVDBufSerial.get_p_cond_dur_chFlat(
                    p_dim_cond_td_chDim, unabs_dim_td_cond_chDim,
                    buf.long(), dur_stim_frs=dur_stim_frs,
                    p1st_dim0=p1st_dim0
                ))
            ps = torch.stack(ps)
            p_cond_dur_chFlat = (ps * prop_buf[:, None, None, None]).sum(0)
            return p_cond_dur_chFlat

        # vectorized version
        p1st_dim = torch.stack([p1st_dim0, torch.tensor(1.) - p1st_dim0])

        n_cond = p_dim_cond_td_chDim.shape[1]
        ndur = len(dur_stim_frs)
        p_cond_dur_chFlat = torch.zeros([n_cond, ndur, consts.N_CH_FLAT])

        p_cond_dim_td_chDim = p_dim_cond_td_chDim.transpose(0, 1)
        unabs_dim_cond_td_chDim = unabs_dim_td_cond_chDim.transpose(1, 2)
        unabs_cond_dim_td_chDim = unabs_dim_cond_td_chDim.transpose(0, 1)
        ichs = torch.arange(consts.N_CH_FLAT)
        dim1sts = torch.arange(consts.N_DIM)

        for idur, dur_stim in enumerate(dur_stim_frs):
            p0 = torch.zeros([n_cond, consts.N_CH_FLAT])

            for td1st in torch.arange(dur_stim):
                max_td2nd = dur_stim - max([td1st - dur_buffer_fr, 0])
                td2nds = torch.arange(max_td2nd)

                dim1st, td2nd, ich = torch.meshgrid([dim1sts, td2nds, ichs])
                dim2nd = consts.get_odim(dim1st)
                ch1st = consts.CHS_TENSOR[dim1st, ich]
                ch2nd = consts.CHS_TENSOR[dim2nd, ich]

                # When both dim1st and dim2nd are absorbed
                p0 = p0 + (
                    (
                        p_cond_dim_td_chDim[
                            :, dim1st, td1st.expand_as(td2nd), ch1st]
                        * p_cond_dim_td_chDim[:, dim2nd, td2nd, ch2nd]
                    ).sum(-2)  # sum across td2nd
                    * p1st_dim[None, :, None]
                ).sum(1)  # sum across p1st

                # When only dim1st is absorbed,
                t2nd = max_td2nd

                dim1st, ich = torch.meshgrid([dim1sts, ichs])
                dim2nd = consts.get_odim(dim1st)
                ch1st = consts.CHS_TENSOR[dim1st, ich]
                ch2nd = consts.CHS_TENSOR[dim2nd, ich]

                p0 = p0 + (
                    (
                        p_cond_dim_td_chDim[:, dim1st, td1st, ch1st]
                        * unabs_cond_dim_td_chDim[:, dim2nd, t2nd, ch2nd]
                    ) * p1st_dim[None, :, None]
                ).sum(1)  # sum across dim1st

            dim1st, ich = torch.meshgrid([dim1sts, ichs])
            dim2nd = consts.get_odim(dim1st)

            ch1st = consts.CHS_TENSOR[dim1st, ich]
            ch2nd = consts.CHS_TENSOR[dim2nd, ich]

            # When neither dim is absorbed,
            # then dim2nd is certainly not absorbed,
            # and stays at the state at t = min([dur_stim, dur_buffer_fr])
            t2nd = min([dur_stim, dur_buffer_fr])
            p0 = p0 + (
                    unabs_cond_dim_td_chDim[:, dim1st, dur_stim, ch1st]
                    * unabs_cond_dim_td_chDim[:, dim2nd, t2nd, ch2nd]
                    * p1st_dim[None, :, None]
            ).sum(1)

            p_cond_dur_chFlat[:, idur, :] = (
                p_cond_dur_chFlat[:, idur, :]
                + p0
            )
        return p_cond_dur_chFlat


class LapseVD(sim1d.TimedModule):
    """
    Different from Lapse for RT in that duration is another form of
    condition, rather than response (unlike RT). Hence, each cond x dur has
    only # choice bins, rather than # rt x # choice as in the RT paradigm.
    """
    kind = 'None'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, p_cond_dur_ch: torch.Tensor) -> torch.Tensor:
        """
        :param p_cond_dur_ch: [cond, dur, ch]
        :return: p_cond_dur_ch[cond, dur, ch]
        """
        raise NotImplementedError()


class LapseVDUniform(LapseVD):
    kind = 'uniform'

    def __init__(self, lapse_max=0.1, **kwargs):
        super().__init__(**kwargs)
        self.p_lapse = ykt.BoundedParameter([lapse_max / 2], 1e-6, lapse_max)

    def forward(self, p_cond_dur_ch: torch.Tensor) -> torch.Tensor:
        """
        Each cond x dur has only # choice bins, rather than # rt x # choice
        as in the RT paradigm.
        :param p_cond_dur_ch:
        :return: p_cond_dur_ch[cond, dur, ch]
        """
        p_cond_dur_ch = (
                p_cond_dur_ch * (1. - self.p_lapse[:])
                + self.p_lapse[:] / np.prod(p_cond_dur_ch.shape[2:])
        )
        return p_cond_dur_ch


class FitVD2D(sim1d.TimedModule):
    """
    Put DTB and Lapse together to fit the data.
    """
    def __init__(
            self,
            dtb2d: Dtb2DVDParallel = Dtb2DVDParallel,
            lapse: LapseVD = LapseVDUniform,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.chs = torch.tensor(consts.CHS)  # [dim, ch_flat] = ch_dim

        if type(dtb2d) is type:
            dtb2d = dtb2d(timer=self.timer, chs=self.chs, **kwargs)
        self.dtb = dtb2d  # type: Dtb2DVDParallel

        if type(lapse) is type:
            lapse = lapse(timer=self.timer, **kwargs)
        self.lapse = lapse  # type: LapseVD

    def forward(
            self, ev: torch.Tensor, durs: torch.Tensor = None,
            return_unabs=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param ev: [cond, fr, dim, (mean, var)] or (ev, durs) fed from
        ykt.optimize()
        :param durs: [ddur] (in fr, dtype int for now)
        :param return_unabs: defaults to False so that optimize() can feed the
        single output to fun_loss()
        :return: p_cond_dur_ch [cond, dur, ch],
        unabs_dim_td_cond_ev[dim, td, cond, ev]
        """
        if durs is None:  # given as a tuple
            ev, durs = ev  # type: (torch.Tensor, torch.Tensor)
        p_cond_dur_ch, unabs_dim_td_cond_ev = self.dtb(ev=ev, durs=durs)[:2]
        p_cond_dur_ch = self.lapse(p_cond_dur_ch)

        if return_unabs:
            return p_cond_dur_ch, unabs_dim_td_cond_ev
        else:
            return p_cond_dur_ch

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict = sim1d.Dtb1D.update_state_dict(state_dict)
        super().load_state_dict(state_dict, *args, **kwargs)


def ____Plot____():
    pass


def plot_coefs_dur_odif(
        n_cond_dur_ch: np.ndarray = None,
        data: Data2DVD = None,
        dif_irrs: Sequence[Union[Iterable[int], int]] = ((0, 1), (2,)),
        ev_cond_dim: np.ndarray = None,
        durs: np.ndarray = None, style='data', kw_plot=(),
        jitter0=0.,
        coefs_to_plot=(0, 1),
        add_rowtitle=True,
        dim_incl=(0, 1),
        axs=None, fig=None,
):
    """

    :param n_cond_dur_ch:
    :param data:
    :param dif_irrs:
    :param ev_cond_dim:
    :param durs:
    :param style:
    :param kw_plot:
    :param jitter0:
    :param coefs_to_plot:
    :param add_rowtitle:
    :param dim_incl:
    :param axs:
    :param fig:
    :return: coef, se_coef, axs, hs[coef, dim, dif]
    """

    if ev_cond_dim is None:
        ev_cond_dim = data.ev_cond_dim
    if durs is None:
        durs = npy(data.durs)
    if n_cond_dur_ch is None:
        n_cond_dur_ch = npy(data.get_data_by_cond('all')[1])

    coef, se_coef = coefdur.get_coefs_mesh_from_histogram(
        n_cond_dur_ch, ev_cond_dim=ev_cond_dim,
        dif_irrs=dif_irrs
    )[:2]  # type: (np.ndarray, np.ndarray)

    coef_names = ['bias', 'slope']
    n_coef = len(coefs_to_plot)
    n_dim = len(dim_incl)

    if axs is None:
        if fig is None:
            fig = plt.figure(figsize=[6, 1.5 * n_coef])

        n_row = n_dim
        n_col = n_coef
        gs = plt.GridSpec(
            nrows=n_row, ncols=n_col,
            figure=fig,
            left=0.25, right=0.95,
            bottom=0.15, top=0.9
        )
        axs = np.empty([n_row, n_col], dtype=np.object)
        for row in range(n_row):
            for col in range(n_col):
                axs[row, col] = plt.subplot(gs[row, col])

    n_dif = len(dif_irrs)
    hs = np.empty([len(coefs_to_plot),
                   len(dim_incl),
                   len(dif_irrs)],
                  dtype=np.object)

    for ii_coef, i_coef in enumerate(coefs_to_plot):
        coef_name = coef_names[i_coef]
        for i_dim, dim_rel in enumerate(dim_incl):
            ax = axs[i_dim, ii_coef]  # type: plt.Axes
            plt.sca(ax)

            cmap = consts.CMAP_DIM[dim_rel](n_dif)

            for idif, dif_irr in enumerate(dif_irrs):
                y = coef[i_coef, dim_rel, idif, :]
                e = se_coef[i_coef, dim_rel, idif, :]
                ddur = durs[1] - durs[0]
                jitter = ddur * jitter0 * (idif - (n_dif - 1) / 2)
                kw = consts.get_kw_plot(style, color=cmap(idif),
                                        for_err=True,
                                        **dict(kw_plot))
                if style.startswith('data'):
                    h = plt.errorbar(durs + jitter, y, yerr=e, **kw)[0]
                else:
                    h = plt.plot(durs + jitter, y, **kw)
                hs[ii_coef, i_dim, idif] = h
            plt2.box_off()

            max_dur = np.amax(npy(durs))
            ax.set_xlim(-0.05 * max_dur, 1.05 * max_dur)
            plt2.detach_axis('x', 0, max_dur)

            if dim_rel == 0:
                ax.set_xticklabels([])
                ax.set_title(coef_name.lower())
            elif i_coef == 0:
                ax.set_xlabel('duration (s)')

    if add_rowtitle:
        plt2.rowtitle(consts.DIM_NAMES_LONG, axes=axs)

    return coef, se_coef, axs, hs


def plot_coefs_dur_irrixn(
        n_cond_dur_ch: np.ndarray = None,
        data: Data2DVD = None,
        ev_cond_dim: np.ndarray = None,
        durs: np.ndarray = None, style='data', kw_plot=(),
        jitter0=0.,
        coefs_to_plot=(2,),
        axs=None, fig=None,
):
    if ev_cond_dim is None:
        ev_cond_dim = data.ev_cond_dim
    if durs is None:
        durs = npy(data.durs)
    if n_cond_dur_ch is None:
        n_cond_dur_ch = npy(data.get_data_by_cond('all')[1])

    coef, se_coef = coefdur.get_coefs_irr_ixn_from_histogram(
        n_cond_dur_ch, ev_cond_dim=ev_cond_dim
    )[:2]  # type: (np.ndarray, np.ndarray)

    coef_names = ['bias', 'slope', 'rel x abs(irr)',
                  'rel x irr', 'abs(irr)', 'irr']
    n_coef = len(coefs_to_plot)

    if axs is None:
        if fig is None:
            fig = plt.figure(figsize=[6, 1.5 * n_coef])

        n_row = consts.N_DIM
        n_col = n_coef
        gs = plt.GridSpec(
            nrows=n_row, ncols=n_col,
            figure=fig,
            left=0.25, right=0.95,
            bottom=0.15, top=0.9
        )
        axs = np.empty([n_row, n_col], dtype=np.object)
        for row in range(n_row):
            for col in range(n_col):
                axs[row, col] = plt.subplot(gs[row, col])

    for ii_coef, i_coef in enumerate(coefs_to_plot):
        coef_name = coef_names[i_coef]
        for dim_rel in range(consts.N_DIM):
            ax = axs[dim_rel, ii_coef]  # type: plt.Axes
            plt.sca(ax)

            y = coef[i_coef, dim_rel, :]
            e = se_coef[i_coef, dim_rel, :]
            jitter = 0.
            kw_plot = {
                'color': 'k',
                'for_err': True,
                **dict(kw_plot)
            }
            kw = consts.get_kw_plot(style, **kw_plot)
            if style.startswith('data'):
                plt.errorbar(durs + jitter, y, yerr=e, **kw)
            else:
                plt.plot(durs + jitter, y, **kw)
            plt2.box_off()

            max_dur = np.amax(npy(durs))
            ax.set_xlim(-0.05 * max_dur, 1.05 * max_dur)
            plt2.detach_axis('x', 0, max_dur) # , ax=ax)

            if dim_rel == 0:
                ax.set_xticklabels([])
                ax.set_title(coef_name.lower())
            elif i_coef == 0:
                ax.set_xlabel('duration (s)')

    plt2.rowtitle(consts.DIM_NAMES_LONG, axes=axs)

    return coef, se_coef, axs


def plot_ch_ev_by_dur(
        n_cond_dur_ch: np.ndarray = None,
        data: Data2DVD = None,
        dif_irrs: Sequence[Union[Iterable[int], int]] = ((0, 1), (2,)),
        ev_cond_dim: np.ndarray = None,
        durs: np.ndarray = None,
        dur_prct_groups=((0, 33), (33, 67), (67, 100)),
        style='data', kw_plot=(),
        jitter=0.,
        axs=None,
        fig=None
):
    """
    Panels[dim, irr_dif_group], curves by dur_group
    :param n_cond_dur_ch:
    :param data:
    :param dif_irrs:
    :param ev_cond_dim:
    :param durs:
    :param dur_prct_groups:
    :param style:
    :param kw_plot:
    :param axs: [row, col]
    :return:
    """
    if ev_cond_dim is None:
        ev_cond_dim = data.ev_cond_dim
    if durs is None:
        durs = data.durs
    if n_cond_dur_ch is None:
        n_cond_dur_ch = npy(data.get_data_by_cond('all')[1])

    n_conds_dur_chs = get_n_conds_dur_chs(n_cond_dur_ch)
    conds_dim = [np.unique(cond1) for cond1 in ev_cond_dim.T]
    n_dur = len(dur_prct_groups)
    n_dif = len(dif_irrs)

    if axs is None:
        if fig is None:
            fig = plt.figure(figsize=[6, 4])

        n_row = consts.N_DIM
        n_col = n_dur
        gs = plt.GridSpec(
            nrows=n_row, ncols=n_col,
            figure=fig
        )
        axs = np.empty([n_row, n_col], dtype=np.object)
        for row in range(n_row):
            for col in range(n_col):
                axs[row, col] = plt.subplot(gs[row, col])

    for dim_rel in range(consts.N_DIM):
        for idif, dif_irr in enumerate(dif_irrs):
            for i_dur, dur_prct_group in enumerate(dur_prct_groups):

                ax = axs[dim_rel, i_dur]
                plt.sca(ax)

                conds_rel = conds_dim[dim_rel]
                dim_irr = consts.get_odim(dim_rel)
                _, cond_irr = np.unique(np.abs(conds_dim[dim_irr]),
                                        return_inverse=True)
                incl_irr = np.isin(cond_irr, dif_irr)

                cmap = consts.CMAP_DIM[dim_rel](n_dif)

                ix_dur = np.arange(len(durs))
                dur_incl = (
                    (ix_dur >= np.percentile(ix_dur, dur_prct_group[0]))
                    & (ix_dur <= np.percentile(ix_dur, dur_prct_group[1]))
                )
                if dim_rel == 0:
                    n_cond_dur_ch1 = n_conds_dur_chs[:, incl_irr, :, :, :].sum(
                        (1, -1))
                else:
                    n_cond_dur_ch1 = n_conds_dur_chs[incl_irr, :, :, :, :].sum(
                        (0, -2))
                n_cond_ch1 = n_cond_dur_ch1[:, dur_incl, :].sum(1)
                p_cond__ch1 = n_cond_ch1[:, 1] / n_cond_ch1.sum(1)

                x = conds_rel
                y = p_cond__ch1

                dx = conds_rel[1] - conds_rel[0]
                jitter1 = 0
                kw = consts.get_kw_plot(style, color=cmap(idif),
                                        **dict(kw_plot))
                if style.startswith('data'):
                    plt.plot(x + jitter1, y, **kw)
                else:
                    plt.plot(x + jitter1, y, **kw)

                plt2.box_off(ax=ax)
                plt2.detach_yaxis(0, 1, ax=ax)

                ax.set_yticks([0, 0.5, 1.])
                if i_dur == 0:
                    ax.set_yticklabels(['0', '', '1'])
                else:
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])

    return axs


def plot_unabs(model, inp):
    ev_cond_fr_dim_meanvar, durs = inp
    with torch.no_grad():
        # # unabs[dim, td, cond, ev]
        # _, unabs = model(*inp, return_unabs=True)

        p_tds = []
        unabss = []
        for dim_rel in range(consts.N_DIM):
            ev1 = ev_cond_fr_dim_meanvar[:, :, dim_rel, :]
            dtb1d = model.dtb.dtb.dtb1ds[dim_rel]  # type: sim1d.Dtb1D
            p_td, unabs = dtb1d.forward(ev1, return_unabs=True)
            unabs = unabs.permute([1, 0, 2])

            nt = p_td.shape[1]
            p_td = npy(p_td).reshape([6, 6, nt, 2])
            unabs = npy(unabs).reshape([6, 6, nt, -1])

            if dim_rel == 0:
                p_td = p_td.mean(1)
                unabs = unabs.mean(1)
            else:
                p_td = p_td.mean(0)
                unabs = unabs.mean(0)

            p_tds.append(p_td)
            unabss.append(unabs)

    dtb1ds = model.dtb.dtb.dtb1ds

    # unabss[dim, cond, td, ev]
    unabss = np.array(unabss)

    # p_tds[dim, cond, td, ch]
    p_tds = np.array(p_tds)

    n_conds = 6
    gs = plt.GridSpec(nrows=n_conds, ncols=consts.N_DIM)
    for dim_rel, (unabs, p_td, dtb1d) in enumerate(zip(unabss, p_tds, dtb1ds)):
        for cond, (unabs1, p_td1) in enumerate(zip(unabs, p_td)):
            ax = plt.subplot(gs[cond, dim_rel])  # type: plt.Axes  # noqa
            dtb1d.plot_unabs(unabs1, p_td1)

            if dim_rel > 0 or cond < len(unabs) - 1:
                ax.set_xticklabels([])
                ax.set_yticklabels([])


def ____Fit____():
    pass


def fun_data(
        data, mode='all', fold_valid=0, epoch=0, n_fold_valid=1,
        mode_train='all', to_debug=False
) -> (Tuple[torch.Tensor], Sequence[float], np.ndarray):
    """

    :param data:
    :param mode:
    :param fold_valid:
    :param epoch:
    :param mode_train:
    :param to_debug:
    :return:
    """
    ev_cond_fr_dim_meanvar, n_cond_dur_ch, durs = data.get_data_by_cond(
        mode=mode, i_fold_valid=fold_valid, epoch=epoch,
        mode_train=mode_train,
        n_fold_valid=n_fold_valid,
        to_debug=to_debug
    )[:3]
    return (ev_cond_fr_dim_meanvar, durs), n_cond_dur_ch


def fit_dtb(model: FitVD2D,
            data: Data2DVD,
            n_fold_valid=1,
            mode_train='all',
            to_debug=False,
            max_epoch=500,
            **kwargs
            ) -> (float, dict):
    """
    Provide functions fun_data() and plot_*() to ykt.optimize().
    See ykt.optimize() for details about fun_data and plot_*
    :param model:
    :param data:
    :param n_fold_valid:
    :param mode_train: 'all'|'easiest' - which conditions to use in training
    :param to_debug:
    :param kwargs: fed to ykt.optimize()
    :return: best_loss, best_state
    """

    def fun_data1(mode='all', fold_valid=0, epoch=0, n_fold_valid=1):
        """
        :param mode:
        :param fold_valid:
        :param epoch:
        :return: (ev_cond_fr_dim_meanvar, durs), n_cond_dur_ch
        """
        return fun_data(data=data,
                        mode=mode, fold_valid=fold_valid, epoch=epoch,
                        n_fold_valid=n_fold_valid,
                        mode_train=mode_train, to_debug=to_debug)

    kw_optim = argsutil.kwdefault(
        argsutil.kwdef({
            'n_fold_valid': n_fold_valid
        }, kwargs),
        filename_suffix='',
        optimizer_kind='Adam',
        learning_rate=.5,
        patience=100,
        max_epoch=max_epoch,
        reduce_lr_after=25,
        reset_lr_after=50,
        thres_patience=1e-4,
        to_print_grad=False
    )

    def plot_coefs_dur_odif1(model, d):
        fig = plt.figure('coefs_dur_odif', figsize=[6, 4])
        axs = None
        axs = plot_coefs_dur_odif(npy(d['out_train_valid']), data, style='pred',
                            axs=axs, fig=fig)[2]
        axs = plot_coefs_dur_odif(npy(d['target_train_valid']), data,
                            style='data', axs=axs, fig=fig)[2]
        return fig, d

    def plot_ch_ev_by_dur1(model, d):
        fig = plt.figure('ch_ev_by_dur', figsize=[6, 4])
        axs = None
        axs = plot_ch_ev_by_dur(npy(d['out_train_valid']), data,
                                style='pred', axs=axs, fig=fig)
        axs = plot_ch_ev_by_dur(npy(d['target_train_valid']), data,
                                style='data', axs=axs, fig=fig)
        return fig, d

    def plot_unabs1(model, d):
        fig = plt.figure('unabs', figsize=[3, 7])
        plot_unabs(model, d['data_train_valid'])
        return fig, d

    plotfuns = [
            ('coefs_dur_odif', plot_coefs_dur_odif1),
            ('ch_ev_by_dur', plot_ch_ev_by_dur1),
            # ('bound', sim2d.plot_bound),
            ('unabs', plot_unabs1),
            ('params', sim2d.plot_params)
    ]

    best_loss, best_state, d = ykt.optimize(
        model, fun_data1, fun_loss,
        plotfuns=plotfuns,
        **kw_optim
    )[:3]

    with torch.no_grad():
        for data_mode in ['train_valid', 'test', 'all']:
            inp, target = fun_data1(data_mode)
            out = model(inp)

            for loss_kind in ['CE', 'NLL', 'BIC']:
                if loss_kind == 'CE':
                    loss = fun_loss(out, target, to_average=True,
                                    base_n_bin=True)
                elif loss_kind in ['NLL', 'BIC']:
                    loss = fun_loss(out, target, to_average=False,
                                    base_n_bin=False)
                if loss_kind == 'BIC':
                    n = npy(target.sum())
                    k = np.sum([
                        v.numel() if v.requires_grad else 0
                        for v in model.parameters()
                    ])
                    loss = loss * 2 + k * np.log(n)
                    d['loss_ndata_%s' % data_mode] = n
                    d['loss_nparam'] = k

                d['loss_%s_%s' % (loss_kind, data_mode)] = loss

    return best_loss, best_state, d, plotfuns
