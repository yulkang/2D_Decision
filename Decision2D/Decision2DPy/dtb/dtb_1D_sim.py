#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
import time
import numpy_groupies as npg
from typing import Iterable, Union, Sequence, Tuple, Dict, List
from copy import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, OneHotCategorical

from data_2d.consts import get_kw_plot
from lib.pylabyk import numpytorch as npt, yktorch as ykt, localfile
from lib.pylabyk.numpytorch import npy, npys
from lib.pylabyk import argsutil, plt2, np2

from data_2d import consts


def ____Utils____():
    pass


def get_demo_ev(n_cond=9, nt=consts.NT, dt=consts.DT):
    levels = 0.5 * np.exp(-np.log(2)
                          * np.arange(n_cond // 2))
    levels = set(-levels).union(set(levels))
    if n_cond % 2 == 1:
        levels = levels.union({0.})
    levels = torch.tensor(np.sort(list(levels), 0))
    ev = levels[:, None] + torch.zeros(1, nt)
    return ev


def simulate_p_cond__rt_ch(p_cond__rt_ch, n_sample=1):
    """

    @param p_cond__rt_ch: [condition, frame, ch]
    @type p_cond__rt_ch: torch.Tensor
    @param ev:
    @return: p_ch_rt_sim[condition, frame, ch]
    @rtype: torch.Tensor
    """
    p_cond__rt_ch_sim = OneHotCategorical(
        p_cond__rt_ch.reshape([p_cond__rt_ch.shape[0], -1])
    ).sample([n_sample]).sum(0).reshape(p_cond__rt_ch.shape)

    # print((p_rt_ch.shape, p_rt_ch_sim.shape))
    return p_cond__rt_ch_sim


def rt_sec2fr(rt_sec: np.ndarray, dt=consts.DT, nt=consts.NT):
    return np.clip(rt_sec // dt, 1, nt).astype(np.long)


def subsample_ev(ev: torch.Tensor, subsample_factor: int = 1):
    """
    :param ev: [cond, frame, [dim]]
    :param subsample_factor: how many consecutive frames to pool into a
    new_frame
    :return: mean_ev[cond, new_frame, dim], var_ev[cond, new_frame, dim]
    """
    shape0 = ev.shape
    nt0 = ev.shape[1]
    assert nt0 % subsample_factor == 0
    nt1 = int(nt0 // subsample_factor)
    ev1 = ev.view(shape0[0], nt1, subsample_factor, *shape0[2:])
    return ev1.mean(2), ev1.var(2)


def aggregate_rt_ch(
        cond1, rt1, ch1, n_cond,
        nt=consts.NT,
        n_ch=consts.N_CH,
):
    """
    @param rt_frame: [trial]
    @type rt_frame: torch.LongTensor
    @param ch: [trial]
    @type ch: torch.LongTensor
    @param cond: [trial]
    @type cond: torch.LongTensor
    @param nt: in frames
    @type nt: int
    @return: p_rt_ch[cond, rt_frame, ch]
    @rtype: torch.FloatTensor
    """
    # Use torch.index_add(dim, index, tensor) along with ravel()
    # see https://pytorch.org/docs/stable/tensors.html#torch.Tensor.index_add
    return torch.tensor(npg.aggregate(
        np.stack([cond1, rt1, ch1]), 1., 'sum',
        [n_cond, nt, n_ch]
    ))


def fun_loss(
        p_cond__rt_ch_pred: torch.Tensor,
        n_cond__rt_ch_data: torch.Tensor,
        to_average=True, base_n_bin=True,
        ix_conds_to_ignore_rt: torch.BoolTensor = None,
) -> torch.Tensor:
    """
    :param p_cond__rt_ch_pred: [condition, rt_frame, choice] = P(rt,ch|cond)
    :param n_cond__rt_ch_data: [condition, rt_frame, choice] = number of
      trials in that bin
    :param to_average:
    :param base_n_bin:
    :param ix_conds_to_ignore_rt:
    :return: cross entropy (if to_average=True) or NLL (otherwise)
    """
    if ix_conds_to_ignore_rt is None:
        ix_conds_to_ignore_rt = torch.zeros(
            n_cond__rt_ch_data.shape[0], dtype=torch.bool)

    loss = 0.
    if torch.any(~ix_conds_to_ignore_rt):
        loss = loss + fun_loss_resp(
            p_cond__rt_ch_pred[~ix_conds_to_ignore_rt],
            n_cond__rt_ch_data[~ix_conds_to_ignore_rt],
            to_average=to_average, base_n_bin=base_n_bin
        )
    if torch.any(ix_conds_to_ignore_rt):
        loss = loss + fun_loss_resp(
            p_cond__rt_ch_pred[ix_conds_to_ignore_rt].sum(1),
            n_cond__rt_ch_data[ix_conds_to_ignore_rt].sum(1),
            to_average=to_average, base_n_bin=base_n_bin
        )
    return loss


def fun_loss_resp(
        p_cond_pred: torch.Tensor,
        n_cond_data: torch.Tensor,
        to_average=True, base_n_bin=True,
) -> torch.Tensor:
    """
    :param p_cond_pred: [condition, response...], e.g.,
      [condition, rt_frame, choice] = P(rt,ch|cond) or
      [condition, choice] = P(ch|cond)
    :param n_cond_data: [condition, response...], e.g.,
      [condition, rt_frame, choice] or
      [condition, choice] = number of trials in that bin
    :return: cross entropy (if to_average=True) or NLL (otherwise)
    """
    n_bin = np.prod(n_cond_data.shape[1:]).astype(np.float)
    loss = -torch.sum(
        n_cond_data
        * torch.log(p_cond_pred + 1e-12 / n_bin)
    )
    if to_average:
        loss = loss / torch.sum(n_cond_data)
    if base_n_bin:
        loss = loss / np.log(n_bin)
    return loss


def ____Model_Classes____():
    pass


class Timer:
    def __init__(self, dt=consts.DT, nt=consts.NT, **kwargs):
        """

        :param dt:
        :param nt:
        :param kwargs:
        """
        self.dt = dt
        self.nt = nt

    def get_t_all(self, dt=None, nt=None):
        if dt is None:
            dt = self.dt
        if nt is None:
            nt = self.nt
        return torch.arange(nt) * dt

    @property
    def t_all(self):
        return self.get_t_all()


class TimedModule(ykt.BoundedModule):
    def __init__(self, timer: Timer = None, share_time=True, **kwargs):
        """

        :param timer:
        :param share_time:
        :param kwargs:
        """
        super().__init__()
        if timer is None:
            timer = Timer(**kwargs)
        self.timer = timer

    @property
    def dt(self):
        return self.timer.dt

    @dt.setter
    def dt(self, v):
        self.timer.dt = v

    @property
    def nt(self):
        return self.timer.nt

    @nt.setter
    def nt(self, v):
        self.timer.nt = v

    @property
    def t_all(self):
        return self.timer.t_all

    @property
    def get_t_all(self):
        return self.timer.get_t_all


class Tnd(TimedModule):
    kind = 'None'
    loc_kind = 'None'
    disper_kind = 'None'

    def __init__(
            self,
            tnd_loc0=0.2,
            tnd_loc_lb=0.1,
            tnd_loc_ub=0.5,
            tnd_disper0=0.1,
            tnd_disper_lb=0.01,
            tnd_disper_ub=0.3,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.loc = ykt.BoundedParameter(
            [tnd_loc0], tnd_loc_lb, tnd_loc_ub)
        self.disper = ykt.BoundedParameter(
            [tnd_disper0], tnd_disper_lb, tnd_disper_ub)

    def get_p_tnd(self, t_all: Sequence[float] = None) -> torch.Tensor:
        raise NotImplementedError()

    def plot_p_tnd(self,
                   t_all: Sequence[float] = None,
                   ax: plt.Axes = None,
                   **kwargs
                   ) -> Union[plt.Line2D, Sequence[plt.Line2D]]:
        if t_all is None:
            t_all = self.t_all
        if ax is None:
            ax = plt.gca()

        p_tnd = self.get_p_tnd(t_all)

        kwargs = argsutil.kwdefault(
            kwargs,
            color='k',
            linestyle='-'
        )
        h = ax.plot(*npys(t_all, p_tnd), **kwargs)
        ax.set_xlabel('time (s)')
        plt2.box_off(ax=ax)
        # ax.set_ylim(ymin=0)
        # ax.set_xlim(xmin=0)
        plt2.detach_axis('y', amin=0, amax=1, ax=ax)
        plt2.detach_axis('x', amin=0, ax=ax)
        return h


class TndInvNorm(Tnd):
    kind = 'invnorm'
    loc_kind = 'mean'
    disper_kind = 'cv'  # coefficient of variation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_p_tnd(self, t_all: Sequence[float] = None) -> torch.Tensor():
        if t_all is None:
            t_all = consts.T_TENSOR
        stdev = self.loc[:] * self.disper[:]

        p = npt.inv_gaussian_pmf_mean_stdev(
            t_all, self.loc[:], stdev[:])

        return p


class TndLogNorm(Tnd):
    kind = 'lognorm'
    loc_kind = 'mean'
    disper_kind = 'cv'  # coefficient of variation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_p_tnd(self, t_all: Sequence[float] = None) -> torch.Tensor():
        if t_all is None:
            t_all = consts.T_TENSOR
        stdev = self.loc[:] * self.disper[:]

        p = npt.lognorm_pmf(t_all, self.loc[:], stdev[:])
        return p


class Bound(TimedModule):
    kind = 'const'

    def __init__(self, bound0=1.0, bound0_lb=0.5, **kwargs):
        super().__init__(**kwargs)
        self.b = ykt.BoundedParameter([bound0], bound0_lb, 2.)

    def get_bound(self, t_all: torch.Tensor = None) -> torch.Tensor:
        if t_all is None:
            t_all = self.t_all
        return torch.zeros_like(t_all) + self.b[0]


class BoundExp(Bound):
    kind = 'exp'

    def __init__(
            self,
            *args,
            bound_t_st0=0.4,
            # t_st_lb was previously 0.05, but 0.1 seems to help distinguish
            # it from lower bound height
            bound_t_st_lb=0.1,
            bound_t_st_ub=0.8,
            bound_t_half0=0.5,
            bound_t_half_lb=0.05,
            # previously 0.25, but seems to have similar effect with t_half
            # and t_st combined, so maybe fix to 0
            bound_asymptote_max=0.,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.b_t_st = ykt.BoundedParameter([bound_t_st0],
                                           bound_t_st_lb, bound_t_st_ub)
        self.b_t_half = ykt.BoundedParameter([bound_t_half0],
                                             bound_t_half_lb, 1.0)
        self.b_asymptote = ykt.BoundedParameter(
            [bound_asymptote_max / 2],
            bound_asymptote_max * 1e-6, bound_asymptote_max
        )

    def get_bound(self, t_all: torch.Tensor = None) -> torch.Tensor:
        if t_all is None:
            t_all = self.t_all
        return (torch.exp(
            -(t_all - self.b_t_st[0]) / self.b_t_half[0]
            * torch.log(torch.tensor(2.))
            ).clamp_max(1.) * (1 - self.b_asymptote[0])
            + self.b_asymptote[0]
        ) * self.b[0]


class BoundWeibull(Bound):
    kind = 'weibull'

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.b_scale = ykt.BoundedParameter([0.3], 0.1, 0.5)
        self.b_shape = ykt.BoundedParameter([5.], 0., 20.)
        self.b_asymptote = ykt.BoundedParameter([0.25], 0., 0.5)

    def get_bound(self, t_all: torch.Tensor = None) -> torch.Tensor:
        if t_all is None:
            t_all = self.t_all
        t_all = t_all / self.t_all[-1] / self.b_scale[:]

        incl1 = t_all > 0
        t_all1 = t_all[incl1]

        # To prevent NaN grad for shape
        b = torch.ones_like(t_all)
        b[incl1] = (
            (-(t_all1 ** self.b_shape[:])).exp()
            * (torch.tensor(1.) - self.b_asymptote[:]) + self.b_asymptote[:]
        )
        b = b * self.b[:]
        return b


class Dtb1D(TimedModule):
    def __init__(
            self,
            kb2_0=50.,
            # bperk0=1. / 20.,
            # kappa0=40.,
            bound: Bound = BoundExp,
            bias0=0.,
            diffusion=1.,
            y0=0.,
            ssq0=1e-3,
            n_ev=2 ** 7 + 1,
            max_ev=3,
            **kwargs
    ):
        """

        :param kappa0:
        :param bound0:
        :param bound_t_st0:
        :param bound_t_half0:
        :param bound_asymptote_max:
        :param bias0:
        :param diffusion:
        :param y0:
        :param ssq0:
        :param n_ev:
        :param max_ev:
        :param dt:
        :param nt: default used for plotting
        """
        super().__init__(**kwargs)

        assert n_ev % 2 == 1  # for padding in conv1d to work
        self.n_ev = n_ev
        self.max_ev = max_ev
        self.ev_bin = torch.linspace(-self.max_ev, self.max_ev, self.n_ev)
        self.dev = self.ev_bin[1] - self.ev_bin[0]

        # mean logit when coherence = 1
        self.kb2 = ykt.BoundedParameter([kb2_0], 0.01, 100.)
        self.bias = ykt.BoundedParameter([bias0], -0.1, 0.1)

        bound = bound(**{**kwargs, 'timer': self.timer})

        # boundmod: bound module (cannot use _bound or bound
        # for backward compatibility)
        self.bound = bound
        # self.bound = self._bound.bound

        self.diffusion = ykt.BoundedParameter([diffusion], 0.99, 1.01)

        # y0 / 2B = guess for choice 1 in excess of 0.5 when mu = 0
        #   therefore, y0 should be parameterized realtive to B
        self.bias_t0 = ykt.BoundedParameter([0.], -0.1, 0.1)
        self.ssq0 = ykt.BoundedParameter([ssq0], 1e-4, 1e-2)

        self.max_ev_kernel = (
            # max diffusion + max drift
            np.sqrt(diffusion * self.dt) * 3.5 + 0.5 * 50 * self.dt
        )
        self.ev_bin_kernel = self.ev_bin[
            torch.abs(self.ev_bin) < self.max_ev_kernel
        ]

    @property
    def y0(self):
        return self.bias_t0[:] * self.bound.b[:] * 2.

    @y0.setter
    def y0(self, v):
        self.bias_t0[:] = v / (2. * self.bound.b[:])


    def get_bound(self, *args, **kwargs):
        return self.bound.get_bound(*args, **kwargs)

    @property
    def kappa(self):
        return self.kb2[:] / torch.tensor(2.) / self.bound.b[:]
        # return (self.kb2[:] / torch.tensor(2.) / self.bperk[:]).sqrt()

    def forward(
            self, ev: torch.Tensor,
            return_unabs=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :param ev: [condition, frame, (mean, var)]
        :param return_unabs:
        :return: p_cond_td_ch[cond, td, ch], unabs_td_cond_ev[td, cond, ev]
        """

        # need to consider var_ev when evidence is subsampled.
        ev, var_ev = ev.permute([2, 0, 1])

        nt = ev.shape[1]
        t_all = self.get_t_all(nt=nt)
        n_cond = ev.shape[0]
        batch_shape = ev.shape[:0]  # empty shape for now

        ev = npt.p2st(ev)
        var_ev = npt.p2st(var_ev)
        var0 = self.diffusion[:]
        norm_kernel = Normal(loc=0., scale=1.)
        ev_bin_kernel = self.ev_bin_kernel.expand(
            torch.Size([1] * (1 + len(batch_shape)) + [1, -1])
        )
        pad = int(ev_bin_kernel.shape[-1] // 2)

        p_absorbed = torch.empty(
            torch.Size([nt])
            + batch_shape
            + torch.Size([n_cond, 2])
        )

        def get_mask(t):
            b = self.get_bound(t)
            mask_abs = torch.stack([
                torch.clamp(
                    (-b - self.ev_bin) / self.dev,
                    0., 1.
                ),  # mask_down
                torch.clamp(
                    (self.ev_bin - b) / self.dev,
                    0., 1.
                )  # mask_up
            ], -1)  # [ch, ev]
            mask_in = (
                    (1. - npt.p2st(mask_abs)[0])  # noqa
                    * (1. - npt.p2st(mask_abs)[1])  # noqa
            )
            return mask_abs, mask_in

        if return_unabs:
            unabs_td_cond_ev = []

        # y0 has nonzero gradient with non-negligible dispersion
        pev = torch.exp(
                Normal(loc=self.y0[:],
                       scale=torch.sqrt(
                           self.ssq0[:])
                       ).log_prob(self.ev_bin)
            ).expand(
                batch_shape + torch.Size([n_cond] + [-1])
        ).unsqueeze(0)  # [1, cond, ev]
        # when var_ev1 == 0, reduces to the original formulation.

        # # No absorption at t=0
        pev = npt.sumto1(pev, -1)

        for it, (t, ev1, var_ev1) in enumerate(zip(t_all, ev, var_ev)):
            mask_abs, mask_in = get_mask(t)

            if it > 0:  # no data at time 0
                # NOTE: Trying bias before kappa to match scale
                kernel = npt.sumto1(
                    torch.exp(norm_kernel.log_prob(
                        ((ev1[:, None, None] - self.bias[:])
                         * self.kappa[:] * self.dt
                         + ev_bin_kernel)
                        / torch.sqrt(self.dt * (
                                self.kappa[:] ** 2 * var_ev1[:, None, None]
                                + var0
                        ))
                    )), -1)  # [cond, 1, ev]

                # # NOTE: Apply bias after kappa, to avoid interaction
                # kernel = npt.sumto1(
                #     torch.exp(norm_kernel.log_prob(
                #         ((ev1[:, None, None]
                #          * self.kappa[:] * self.dt
                #          + ev_bin_kernel + self.bias[:]))
                #         / torch.sqrt(self.dt * (
                #                 self.kappa[:] ** 2 * var_ev1[:, None, None]
                #                 + var0
                #         ))
                #     )), -1)  # [cond, 1, ev]

                # kernel = npt.sumto1(
                #     torch.exp(norm_kernel.log_prob(
                #         ((ev1[:, None, None] + self.bias[:])
                #          * self.kappa[:] * self.dt
                #          + ev_bin_kernel)
                #         / torch.sqrt(self.dt * (
                #                 self.kappa[:] ** 2 * var_ev1[:, None, None]
                #                 + var0
                #         ))
                #     )), -1)  # [cond, 1, ev]
                pev = F.conv1d(
                    pev,  # [1, cond, ev]
                    kernel,  # [cond, 1, ev]
                    groups=n_cond,
                    padding=pad
                )

            a = torch.sum(
                pev.unsqueeze(-1) * mask_abs[None, None, :], -2
            ).squeeze(-3)  # [cond, ch]

            p_absorbed[it] = a  # [fr, cond, ch]
            pev = pev * mask_in[None, None, :]

            if return_unabs:
                unabs_td_cond_ev.append(pev.clone())  # noqa

        p_cond_td_ch = p_absorbed.permute([1, 0, 2])

        if return_unabs:
            unabs_td_cond_ev = torch.cat(unabs_td_cond_ev, 0)
            return p_cond_td_ch, unabs_td_cond_ev
        else:
            return p_cond_td_ch

    def plot_bound(self, t_all:Sequence[float]=None, ax:plt.Axes=None, **kwargs
                   ) -> plt.Line2D:
        if ax is None:
            ax = plt.gca()
        if t_all is None:
            t_all = self.t_all

        kwargs = argsutil.kwdefault(
            kwargs,
            color='k',
            linestyle='-'
        )
        h = ax.plot(*npys(t_all, self.get_bound(t_all)),
                    **kwargs)
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r"$b(t)$")
        y_lim = ax.get_ylim()
        y_min = -y_lim[1] * 0.05
        ax.set_ylim(ymin=y_min)
        plt2.detach_axis('y', amin=0)
        plt2.detach_axis('x', amin=0)
        plt2.box_off()
        return h

    def plot_unabs(
            self, unabs_td_ev: np.ndarray, p_td_ch: np.ndarray = None,
            prcts=(25., 50., 75.),
    ):
        nt = unabs_td_ev.shape[0]
        t = np.arange(nt) * self.dt
        ev = self.ev_bin

        b = npy(self.get_bound(torch.tensor(t)))
        b_max = np.amax(b)

        unabs_ev = unabs_td_ev.sum(0)
        iev_unabs_max = np.amax(np.nonzero(unabs_ev)[0])
        iev_unabs_min = np.amin(np.nonzero(unabs_ev)[0])
        # ev_unabs_max = ev[iev_unabs_max]
        dev = ev[1] - ev[0]
        iev_bound_max = int(np.clip(iev_unabs_max + 2, 0, len(ev) - 1))
        iev_bound_min = int(np.clip(iev_unabs_min - 2, 0, len(ev) - 1))

        ev_max = ev[iev_bound_max] + dev / 2
        ev_min = ev[iev_bound_min] - dev / 2

        extent = [t[0], t[-1], ev_min, ev_max]
        unabs_plot = unabs_td_ev.copy()[:, iev_bound_min:iev_bound_max]
        if p_td_ch is not None:
            unabs_plot[:, 0] += p_td_ch[:, 0]
            unabs_plot[:, -1] += p_td_ch[:, 1]

        unabs_plot = np2.sumto1(unabs_plot, 1)
        plt.imshow(npy(unabs_plot.T), extent=[
            npy(v) for v in extent])

        plt.plot(t, b, 'w-')
        plt.plot(t, -b, 'w-')

    def _load_from_state_dict(
            self, state_dict, *args, **kwargs):
        # ---- For backward compatibility: put bound*._* into bound.bound*._*
        state_dict = self.update_state_dict(state_dict)
        return super()._load_from_state_dict(state_dict, *args,
                                             **kwargs)

    def load_state_dict(self,
                        state_dict,
                        strict: bool = ...):
        state_dict = self.update_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    @staticmethod
    def update_state_dict(state_dict: dict):  # , prefix=''):
        """
        Call from the top level module by overriding load_state_dict()
        Example:
        def load_state_dict(self, state_dict, *args, **kwargs):
            state_dict = Dtb1D.update_state_dict(state_dict)
            super().load_state_dict(state_dict, *args, **kwargs)

        :param state_dict:
        :return:
        """
        keys = list(state_dict.keys())
        state_dict = copy(state_dict)  # type: Dict[str, nn.Parameter]
        for k0 in keys:  # type: str
            # convert scalar to length 1 vector, as in the new convention
            if k0.endswith('_param') and state_dict[k0].ndim == 0:
                state_dict[k0] = torch.tensor([state_dict[k0]])

            # convert to the new name
            if (k0.find('.bound') != -1
                    and k0.find('.bound.b') == -1):
                k1 = k0.replace('.bound', '.bound.b')
                state_dict[k1] = state_dict.pop(k0)

        return state_dict


class Lapse(TimedModule):
    kind = 'None'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, p_cond_rt_ch: torch.Tensor) -> torch.Tensor:
        """
        :param p_cond_rt_ch: [cond, rt, ch]
        :return: p_cond_rt_ch: [cond, rt, ch]
        """
        raise NotImplementedError()


class LapseFlip(Lapse):
    kind = 'flip'

    def __init__(self, lapse_max=0.1, **kwargs):
        super().__init__(**kwargs)
        self.p_lapse = ykt.BoundedParameter([lapse_max / 2], 1e-6, lapse_max)

    def forward(self, p_cond_rt_ch: torch.Tensor) -> torch.Tensor:
        """
        :param p_cond_rt_ch: [cond, rt, ch]
        :return: p_cond_rt_ch: [cond, rt, ch]
        """
        p_cond_rt_ch = (
                p_cond_rt_ch * (1. - self.p_lapse[:])
                + p_cond_rt_ch.flip(-1) * self.p_lapse[:]
        )
        return p_cond_rt_ch


class LapseUniform(Lapse):
    kind = 'uniform'

    def __init__(self, lapse_max=0.1, **kwargs):
        super().__init__(**kwargs)
        self.p_lapse = ykt.BoundedParameter([lapse_max / 2], 1e-6, lapse_max)

    def forward(self, p_cond_rt_ch: torch.Tensor) -> torch.Tensor:
        p_cond_rt_ch = (
            p_cond_rt_ch * (1. - self.p_lapse[:])
            + self.p_lapse[:]
            * p_cond_rt_ch.sum([-1, -2], keepdim=True)
            / np.prod(p_cond_rt_ch.shape[1:])
            # + self.p_lapse[:] / np.prod(p_cond_rt_ch.shape[1:])
        )
        return p_cond_rt_ch


class LapseUniformRT(Lapse):
    kind = 'uniform_rt'

    def __init__(self, lapse_max=0.1, **kwargs):
        super().__init__(**kwargs)
        self.p_lapse = ykt.BoundedParameter([lapse_max / 2], 1e-6, lapse_max)

    def forward(self, p_cond_rt_ch: torch.Tensor) -> torch.Tensor:
        p_cond_rt_ch = (
            p_cond_rt_ch * (1. - self.p_lapse[:])
            + self.p_lapse[:] / p_cond_rt_ch.shape[1]
            * p_cond_rt_ch.sum(1, keepdim=True)
        )
        return p_cond_rt_ch


def ____Plot____():
    pass


def plot_p_ch_vs_ev(ev_cond, p_ch, style='pred',
                    ax: plt.Axes = None,
                    **kwargs
                    ) -> plt.Line2D:
    """
    @param ev_cond: [condition] or [condition, frame]
    @type ev_cond: torch.Tensor
    @param p_ch: [condition, ch] or [condition, rt_frame, ch]
    @type p_ch: torch.Tensor
    @return:
    """
    if ax is None:
        ax = plt.gca()
    if ev_cond.ndim != 1:
        if ev_cond.ndim == 3:
            ev_cond = npt.p2st(ev_cond)[0]
        assert ev_cond.ndim == 2
        ev_cond = ev_cond.mean(1)
    if p_ch.ndim != 2:
        assert p_ch.ndim == 3
        p_ch = p_ch.sum(1)

    kwargs = get_kw_plot(style, **kwargs)

    h = ax.plot(*npys(
        ev_cond,
        npt.p2st(npt.sumto1(p_ch, -1))[1]
    ), **kwargs)
    plt2.box_off(ax=ax)
    x_lim = ax.get_xlim()
    plt2.detach_axis('x', amin=x_lim[0], amax=x_lim[1], ax=ax)
    plt2.detach_axis('y', amin=0, amax=1, ax=ax)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '', '1'])
    ax.set_xlabel('evidence')
    ax.set_ylabel(r"$\mathrm{P}(z=1 \mid c)$")
    return h


def choose_correct_ch(n_cond__rt_ch):
    """
    :param n_cond__rt_ch: [condition, frame, ch]
    :return: n_cond__rt_correct_ch[cond, frame]
    """
    n_cond__rt_ch = npy(n_cond__rt_ch)
    n_cond__ch = n_cond__rt_ch.sum(1)

    n_cond = n_cond__rt_ch.shape[0]
    correct_ch = np.argmax(n_cond__ch, axis=1)
    return n_cond__rt_ch[np.arange(n_cond), :, correct_ch]


def plot_rt_vs_ev(
        ev_cond, n_cond__rt_ch: Union[torch.Tensor, np.ndarray],
        style='pred',
        pool='mean', dt=consts.DT,
        correct_only=True,
        thres_n_trial=10,
        color='k',
        color_ch=('tab:red', 'tab:blue'),
        ax: plt.Axes = None,
        kw_plot=(),
) -> (Sequence[plt.Line2D], List[np.ndarray]):
    """
    @param ev_cond: [condition]
    @type ev_cond: torch.Tensor
    @param n_cond__rt_ch: [condition, frame, ch]
    @type n_cond__rt_ch: torch.Tensor
    @return:
    """
    if ax is None:
        ax = plt.gca()
    if ev_cond.ndim != 1:
        if ev_cond.ndim == 3:
            ev_cond = npt.p2st(ev_cond)[0]
        assert ev_cond.ndim == 2
        ev_cond = ev_cond.mean(1)
    assert n_cond__rt_ch.ndim == 3

    ev_cond = npy(ev_cond)
    n_cond__rt_ch = npy(n_cond__rt_ch)

    def plot_rt_given_cond_ch(ev_cond1, n_rt_given_cond_ch, **kw1):
        # n_rt_given_cond_ch[cond, fr]
        # p_rt_given_cond_ch[cond, fr]
        p_rt_given_cond_ch = np2.sumto1(n_rt_given_cond_ch, 1)

        nt = n_rt_given_cond_ch.shape[1]
        t = np.arange(nt) * dt

        # n_in_cond_ch[cond]
        n_in_cond_ch = n_rt_given_cond_ch.sum(1)
        if pool == 'mean':
            rt_pooled = (t[None, :] * p_rt_given_cond_ch).sum(1)
        elif pool == 'var':
            raise NotImplementedError()
        else:
            raise ValueError()

        if style.startswith('data'):
            rt_pooled[n_in_cond_ch < thres_n_trial] = np.nan

        kw = get_kw_plot(style, **kw1)
        h = ax.plot(ev_cond1, rt_pooled, **kw)
        return h, rt_pooled

    if correct_only:
        hs = []
        rtss = []

        n_cond__rt_ch1 = n_cond__rt_ch.copy()  # type: np.ndarray
        if style.startswith('data'):
            cond0 = ev_cond == 0
            n_cond__rt_ch1[cond0, :, :] = np.sum(
                n_cond__rt_ch1[cond0, :, :], axis=-1, keepdims=True)

        # # -- Choose the ch with correct sign (or both chs if cond == 0)
        for ch in range(consts.N_CH):
            cond_sign = np.sign(ev_cond)
            ch_sign = consts.ch_bool2sign(ch)
            cond_accu = cond_sign != -ch_sign

            h, rts = plot_rt_given_cond_ch(
                ev_cond[cond_accu], n_cond__rt_ch1[cond_accu, :, ch],
                color=color, **dict(kw_plot)
            )
            hs.append(h)
            rtss.append(rts)
        # rts = np.stack(rtss)
        rts = rtss

    else:
        hs = []
        rtss = []
        for ch in range(consts.N_CH):
            # n_rt_given_cond_ch[cond, fr]

            n_rt_given_cond_ch = n_cond__rt_ch[:, :, ch]

            h, rts = plot_rt_given_cond_ch(
                ev_cond,
                n_rt_given_cond_ch, color=color_ch[ch])

            hs.append(h)
            rtss.append(rts)
        rts = np.stack(rtss)

    y_lim = ax.get_ylim()
    x_lim = ax.get_xlim()
    plt2.box_off()
    plt2.detach_axis('x', ax=ax, amin=x_lim[0], amax=x_lim[1])
    plt2.detach_axis('y', ax=ax, amin=y_lim[0], amax=y_lim[1])
    ax.set_xlabel('evidence')
    ax.set_ylabel(r"$\mathrm{E}[T^\mathrm{r} \mid c]~(\mathrm{s})$")
    return hs, rts


def ____Main____():
    pass


if __name__ == '__main__':
    torch.set_default_dtype(torch.double)


def save_fit_results(model, best_state, d, plotfuns,
                     locfile: localfile.LocalFile,
                     dict_cache: dict,
                     subdir: Union[str, dict]) -> Iterable[str]:
    def fun_file(kind, ext):
        return locfile.get_file('tab', kind, dict_cache, ext=ext,
                                subdir=subdir)

    def fun_fig_file(kind, ext):
        return locfile.get_file_fig(kind, dict_cache, ext=ext,
                                    subdir=subdir)

    files = ykt.save_optim_results(
        model, best_state, d, plotfuns,
        fun_tab_file=fun_file, fun_fig_file=fun_fig_file)
    return files