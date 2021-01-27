#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
import torch
from lib.pylabyk import plt2, argsutil

N_CH = 2  # number of options per dimension
REFRESH_RATE = 75
DT = 1 / REFRESH_RATE
T_MAX = 5
NT = int(T_MAX / DT)

# Starting from t=0 gives NaN gradient, so start from DT instead
T_ALL = np.arange(NT) * DT
T_TENSOR = torch.arange(NT) * DT

N_DIM = 2
DIM_NAMES_LONG = ['Motion', 'Color']
DIM_NAMES_SHORT = ['M', 'C']
CH_NAMES = [['left', 'right'], ['yellow', 'blue']]

TASK2NAME = {
        'A': '2D',
        'H': '1D Motion',
        'V': '1D Color'
    }

XTICKLABELS = [['strong\nleft', 'weak\nmotion', 'strong\nright'],
               ['strong\nyellow', 'weak\ncolor', 'strong\nblue']]
CMAP_DIM = [plt2.winter2_rev, plt2.cool2_rev]

# [dim, ch_flat] = ch_dim
CHS = ((0, 0, 1, 1), (0, 1, 0, 1))

# [dim, ch_flat] = ch_dim
CHS_TENSOR = torch.tensor(CHS)
CHS_ARRAY = np.array(CHS)

N_CH_FLAT = len(CHS[0])

SUBJS = {
    # --- New participant IDs
    'RT': ['S1', 'S2', 'S3'],
    'sh': ['S1', 'S2', 'S3'],
    'VD': ['S4', 'S5'],
    'eye': ['S1', 'S2', 'S3', 'S4', 'S5'],
    'MANUAL': ['S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'],
    'unimanual': ['S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'],
    'bimanual': ['S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'],
    'unibimanual': ['S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'],
}

SUBJS_OLD = [
    'S1', 'S2', 'S3',
    'ID2', 'ID3',
    'ID4', 'ID5', 'ID6', 'ID7', 'ID9', 'ID16', 'ID18'
]

COHS_COLOR_MANUAL = np.array([0, 0.128, 0.257, 0.511, 1.025, 1.983])
COHS_COLOR = {
    'VD': {
        'S4':  [0.104, 0.209, 1.983],
        'S5':  [0.092, 0.209, 1.983],
    },
    'unibimanual': {
        subj: COHS_COLOR_MANUAL
        for subj in SUBJS['unibimanual']
    }
}


def logit2c(logit):
    return 2. * (np.exp(logit) / (1. + np.exp(logit))) - 1.


def subj_parad_old2new(subj_old: str, parad: str) -> str:
    return SUBJS[parad][SUBJS[parad].index(subj_old)]


SUBJS_VD = ['S4', 'S5']
SUBJS_RT_HUMAN = ['S1', 'S2', 'S3']

SUBJ_PARAD = [
     (subj, 'RT') for subj in SUBJS['RT']
] + [
    (subj, 'unimanual') for subj in SUBJS['MANUAL']
] + [
    (subj, 'bimanual') for subj in SUBJS['MANUAL']
]

SUBJ_PARAD_BI = []
for subj in SUBJS['RT']:
    SUBJ_PARAD_BI.append((subj, 'RT', False))
for bimanual in [False, True]:
    for subj in SUBJS['MANUAL']:
        if subj == 'ID2' and bimanual:
            continue
        SUBJ_PARAD_BI.append((subj, 'unibimanual', bimanual))


def parad_bi2parad(parad: str, bimanual: bool) -> str:
    """
    :param parad:
    :param bimanual:
    :return: parad ('unimanual', 'bimanual', etc.)
    """
    if parad == 'unibimanual':
        return 'bimanual' if bimanual else 'unimanual'
    else:
        return parad


def parad2parad_bi(parad: str) -> (str, bool):
    """
    :param parad:
    :return: parad ('unibimanual', etc.), bimanual
    """
    if parad == 'unimanual':
        return 'unibimanual', False
    elif parad == 'bimanual':
        return 'unibimanual', True
    else:
        return parad, False


def ch_by_dim2ch_flat(ch_by_dim: np.ndarray):
    """
    :param ch_by_dim: [tr, dim]
    :return: ch_flat[tr]
    """
    if ch_by_dim.ndim == 1:
        return ch_by_dim[0] * 2 + ch_by_dim[1]
    else:
        assert ch_by_dim.ndim == 2
        return ch_by_dim[:, 0] * 2 + ch_by_dim[:, 1]


def ch_flat2ch_by_dim(ch_flat, return_numpy=True):
    """
    :param ch_flat: [tr]
    :return: ch_by_dim[tr, dim]
    """
    if return_numpy:
        return np.stack([ch_flat // 2, ch_flat % 2], -1)
    else:
        return torch.stack([ch_flat // 2, ch_flat % 2], -1)


def ch_bool2sign(ch_bool):
    return np.sign(ch_bool - 0.5)


def ch_sign2bool(ch_sign):
    return (0.5 + 0.5 * ch_sign).astype(np.bool)


def get_odim(dim: int) -> int:
    return N_DIM - 1 - dim


class Style(object):
    axhlinestyle = {
        'color': 'grey',
        'linestyle': '--',
        'zorder': -1,
        'linewidth': 0.5
    }


def get_kw_plot(style='pred', color='k', for_err=False, **kwargs):
    if style == 'pred':
        kwargs = argsutil.kwdefault(
            kwargs,
            color=color,
            marker='None',
            linestyle='-',
            linewidth=0.75,
        )
    elif style == 'data':
        kwargs = argsutil.kwdefault(
            kwargs,
            marker='o',
            mfc=color,
            mec='w',
            linestyle='None',
            mew=0.75,
        )
    elif style == 'data_fit':
        kwargs = argsutil.kwdefault(
            kwargs,
            marker='o',
            mfc='w',
            mec=color,
            linestyle='None'
        )
    elif style == 'data_pred':
        kwargs = argsutil.kwdefault(
            kwargs,
            marker='o',
            mfc=color,
            mec='w',
            linestyle='None'
        )
    else:
        raise ValueError()

    if for_err:
        if style.startswith('data'):
            kwargs.update({
                'ecolor': kwargs['mfc'],
                'barsabove': True
            })
            if 'elinewidth' not in kwargs.keys():
                kwargs['elinewidth'] = kwargs['mew']

    return kwargs