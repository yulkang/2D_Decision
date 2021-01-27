#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
from scipy.io import loadmat

from lib.pylabyk import np2
from lib.pylabyk.np2 import filt_dict

from data_2d import consts

#%%
def ____LOAD____():
    pass


file_matlab_combined = '../../data/orig/data_RT_VD.mat'
dat = loadmat(file_matlab_combined)
print('Loaded %s' % file_matlab_combined)
for key in ['parads', 'subjs']:
    d = dat[key]
    d2 = [s[0] for s in d.flatten()]
    dat[key] = d2

for key in dat.keys():
    if key.startswith('id_'):
        dat[key] -= 1  # MATLAB starts with 1; Python starts with 0

for key, val in dat.items():
    if (isinstance(val, np.ndarray)
            and val.ndim == 2
            and val.shape[1] == 1):
        dat[key] = val.flatten()

dat['dim_rel'] = dat['dim_rel'].astype(np.bool)
dat['to_excl'] = dat['to_excl'].astype(np.bool)


def load_data_combined():
    return dat


def load_data_parad(parad='VD'):
    raise NotImplementedError()
