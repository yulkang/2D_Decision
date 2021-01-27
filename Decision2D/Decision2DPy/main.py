#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

#%%
from dtb.RT import plot_nonparam_RT_MATLAB as nonparam_rt
from dtb.RT import plot_nonparam_RT_recovery_MATLAB as \
    nonparam_rt_recover
from dtb.VD import dtb_2D_fit_VD as fit_VD
from dtb.VD import dtb_2D_recover_VD as recover_VD

nonparam_rt.main()
nonparam_rt_recover.main()
fit_VD.main()
recover_VD.main()