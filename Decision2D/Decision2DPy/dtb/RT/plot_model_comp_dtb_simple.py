"""
Make bar plots of model comparison & recovery from the simple DTB model fits
to RT-style tasks.
"""

#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.


import csv
import os
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from dtb.RT.dtb_2D_recover_RT_nonparam \
    import plot_bar_dloss_across_subjs, axvline_dcost, xticks_serial_vs_parallel
from data_2d import consts
from lib.pylabyk import localfile, np2, plt2

locfile_in = localfile.LocalFile(
    pth_root='../../data/Data_2D_Py/dtb/RT_dtb_simple/orig_mat',
    cache_dir=''
)

locfile_out = localfile.LocalFile(
    pth_root='../../data/Data_2D_Py/dtb/RT_dtb_simple',
    cache_dir=''
)

use_easiest_only = 1

file_model_comp = 'model_comparison_delta_logl_predictions.mat'
files_model_recovery = [
    'model_comparison_delta_logl_predictions_groundtruth_serial.mat',
    'model_comparison_delta_logl_predictions_groundtruth_parallel.mat'
]


def main(base=10.):
    ds = load_comp(
        os.path.join(locfile_in.pth_root, file_model_comp))
    ds['dcost'] = np.array(ds['dcost']) / np.log(base)

    vmax = 160

    m = np.empty(3)
    e = np.empty(3)

    axs = plot_bar_dloss_across_subjs(
        dlosses=np.array(ds['dcost']),
        subj_parad_bis=ds['subj_parad_bi'],
        vmax=vmax
    )
    axs = plt2.GridAxes(
        nrows=1, ncols=3,
        heights=axs.heights, widths=[2],
        left=axs.left, right=axs.right, bottom=axs.bottom
    )
    plot_bar_dloss_across_subjs(
        dlosses=np.array(ds['dcost']),
        subj_parad_bis=ds['subj_parad_bi'],
        vmax=vmax,
        axs=axs[:, [0]],
        base=base,
    )
    plt.title('Data')

    m[0] = np.mean(ds['dcost'])
    e[0] = np2.sem(ds['dcost'])

    print('--')

    titles = ['Simulated\nSerial', 'Simulated\nParallel']
    for i in range(2):
        ds = load_comp(
            os.path.join(locfile_in.pth_root,
                         files_model_recovery[i]))
        ds['dcost'] = np.array(ds['dcost']) / np.log(base)

        plot_bar_dloss_across_subjs(
            dlosses=ds['dcost'],
            subj_parad_bis=ds['subj_parad_bi'],
            vmax=vmax,
            axs=axs[:, [i + 1]],
            base=base,
        )
        plt.title(titles[i])

        m[i + 1] = np.mean(ds['dcost'])
        e[i + 1] = np2.sem(ds['dcost'])

        plt.sca(axs[0, i + 1])
        plt2.box_off(['left'])
        plt.yticks([])

    for ext in ['.pdf', '.png']:
        file = locfile_out.get_file_fig('model_comp_recovery', {
            'easiest_only': use_easiest_only
        }, ext=ext)
        plt.savefig(file, dpi=300)
        print('Saved to %s' % file)

    # --- Print mean +- SEM to CSV
    csv_file = locfile_out.get_file_csv('model_comp_recovery', {
        'easiest_only': use_easiest_only
    })
    d_csv = np2.dictlist2listdict({
        'data': ['original', 'simulated serial', 'simulated parallel'],
        'mean_dcost': m,
        'sem_dcost': e,
    })
    with open(csv_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=d_csv[0].keys())
        writer.writeheader()
        for d in d_csv:
            writer.writerow(d)
        print('Wrote to %s' % csv_file)

    # --- Mean NLL within each
    axs = plt2.GridAxes(
        1, 1, left=0.85, right=0.25, heights=[1.5],
        top=0.1, bottom=0.9,
    )
    plt.sca(axs[0, 0])
    plt.barh(np.arange(3), m, xerr=e, color='w', edgecolor='k')
    plt.yticks(np.arange(3), ['data',
                              'simulated\nserial',
                              'simulated\nparallel'])
    plt2.box_off(['top', 'right'])
    vmax = np.amax(np.abs(m) + np.abs(e))
    plt.xlim(np.array([-vmax, vmax]) * 1.1)
    plt2.detach_axis(xy='y', amin=0, amax=2)
    plt2.detach_axis(xy='x', amin=-vmax, amax=vmax)

    axvline_dcost()
    xticks_serial_vs_parallel(vmax, base)

    for ext in ['.pdf', '.png']:
        file = locfile_out.get_file_fig('model_comp_vs_recovery', {
                'easiest_only': use_easiest_only
        }, ext=ext)
        plt.savefig(file, dpi=300)
        print('Saved to %s' % file)

    print('--')


def load_comp(file: str) -> Dict[str, list]:
    d = loadmat(file)

    ds = {'subj': [], 'dcost': [], 'parad': [], 'bimanual': []}
    for parad, key, new_parad in [
        ('RT', 'yuls_exp', 'RT'),
        ('unimanual', 'annes_mono', 'unimanual'),
        ('bimanual', 'annes_bi', 'bimanual'),
    ]:
        ds['subj'] += consts.SUBJS[new_parad]

        dcost1 = list(d['delta_logl_predictions'][0, 0][key].reshape(-1))
        ds['dcost'] += dcost1
        ds['parad'] += [parad] * len(dcost1)
        ds['bimanual'] += [parad == 'bimanual'] * len(dcost1)

    ds['dcost'] = np.array(ds['dcost'])

    ds['subj_parad_bi'] = np.array(list(zip(
        ds['subj'],
        ['RT' if v in ['RT', 'eye'] else 'unibimanual' for v in ds['parad']],
        ds['bimanual'])), dtype=np.object)

    return ds


if __name__ == '__main__':
    main()