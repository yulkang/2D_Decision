"""
Make bar plots of model comparison & recovery from the descriptive model fits
("gamma models") to RT-style tasks.
"""

#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

import csv
import os
from pprint import pprint
from typing import Iterable, Tuple

import numpy as np
from matplotlib import pyplot as plt

from dtb.RT.dtb_2D_recover_RT_nonparam \
    import plot_bar_dloss_across_subjs, axvline_dcost, xticks_serial_vs_parallel
from data_2d import consts
from lib.pylabyk import localfile, np2, plt2

locfile_in = localfile.LocalFile(
    pth_root='../../data/Fit.D2.RT.Td2Tnd.Main',
    cache_dir=''
)

locfile_out = localfile.LocalFile(
    pth_root='../../data/Data_2D_Py/dtb/RTNonparamMATLAB',
    cache_dir=''
)

use_easiest_only = 0

files_model_comp = [
    ('eye', 'sbj=S1-S3+prd=RT+tsk=A+dtk=2+dmr=1+trm=1+eor=0+msf=1+ef=%d+ec=-%d+'
            'lf=0+eb=10+td={Ser,Par}+smr=NaN+um=2+us=2+comp=fval.csv'
            % (use_easiest_only, use_easiest_only)
     ),
    ('unimanual', 'sbj=S6-S13+prd=unimanual+tsk=A+dtk=2+dmr=1+trm=1+eor=0'
                  '+msf=1+ef=%d+ec=-%d+lf=0+eb=10+td={Ser,'
                  'Par}+smr=NaN+um=2+us=2'
                  '+comp=fval.csv'
                  % (use_easiest_only, use_easiest_only)
    ),
    ('bimanual', 'sbj=S6-S13+prd=bimanual+tsk=A+dtk=2+dmr=1+trm=1+eor=0+msf=1'
                 '+ef=%d+ec=-%d+lf=0+eb=10+td={Ser,Par}+smr=NaN+um=2+us=2'
                 '+comp=fval.csv'
                 % (use_easiest_only, use_easiest_only)
    )
]

files_model_recovery = [
    ('eye', 'sbj=S1_tdSer_seed1_ef%d-S3_tdPar_seed1_ef%d+prd=RT+tsk=A+dtk=2'
            '+dmr=1'
            '+trm=1+eor=0+msf=1+ef=%d+ec=-%d+lf=0+eb=10+td={Ser,Par}'
            '+smr=NaN+um=2+us=2+comp=fval.csv'
            % tuple([use_easiest_only] * 4)),
    ('unimanual', 'sbj=S6_tdSer_seed1_ef%d-S13_tdPar_seed1_ef%d+prd=unimanual'
                  '+tsk=A+dtk=2+dmr=1+trm=1+eor=0+msf=1+ef=%d+ec=-%d+lf=0+eb=10'
                  '+td={Ser,Par}+smr=NaN+um=2+us=2+comp=fval.csv'
                  % tuple([use_easiest_only] * 4)),
    ('bimanual', 'sbj=S6_tdSer_seed1_ef%d-S13_tdPar_seed1_ef%d+prd=bimanual'
                 '+tsk=A+dtk=2+dmr=1+trm=1+eor=0+msf=1+ef=%d+ec=-%d+lf=0+eb=10'
                 '+td={Ser,Par}+smr=NaN+um=2+us=2+comp=fval.csv'
                  % tuple([use_easiest_only] * 4)),
]


def main(base=10):
    ds = load_comp(files_model_comp)
    ds['dcost'] = np.array(ds['dcost']) / np.log(base)

    ds['subj'] += ['S7', 'S12']
    ds['dcost'] = np.concatenate([
        ds['dcost'],
        np.array([-7.77, -8.04])
    ], 0)
    ds['subj_parad_bi'] = np.concatenate([
        ds['subj_parad_bi'], np.array([
            ['S7', 'binary', False],
            ['S12', 'binary', False]
        ], dtype=np.object)
    ], 0)

    vmax = 20 * 2 # np.log(100)

    m = np.empty(3)
    e = np.empty(3)

    axs = plot_bar_dloss_across_subjs(
        dlosses=ds['dcost'], subj_parad_bis=ds['subj_parad_bi'],
        vmax=vmax, base=base
    )
    axs = plt2.GridAxes(
        nrows=1, ncols=3,
        heights=axs.heights, widths=[2],
        left=axs.left, right=axs.right, bottom=axs.bottom
    )
    plot_bar_dloss_across_subjs(
        dlosses=ds['dcost'], subj_parad_bis=ds['subj_parad_bi'],
        vmax=vmax, base=base,
        axs=axs[:, [0]]
    )
    plt.title('Data')
    print('--')

    m[0] = np.mean(ds['dcost'])
    e[0] = np2.sem(ds['dcost'])

    ds = load_comp(files_model_recovery)
    ds['dcost'] = np.array(ds['dcost']) / np.log(base)

    ds['subj'] += ['S7', 'S7', 'S12', 'S12']
    ds['dcost'] = np.concatenate([
        ds['dcost'],
        np.array([-47.06, 4.86, -11.29, 3.73])
    ], axis=0)
    ds['subj_parad_bi'] = np.concatenate([
        ds['subj_parad_bi'],
        np.array([
            ['S7', 'binary', False],
            ['S7', 'binary', False],
            ['S12', 'binary', False],
            ['S12', 'binary', False]
        ], dtype=np.object)
    ], axis=0)
    ds['is_ser_sim'] = np.concatenate([
        ds['is_ser_sim'],
        np.array([True, False, True, False]),
    ], axis=0)

    pprint(ds)

    titles = ['Simulated\nSerial', 'Simulated\nParallel']
    for i in [0, 1]:
        ds1 = np2.filt_dict(ds, ds['is_ser_sim'] == 1 - i)
        plot_bar_dloss_across_subjs(
            dlosses=ds1['dcost'], subj_parad_bis=ds1['subj_parad_bi'],
            vmax=vmax, axs=axs[:, [i + 1]], base=base)
        plt.title(titles[i])

        m[i + 1] = np.mean(ds1['dcost'])
        e[i + 1] = np2.sem(ds1['dcost'])

        plt.sca(axs[0, i + 1])
        plt2.box_off(['left'])
        plt.yticks([])

    from lib.pylabyk.cacheutil import mkdir4file
    for ext in ['.pdf', '.png']:
        file = locfile_out.get_file_fig('model_comp_recovery', {
            'easiest_only': use_easiest_only
        }, ext=ext)
        mkdir4file(file)
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


def load_comp(files: Iterable[Tuple[str, str]]):
    ds = []
    for parad, fname in files:
        file = os.path.join(locfile_in.pth_root, fname)
        with open(file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row['parad'] = parad
                ds.append(row)
    ds = np2.listdict2dictlist(ds)

    if ds['subj'][0].find('_td') != -1:
        ds['is_ser_sim'] = np.array([v.find('tdSer') != -1 for v in ds['subj']])
        ds['subj'] = [v[:v.find('_')] for v in ds['subj']]

    ds['subj'] = [consts.subj_parad_old2new(subj, parad)
                  for subj, parad in zip(ds['subj'], ds['parad'])]

    for k in ['dcost', 'Ser', 'Par', 'BayesFactor']:
        ds[k] = [float(v) for v in ds[k]]

    # Here, dcost is NLL_Par - NLL_Ser
    ds['dcost'] = np.array([-v for v in ds['dcost']])
    ds['bimanual'] = [v == 'bimanual' for v in ds['parad']]
    ds['parad'] = ['RT' if v in ['RT', 'eye']
                   else 'unibimanual' for v in ds['parad']]

    ds['subj_parad_bi'] = np.array(list(zip(
        ds['subj'],
        ds['parad'],
        ds['bimanual']
    )), dtype=np.object)
    return ds


if __name__ == '__main__':
    main()