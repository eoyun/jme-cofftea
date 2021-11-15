#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from bucoffea.plot.trigger import get_xy, lumi_by_region, trgname
from bucoffea.plot.style import markers, matplotlib_rc
from matplotlib import pyplot as plt
from pprint import pprint

pjoin = os.path.join

matplotlib_rc()

def get_txt_files(inputdir, year):
    
    txtfilematch = lambda f: re.match(f'table_1m_recoil_SingleMuon_{year}.*txt', f)
    return [pjoin(inputdir, f) for f in os.listdir(inputdir) if txtfilematch(f)]

def main():
    inputdir = sys.argv[1]

    outdir = pjoin(inputdir, 'overlayed')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    jettags = {
        'two_central_jets' : 'Central-Central',
        'one_jet_forward_one_jet_central' : 'Central-Forward',
        'two_hf_jets' : 'HF-HF',
    }

    for year in [2017, 2018]:
        xs = {}
        ys = {}
        yerrs = {}
        files = get_txt_files(inputdir, year)
        for f in files:
            for tag, label in jettags.items():
                if re.match(f'.*{tag}.*', f):
                    xs[label], _, ys[label], yerrs[label] = get_xy(f)

        fig, ax = plt.subplots()
        opts = markers('data')
        emarker = opts.pop('emarker', '')
        for idx,label in enumerate(jettags.values()):
            opts['color'] = f'C{idx}'
            ax.errorbar(xs[label], ys[label], yerr=yerrs[label], label=label, **opts)
        
        ax.legend(title='Two Leading Jets', loc='center right')
        ax.axvline(250, ymin=0, ymax=1, color='k', linewidth=2)
        ax.set_xlabel('Recoil (GeV)')
        ax.set_ylabel('Efficiency')

        ax.set_xlim(0,1000)
        ax.set_ylim(0,1.1)

        ax.text(0,1,f'{year} Data',
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        ax.text(1., 1., r"%.1f fb$^{-1}$ (13 TeV)" % lumi_by_region('1m',year),
                fontsize=14,
                ha='right',
                va='bottom',
                transform=ax.transAxes
                )

        ax.text(1., 0., f'{trgname(year, "120pfht_recoil")}',
                fontsize=10,
                ha='right',
                va='bottom',
                transform=ax.transAxes
                )

        outpath = pjoin(outdir, f'data_eff_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

if __name__ == '__main__':
    main()