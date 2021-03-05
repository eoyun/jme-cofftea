#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea import hist
from scipy.stats import distributions
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def setup(acc, distribution, outtag):
    '''Prepare the histogram to be plotted, common operation to each of the functions below.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)    

    return h, outdir

def plot_split_by_eta(acc, outtag, distribution, dataset_regex='DY.*2017'):
    h, outdir = setup(acc, distribution, outtag)

    # Z(mumu) region without HF-HF veto
    h = h.integrate('region', 'cr_2m_vbf_no_noisecuts').integrate('dataset', re.compile(dataset_regex))

    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax, overlay='jeta')

    ax.set_yscale('log')
    ax.set_ylim(1e-3,1e3)

    if dataset_regex == 'DY.*2017':
        outnametag = 'mc'
        plottag = 'DYJetsToLL 2017'
    elif dataset_regex == 'MET.*2017':
        outnametag = 'data'
        plottag = 'MET 2017'

    ax.text(0., 1., plottag,
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

    ax.text(1., 1., r'$Z(\mu\mu)$',
            fontsize=14,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

    outpath = pjoin(outdir, f'low_eta_vs_high_eta_{distribution}_{outnametag}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def plot_zmumu_data_vs_mc(acc, outtag, distribution):
    h, outdir = setup(acc, distribution, outtag)

    # Z(mumu) region without HF-HF veto
    h = h.integrate('region', 'cr_2m_vbf_no_noisecuts')

    # Integrate over all |eta| > 3.0 for now
    h = h.integrate('jeta')


    for year in [2017]:
        _h = h[re.compile(f'.*{year}')]

        fig, ax = plt.subplots()
        hist.plot1d(_h, ax=ax, overlay='dataset')

        ax.set_yscale('log')
        ax.set_ylim(1e-3,1e3)

        ax.text(0., 1., r'HF jets with $p_T > 80 \ GeV$',
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        ax.text(1., 1., r'$Z(\mu\mu)$',
            fontsize=14,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

        # Save figure
        outpath = pjoin(outdir, f'data_vs_mc_{distribution}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/','')

    distributions = [
        'ak4_sigma_eta_eta',
        'ak4_sigma_phi_phi',
        'ak4_etastripsize',
    ]

    for distribution in distributions:
        plot_zmumu_data_vs_mc(acc, outtag, distribution)

        plot_split_by_eta(acc, outtag, distribution, dataset_regex='DY.*2017')
        plot_split_by_eta(acc, outtag, distribution, dataset_regex='MET.*2017')

if __name__ == '__main__':
    main()