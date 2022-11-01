#!/usr/bin/env python3

import os
import sys
import re
import scipy
import numpy as np

from matplotlib import pyplot as plt
from coffea import hist
from klepto.archives import dir_archive
from tqdm import tqdm

pjoin = os.path.join

Bin = hist.Bin

error_opts = {
    'linestyle':'none',
    'marker': '.',
    'markersize': 10.,
    'elinewidth': 1,
}

NEW_BINS = {
    'recoil' : Bin("recoil", "Recoil (GeV)", list(range(0,500,20)) + list(range(500,1000,40))),
    'ak4_pt0' : Bin("jetpt", r"Leading Jet $p_{T}$ (GeV)", list(range(0,500,20)) + list(range(500,1000,40))),
    'ak4_eta0' : Bin("jeteta", r"Leading Jet $\eta$", 25, -5, 5),
    'ak4_abseta0_pt0' : Bin("jetpt", r"Leading Jet $p_{T}$ (GeV)", list(range(0,500,20)) + list(range(500,1000,40))),
    'ht' : Bin("ht", r"$H_{T}$ (GeV)", list(range(0,2000,80)) + list(range(2000,4000,160))),
}

TRIGGER_NAMES = {
    'tr_metnomu' : 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
    'tr_metnomu_filterhf' : 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF',
    'tr_jet' : 'HLT_PFJet500',
    'tr_ht' : 'HLT_PFHT1050',
}

DISTRIBUTIONS = {
    'tr_metnomu' : 'recoil',
    'tr_metnomu_filterhf' : 'recoil',
    'tr_jet' : 'ak4_pt0',
    'tr_ht' : 'ht',
}

def err_func(x, mean, width):
    """Error function to use for fitting."""
    return scipy.special.erf((x-mean)*width)


def plot_turnons_CD_vs_E(acc, outdir, region='tr_metnomu', plotF=True):
    """Plot METNoMu trigger turn on for Runs C+D vs E."""
    distribution = DISTRIBUTIONS[region]
    acc.load(distribution)
    h = acc[distribution]

    if distribution in NEW_BINS:
        new_ax = NEW_BINS[distribution]
        h = h.rebin(new_ax.name, new_ax)

    h_num = h.integrate('region', f'{region}_num')
    h_den = h.integrate('region', f'{region}_den')

    fig, ax = plt.subplots()

    hist.plotratio(
        h_num.integrate('dataset', re.compile('Muon.*2022[CD]')),
        h_den.integrate('dataset', re.compile('Muon.*2022[CD]')),
        ax=ax,
        label='2022C+D',
        error_opts=error_opts,
        clear=False
    )
    hist.plotratio(
        h_num.integrate('dataset', re.compile('Muon.*2022E')),
        h_den.integrate('dataset', re.compile('Muon.*2022E')),
        ax=ax,
        label='2022E',
        error_opts=error_opts,
        clear=False
    )
    if plotF:
        hist.plotratio(
            h_num.integrate('dataset', re.compile('Muon.*2022F')),
            h_den.integrate('dataset', re.compile('Muon.*2022F')),
            ax=ax,
            label='2022F',
            error_opts=error_opts,
            clear=False
        )

    # Fit error function to individual curves
    h_num_cd = h_num.integrate('dataset', re.compile('Muon.*2022[CD]'))
    h_den_cd = h_den.integrate('dataset', re.compile('Muon.*2022[CD]'))

    ratio_cd = h_num_cd.values()[()] / h_den_cd.values()[()]
    centers = h_num_cd.axes()[0].centers()  

    popt, pcov = scipy.curve_fit(err_func, centers, ratio_cd)
    print(popt)

    ax.plot(centers, err_func(centers, *popt), label='Fit C+D')

    ax.legend(title='Run')
    ax.set_ylabel('Trigger Efficiency')

    ax.text(0,1,'Muon 2022',
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes
    )
    
    ax.text(1,1,TRIGGER_NAMES[region],
        fontsize=10,
        ha='right',
        va='bottom',
        transform=ax.transAxes
    )

    ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')

    ax.set_ylim(bottom=0)

    outpath = pjoin(outdir, f'turnons_CD_vs_E_{region}.pdf')
    fig.savefig(outpath)
    plt.close(fig)


def plot_turnons_by_eta(acc, outdir, region, plotF=True):
    """
    Plot the turn-ons for runs 2022C+D vs run 2022E, split by 
    different leading jet eta slices.
    """
    distribution = 'ak4_abseta0_pt0'
    acc.load(distribution)
    h = acc[distribution]

    if distribution in NEW_BINS:
        new_ax = NEW_BINS[distribution]
        h = h.rebin(new_ax.name, new_ax)

    # Different eta slices
    etaslices = {
        'central' : slice(0, 1.3),
        'endcap' : slice(1.3, 2.5),
        'forward' : slice(2.5, 5.0),
    }

    for label, etaslice in etaslices.items():
        # Integrate the eta slice for the leading jet
        histo = h.integrate('jeteta', etaslice)

        h_num = histo.integrate('region', f'{region}_num')
        h_den = histo.integrate('region', f'{region}_den')

        fig, ax = plt.subplots()

        hist.plotratio(
            h_num.integrate('dataset', re.compile('Muon.*2022[CD]')),
            h_den.integrate('dataset', re.compile('Muon.*2022[CD]')),
            ax=ax,
            label='2022C+D',
            error_opts=error_opts,
            clear=False
        )
        hist.plotratio(
            h_num.integrate('dataset', re.compile('Muon.*2022E')),
            h_den.integrate('dataset', re.compile('Muon.*2022E')),
            ax=ax,
            label='2022E',
            error_opts=error_opts,
            clear=False
        )
        if plotF:
            hist.plotratio(
                h_num.integrate('dataset', re.compile('Muon.*2022F')),
                h_den.integrate('dataset', re.compile('Muon.*2022F')),
                ax=ax,
                label='2022F',
                error_opts=error_opts,
                clear=False
            )

        ax.legend(title='Run')
        ax.set_ylabel('Trigger Efficiency')

        ax.text(0,1,'Muon 2022',
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )
        
        rlabel = f'{TRIGGER_NAMES[region]}, {etaslice.start:.1f} < $|\\eta|$ < {etaslice.stop:.1f}'

        ax.text(1,1,rlabel,
            fontsize=10,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

        ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')

        ax.set_ylim(bottom=0)

        outpath = pjoin(outdir, f'turnons_CD_vs_E_{region}_eta_{label}.pdf')
        fig.savefig(outpath)
        plt.close(fig)


def plot_eta_efficiency(acc, outdir, region, plotF=True):
    """Plot the efficiency of the jet500 trigger as a function of leading jet eta."""
    distribution = 'ak4_eta0'
    acc.load(distribution)
    h = acc[distribution]

    if distribution in NEW_BINS:
        new_ax = NEW_BINS[distribution]
        h = h.rebin(new_ax.name, new_ax)

    h_num = h.integrate('region', f'{region}_num')
    h_den = h.integrate('region', f'{region}_den')

    fig, ax = plt.subplots()

    hist.plotratio(
        h_num.integrate('dataset', re.compile('Muon.*2022[CD]')),
        h_den.integrate('dataset', re.compile('Muon.*2022[CD]')),
        ax=ax,
        label='2022C+D',
        error_opts=error_opts,
        clear=False
    )
    hist.plotratio(
        h_num.integrate('dataset', re.compile('Muon.*2022E')),
        h_den.integrate('dataset', re.compile('Muon.*2022E')),
        ax=ax,
        label='2022E',
        error_opts=error_opts,
        clear=False
    )
    if plotF:
        hist.plotratio(
            h_num.integrate('dataset', re.compile('Muon.*2022F')),
            h_den.integrate('dataset', re.compile('Muon.*2022F')),
            ax=ax,
            label='2022F',
            error_opts=error_opts,
            clear=False
        )

    ax.legend(title='Run')
    ax.set_ylabel('Trigger Efficiency')

    ax.set_ylim(bottom=0)

    ax.text(0,1,'Muon 2022',
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
    )

    ax.text(1,1,'HLT_PFJet500',
            fontsize=10,
            ha='right',
            va='bottom',
            transform=ax.transAxes
    )

    outpath = pjoin(outdir, f'turnons_CD_vs_E_{region}_eta.pdf')
    fig.savefig(outpath)
    plt.close(fig)


def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)

    outtag = os.path.basename(inpath.rstrip('/'))
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    regions = [
        'tr_metnomu',
        'tr_metnomu_filterhf',
        'tr_jet',
        'tr_ht',
    ]

    for region in regions:
        plot_turnons_CD_vs_E(acc, outdir, region=region)

    # Eta-separated plots for leading jet eta (PFJet500)
    plot_turnons_by_eta(acc, outdir, region='tr_jet')

    plot_eta_efficiency(acc, outdir, region='tr_jet')

if __name__ == '__main__':
    main()