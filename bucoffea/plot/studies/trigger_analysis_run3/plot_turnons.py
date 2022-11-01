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

def sigmoid(x,a,b):
    return 1 / (1 + np.exp(-a * (x-b)) )

def do_sigmoid_fit(h_num, h_den, fit_func, p0):
    """
    Given the num and denom histogram objects, do the sigmoid fit.
    Return the array of fit parameters.
    """
    ratio = h_num.values()[()] / h_den.values()[()]
    x = h_num.axes()[0].centers()

    valid = ~(np.isnan(ratio) | np.isinf(ratio))

    popt, pcov = scipy.optimize.curve_fit(fit_func, x[valid], ratio[valid], p0=p0)
    return popt


def plot_turnons_for_different_runs(acc, outdir, fit_init, region='tr_metnomu'):
    """Plot METNoMu trigger turn on for different set of runs."""
    distribution = DISTRIBUTIONS[region]
    acc.load(distribution)
    h = acc[distribution]

    if distribution in NEW_BINS:
        new_ax = NEW_BINS[distribution]
        h = h.rebin(new_ax.name, new_ax)

    h_num = h.integrate('region', f'{region}_num')
    h_den = h.integrate('region', f'{region}_den')

    fig, ax = plt.subplots()

    # Dataset regex -> Legend label to plot
    datasets_labels = {
        'Muon.*2022[CD]' : '2022C+D',
        'Muon.*2022E'    : '2022E',
        'Muon.*2022F'    : '2022F',
    }

    for index, (regex, label) in enumerate(datasets_labels.items()):
        num = h_num.integrate('dataset', re.compile(regex))
        den = h_den.integrate('dataset', re.compile(regex))
        error_opts['color'] = f'C{index}'
        
        # Do the sigmoid fit
        popt = do_sigmoid_fit(num, den, sigmoid, p0=fit_init)

        # Plot the fit result
        centers = num.axes()[0].centers()
        x = np.linspace(min(centers), max(centers), 200)
        ax.plot(x,
            sigmoid(x, *popt), 
            color=f'C{index}',
        )

        # Plot the individual ratio of histograms
        hist.plotratio(
            num,
            den,
            ax=ax,
            label=f'{label}, $\\mu={popt[1]:.2f}$, $\\sigma={1/popt[0]:.2f}$',
            error_opts=error_opts,
            clear=False
        )


    ax.legend(title='Run', prop={'size' : 8})
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

    ax.text(0.98,0.3,r'$Eff(x) = \frac{1}{1 + e^{-(x - \mu) / \sigma}}$',
        fontsize=12,
        ha='right',
        va='bottom',
        transform=ax.transAxes
    )

    ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')

    ax.set_ylim(bottom=0)

    outpath = pjoin(outdir, f'turnons_{region}.pdf')
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
    
    regions_fit_guesses = {
        'tr_jet' : (0.02, 500),
        'tr_ht' : (0.04, 1050),
        'tr_metnomu' : (0.05, 200),
        'tr_metnomu_filterhf' : (0.05, 200),
    }

    for region, fit_init in tqdm(regions_fit_guesses.items(), desc='Plotting turn-ons'):
        plot_turnons_for_different_runs(acc, 
            outdir, 
            fit_init=fit_init, 
            region=region
        )

    # Eta-separated plots for leading jet eta (PFJet500)
    # plot_turnons_by_eta(acc, outdir, region='tr_jet')

    # plot_eta_efficiency(acc, outdir, region='tr_jet')

if __name__ == '__main__':
    main()