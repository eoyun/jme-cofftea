#!/usr/bin/env python3

import os
import sys
import re
import scipy
import argparse
import warnings
import numpy as np
import mplhep as hep

from matplotlib import pyplot as plt
from coffea import hist
from klepto.archives import dir_archive
from tqdm import tqdm
from tabulate import tabulate

# Ignore division warnings from Coffea + Numpy
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Use default CMS styling
plt.style.use(hep.style.CMS)

pjoin = os.path.join

Bin = hist.Bin

error_opts = {
    'linestyle':'none',
    'marker': '.',
    'markersize': 14.,
    'elinewidth': 1,
}

NEW_BINS = {
    'met' : Bin("met", r"Offline $p_T^{miss}$ [GeV]", list(range(0,500,20)) + list(range(500,1000,40))),
    'recoil' : Bin("recoil", r"Offline $p_{T,no-\mu}^{miss}$ [GeV]", list(range(0,500,20)) + list(range(500,1000,40))),
    'ak4_pt0' : Bin("jetpt", r"Offline leading jet $p_{T}$ [GeV]", list(range(0,500,20)) + list(range(500,1000,20))),
    'ak4_eta0' : Bin("jeteta", r"Offline leading jet $\eta$", 25, -5, 5),
    'ak4_abseta0_pt0' : Bin("jetpt", r"Offline leading jet $p_{T}$ [GeV]", list(range(0,500,20)) + list(range(500,1000,40))),
    'ht' : Bin("ht", r"Offline $H_{T}$ [GeV]", list(range(0,2000,80)) + list(range(2000,3200,200))),
    # 'ht' : Bin("ht", r"Offline $H_{T}$ [GeV]", list(range(0,2000,80)) + list(range(2000,4000,160))),
}

TRIGGER_NAMES = {
    'tr_met' : 'HLT_PFMET120_PFMHT120_IDTight',
    'tr_metnomu' : 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
    'tr_metnomu_L1ETMHF100' : 'HLT_PFMETNoMu120 (+ L1_ETMHF100)',
    'tr_metnomu_filterhf' : 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF',
    'tr_jet' : 'HLT_PFJet500',
    'tr_ht' : 'HLT_PFHT1050',
}

TRIGGER_LABELS = {
    'tr_jet'     : 'AK4PF jet with \n $p_T > 500 \ GeV$',
    'tr_ht'      : '$H_T > 1050 \ GeV$',
    'tr_metnomu' : '$p_{T,no-\mu}^{miss} > 120 \ GeV$ \n$H_{T,no-\mu}^{miss} > 120 \ GeV$',
    'tr_metnomu_filterhf' : '$p_{T,no-\mu}^{miss} > 120 \ GeV$ \n$H_{T,no-\mu}^{miss} > 120 \ GeV$',
}

DISTRIBUTIONS = {
    'tr_met' : 'recoil',
    'tr_metnomu' : 'recoil',
    'tr_metnomu_filterhf' : 'recoil',
    'tr_jet' : 'ak4_pt0',
    'tr_ht' : 'ht',
}


def parse_cli():
    """Command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to input merged coffea accumulator.')
    parser.add_argument('-f', '--fit-func', help='Fit function to use.', default='erf', choices=['sigmoid', 'erf'])
    parser.add_argument('-r', '--region', default='.*', help='Regex specifying the regions to look for turn-ons.')
    args = parser.parse_args()
    return args


def sigmoid(x, a, b):
    """Sigmoid function to use for turn-on fits."""
    return 1 / (1 + np.exp( - (x - a) / b) )


def error_func(x, a, b):
    """Returns an error function where the range is adjusted to [0,1]."""
    return 0.5 * (1 + scipy.special.erf((x - a) / b))


def compute_chi2(h_num, h_den, fit_func, *popt):
    """
    Given a fit to the ratio of h_num / h_den using fit_func,
    this function computes the chi2 of the fit and returns it.
    """
    xdata = h_num.axes()[0].centers()
    
    # Compute the ratio and the error on ratio
    sumw_num = h_num.values()[()]
    sumw_den = h_den.values()[()]
    ydata = sumw_num / sumw_den

    yerr = np.abs(
        hist.clopper_pearson_interval(sumw_num, sumw_den) - ydata
    )
    
    # Symmetrize
    yerr = (yerr[0] + yerr[1]) / 2

    # Compute the absolute difference
    r = ydata - fit_func(xdata, *popt)
    r /= yerr
    r[np.isinf(r) | np.isnan(r)] = 0.
    return np.sum(r**2) / (len(r)-1)


def fit_turnon(h_num, h_den, fit_func, p0):
    """
    Given the num and denom histogram objects, do the sigmoid fit.
    Return the array of fit parameters.
    """
    ratio = h_num.values()[()] / h_den.values()[()]
    x = h_num.axes()[0].centers()

    valid = ~(np.isnan(ratio) | np.isinf(ratio))

    popt, pcov = scipy.optimize.curve_fit(fit_func, x[valid], ratio[valid], p0=p0)
    return popt, pcov


def plot_turnons_for_different_runs(
    acc, 
    outdir, 
    fit_init, 
    fit_func, 
    region='tr_metnomu',
    plot_fit=True,
    ):
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
        "Muon.*2022[CDE]" : "pre-HCAL update",
        "Muon.*2022[FG]"  : "post-HCAL update",
    }

    chi2_vals = {
        "Dataset"     : [],
        "Chi2 / dof"  : [],
    }

    for index, (regex, label) in enumerate(datasets_labels.items()):
        num = h_num.integrate('dataset', re.compile(regex))
        den = h_den.integrate('dataset', re.compile(regex))
        
        err_opts_copy = error_opts.copy()
        err_opts_copy['color'] = f'C{index}'
        
        # Fit the turn-on curve with the given fit function
        popt, pcov = fit_turnon(num, den, fit_func, p0=fit_init)

        # Plot the fit result
        centers = num.axes()[0].centers()

        # Compute the chi2 for this fit
        chi2 = compute_chi2(num, den, fit_func, *popt)

        chi2_vals["Dataset"].append(label)
        chi2_vals["Chi2 / dof"].append(chi2)

        x = np.linspace(min(centers), max(centers), 200)
        
        # Plot the fitted erf function if we want to
        if plot_fit:
            ax.plot(x,
                fit_func(x, *popt), 
                color=f'C{index}',
            )

            legend_label = f'{label}, $\\mu={popt[0]:.2f}$, $\\sigma={popt[1]:.2f}$'
        
        else:
            legend_label = label

        # Plot the individual ratio of histograms
        hist.plotratio(
            num,
            den,
            ax=ax,
            label=legend_label,
            error_opts=err_opts_copy,
            clear=False
        )

    ax.set_xlabel(NEW_BINS[distribution].label, horizontalalignment='right', x=1)
    ax.set_ylabel('Efficiency', verticalalignment='bottom', y=0.9)
    ax.legend()
    ax.grid(True, which='major')

    ax.text(0.8, 0.05, TRIGGER_LABELS[region],
        fontsize=24,
        ha='center',
        va='bottom',
        transform=ax.transAxes,
    )

    if "recoil" in distribution:
        ax.set_xscale("log")
        ax.set_xlim(1e1,1e3)

    # ax.text(0,1,'Muon 2022',
    #     fontsize=14,
    #     ha='left',
    #     va='bottom',
    #     transform=ax.transAxes
    # )
    
    # ax.text(1,1,TRIGGER_NAMES[region],
    #     fontsize=10,
    #     ha='right',
    #     va='bottom',
    #     transform=ax.transAxes
    # )

    hep.cms.label(year="2022", paper=True, llabel=" Preliminary", rlabel=r"$34.3 \ fb^{-1}$, 2022 (13.6 TeV)")
    hep.cms.text()

    # Setup the equation label and the output directory to save
    if fit_func == sigmoid:
        eqlabel = r'$Eff(x) = \frac{1}{1 + e^{-(x - \mu) / \sigma}}$' 
        fontsize = 12
        outdir = pjoin(outdir, 'sigmoid_fit')
    elif fit_func == error_func:
        eqlabel = r'$Eff(x) = 0.5 * (1 + erf(x - \mu) / \sigma)$'
        fontsize = 10
        outdir = pjoin(outdir, 'erf_fit')
    else:
        raise RuntimeError('An unknown fit function is specified.')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Equation label (if we're plotting the fitted function)
    if plot_fit:
        ax.text(0.98,0.3,eqlabel,
            fontsize=fontsize,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

    # Print out chi2 information for the fits
    print(tabulate(chi2_vals, headers='keys', floatfmt=".2f"))

    ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')

    ax.set_ylim((0,1.5))

    for fformat in ['png', 'pdf']:
        outpath = pjoin(outdir, f'turnons_{region}.{fformat}')
        fig.savefig(outpath)

    plt.close(fig)


def plot_turnons_by_eta(acc, outdir, region, datasets):
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

        for dataset in datasets:
            hist.plotratio(
                h_num.integrate('dataset', re.compile(dataset["regex"])),
                h_den.integrate('dataset', re.compile(dataset["regex"])),
                ax=ax,
                label=dataset["label"],
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


def plot_eta_efficiency(acc, outdir, region, datasets):
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

    for dataset in datasets:
        hist.plotratio(
            h_num.integrate('dataset', re.compile(dataset["regex"])),
            h_den.integrate('dataset', re.compile(dataset["regex"])),
            ax=ax,
            label=dataset["label"],
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

def plot_turnons_with_without_water_leak(acc, outdir, dataset='Muon.*2022E'):
    """
    Plot Jet500 trigger turn-on for two cases:
    1. Leading jet is NOT in the water leak region
    2. Leading jet is in the water leak region
    """
    distribution = 'ak4_pt0'
    acc.load(distribution)
    h = acc[distribution]

    if distribution in NEW_BINS:
        new_ax = NEW_BINS[distribution]
        h = h.rebin(new_ax.name, new_ax)

    h = h.integrate('dataset', re.compile(dataset))

    histos = {}
    histos['water_leak'] = {
        'num' : h.integrate('region', 'tr_jet_water_leak_num'),
        'den' : h.integrate('region', 'tr_jet_water_leak_den'),
    }
    histos['no_water_leak'] = {
        'num' : h.integrate('region', 'tr_jet_water_leak_veto_num'),
        'den' : h.integrate('region', 'tr_jet_water_leak_veto_den'),
    }

    # Plot the two turn-ons side by side
    fig, ax = plt.subplots()
    for label, histograms in histos.items():
        hist.plotratio(
            histograms['num'],
            histograms['den'],
            ax=ax,
            error_opts=error_opts,
            label=label,
            clear=False
        )

    ax.legend()
    ax.set_ylabel('Trigger Efficiency')

    ax.text(0,1,'Muon 2022E',
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

    ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')
    ax.set_ylim(bottom=0)

    outpath = pjoin(outdir, 'turnons_water_leak_Muon2022E.pdf')
    fig.savefig(outpath)
    plt.close(fig)


def plot_l1_vs_hlt_HT1050(acc, outdir, dataset='Muon.*2022E.*'):
    """Plot the L1 vs HLT turn-ons for HT1050 trigger."""
    distribution = 'ht'
    acc.load(distribution)
    h = acc[distribution]

    if distribution in NEW_BINS:
        new_ax = NEW_BINS[distribution]
        h = h.rebin(new_ax.name, new_ax)

    h = h.integrate('dataset', re.compile(dataset))

    # Get the histograms for L1 and HLT turn-ons
    histos = {}
    histos['hlt_ht1050'] = {
        'num' : h.integrate('region', 'tr_ht_num'),
        'den' : h.integrate('region', 'tr_ht_den'),
    }

    histos['l1_ht1050'] = {
        'num' : h.integrate('region', 'tr_l1_ht_num'),
        'den' : h.integrate('region', 'tr_l1_ht_den'),
    }

    fig, ax = plt.subplots()

    for triglabel, histograms in histos.items():
        hist.plotratio(
            histograms['num'],
            histograms['den'],
            ax=ax,
            label=triglabel.upper(),
            error_opts=error_opts,
            clear=False,
        )

    ax.legend()
    ax.set_ylabel('Trigger Efficiency')
    ax.set_ylim(bottom=0)

    ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')

    ax.text(0,1,'Muon 2022E',
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes
    )

    outpath = pjoin(outdir, 'l1_vs_hlt_HT1050.pdf')
    fig.savefig(outpath)
    plt.close(fig)


def plot_efficiency_vs_nvtx(acc, 
    outdir, 
    region,
    distribution='recoil_npvgood',
    dataset='Muon.*2022[FG].*'
    ):
    """
    Plots the efficiency vs number of vertices.
    """
    acc.load(distribution)
    h = acc[distribution]

    h = h.integrate('dataset', re.compile(dataset))
    
    # Rebin Nvtx axis
    new_bin_edges = [-0.5,9.5,14.5] + list(np.arange(15.5,50.5,1)) + list(np.arange(50.5,56.5,2)) + [56.5,60.5,64.5,69.5,79.5]
    new_ax = hist.Bin('nvtx','Number of vertices',new_bin_edges)
    h = h.rebin("nvtx", new_ax)

    # List of recoil slices to plot (i.e. recoil > X)   
    recoil_slices = [
        slice(200,None),
        slice(250,None),
        slice(300,None),
    ]

    fig, ax = plt.subplots()

    for recoil_slice in tqdm(recoil_slices, desc="Plotting eff vs nvtx"):
        # Integrate out the recoil slice
        if distribution == 'recoil_npvgood':
            histo = h.integrate('recoil', recoil_slice)
        elif distribution == 'met_npvgood':
            histo = h.integrate('met', recoil_slice)
        else:
            raise RuntimeError(f'Unrecognized distribution: {distribution}')

        # Get the num and denom histograms and plot!
        hist.plotratio(
            histo.integrate('region', f'{region}_num'),
            histo.integrate('region', f'{region}_den'),
            ax=ax,
            error_opts=error_opts,
            label=f'Offline $p_{{T,no-\\mu}}^{{miss}} > {recoil_slice.start:.0f} \\ GeV$',
            clear=False,
        )

    ax.legend()
    ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')
    ax.set_ylim(0.5,1.3)

    ax.set_xlabel(new_ax.label, horizontalalignment='right', x=1)
    ax.set_ylabel('Efficiency', verticalalignment='bottom', y=0.9)
    ax.grid(True, which='major')

    ax.text(0.8, 0.05, TRIGGER_LABELS[region],
        fontsize=24,
        ha='center',
        va='bottom',
        transform=ax.transAxes,
    )

    # CMS label & text
    hep.cms.label(year="2022", paper=True, llabel=" Preliminary", rlabel=r"$34.3 \ fb^{-1}$, 2022 (13.6 TeV)")
    hep.cms.text()


    # ax.text(0,1,'Muon 2022 F+G',
    #     fontsize=14,
    #     ha='left',
    #     va='bottom',
    #     transform=ax.transAxes
    # )

    # ax.text(1,1,TRIGGER_NAMES[region],
    #     fontsize=10,
    #     ha='right',
    #     va='bottom',
    #     transform=ax.transAxes
    # )

    # Save the plot
    for fformat in ['pdf', 'png']:
        filename = f'{region}_eff_vs_nvtx.{fformat}'
        outpath = pjoin(outdir, filename)
        fig.savefig(outpath)
    plt.close(fig)


def plot_turnon_wrt_nvtx(acc, outdir, region, distribution, dataset='Muon.*2022[FG].*'):
    """
    Plot METNoMu turn-on on different Nvtx bins.
    """
    acc.load(distribution)
    h = acc[distribution]

    # Rebinning
    if distribution == 'recoil_npvgood':
        new_ax = NEW_BINS['recoil']
    elif distribution == 'met_npvgood':
        new_ax = NEW_BINS['met']
    else:
        raise RuntimeError(f'Unrecognized distribution: {distribution}')

    h = h.rebin(new_ax.name, new_ax)

    h = h.integrate('dataset', re.compile(dataset))

    # Different Nvtx bins
    nvtx_bins = [
        slice(10, 20),
        slice(20, 40),
        slice(40, 60),
    ]

    fig, ax = plt.subplots()

    for nvtx_bin in nvtx_bins:
        histo = h.integrate('nvtx', nvtx_bin)

        hist.plotratio(
            histo.integrate('region', f'{region}_num'),
            histo.integrate('region', f'{region}_den'),
            ax=ax,
            label=f'{nvtx_bin.start:.0f} < $N_{{vtx}}$ < {nvtx_bin.stop:.0f}',
            error_opts=error_opts,
            clear=False,
        )

    ax.legend()
    ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')
    ax.set_ylim(bottom=0)
    
    ax.set_xscale("log")
    ax.set_xlim(1e1,1e3)
    ax.set_ylim(0,1.5)

    ax.set_xlabel(new_ax.label, horizontalalignment='right', x=1)
    ax.set_ylabel('Efficiency', verticalalignment='bottom', y=0.9)
    
    ax.grid(True, which='major')

    ax.text(0.8, 0.05, TRIGGER_LABELS[region],
        fontsize=24,
        ha='center',
        va='bottom',
        transform=ax.transAxes,
    )

    # CMS text & labels
    hep.cms.label(year="2022", paper=True, llabel=" Preliminary", rlabel=r"$34.3 \ fb^{-1}$, 2022 (13.6 TeV)")
    hep.cms.text()

    # ax.text(0,1,'Muon 2022 F+G',
    #     fontsize=14,
    #     ha='left',
    #     va='bottom',
    #     transform=ax.transAxes
    # )
    
    # ax.text(1,1,TRIGGER_NAMES[region],
    #     fontsize=10,
    #     ha='right',
    #     va='bottom',
    #     transform=ax.transAxes
    # )

    for fformat in ['pdf', 'png']:
        outpath = pjoin(outdir, f'{region}_turnon_vs_nvtx.{fformat}')
        fig.savefig(outpath)
    plt.close(fig)


def compare_turnons(acc, outdir, regions, dataset, distribution='recoil'):
    """
    Compare the turn-ons in the two regions.
    """
    acc.load(distribution)
    h = acc[distribution]

    # Rebin
    if distribution in NEW_BINS:
        new_ax = NEW_BINS[distribution]
        h = h.rebin(new_ax.name, new_ax)
    
    h = h.integrate('dataset', re.compile(dataset))

    # Retrieve num and denom histograms + plot
    fig, ax = plt.subplots()
    for region, region_label in regions.items():
        hist.plotratio(
            h.integrate('region', f'{region}_num'),
            h.integrate('region', f'{region}_den'),
            ax=ax,
            label=region_label,
            error_opts=error_opts,
            clear=False,
        )

    ax.legend(title='Passing')
    ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Trigger Efficiency')

    ax.text(0,1,'Muon 2022 F+G',
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes
    )
    
    # Save figure
    outpath = pjoin(outdir, f'L1ETMHF_check.pdf')
    fig.savefig(outpath)
    plt.close(fig)


def compare_turnons_with_PU60_fill(acc, outdir, region):
    """
    Compare turn-ons comparing the PU60 fill with other fills.
    """
    distribution = DISTRIBUTIONS[region]
    acc.load(distribution)
    h = acc[distribution]

    if distribution in NEW_BINS:
        new_ax = NEW_BINS[distribution]
        h = h.rebin(new_ax.name, new_ax)

    histograms = {}

    # 2022F histograms
    h_2022F = h.integrate("dataset", re.compile("Muon.*2022F"))
    histograms["2022F"] = {
        "num" : h_2022F.integrate("region", f"{region}_num"),
        "den" : h_2022F.integrate("region", f"{region}_den"),
    }

    # PU=60 histograms
    h_2022G = h.integrate("dataset", re.compile("Muon.*2022G"))

    histograms["2022G"] = {
        "num" : h_2022G.integrate("region", f"{region}_num"),
        "den" : h_2022G.integrate("region", f"{region}_den"),
    }

    histograms["2022G (PU=60)"] = {
        "num" : h_2022G.integrate("region", f"{region}_highpu_num"),
        "den" : h_2022G.integrate("region", f"{region}_highpu_den"),
    }

    # Plot the histograms!
    fig, ax = plt.subplots()
    for label, histos in histograms.items():
        hist.plotratio(
            histos["num"],
            histos["den"],
            ax=ax,
            error_opts=error_opts,
            clear=False,
            label=label
        )

    ax.set_ylabel("Trigger Efficiency")
    ax.legend(title="Dataset")

    ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')
    ax.set_ylim(bottom=0)
    
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

    outpath = pjoin(outdir, f"{region}_pu60_comparison.pdf")
    fig.savefig(outpath)
    plt.close(fig)


def compare_metnomu_turnon_for_different_thresh(acc, outdir, regions_num, region_den, dataset='Muon.*2022.*'):
    """
    Compare the METNoMu turn-on for different paths.
    """
    acc.load("recoil")
    h = acc["recoil"]

    h = h.integrate("dataset", re.compile(dataset))

    fig, ax = plt.subplots()
    for region_num, label in regions_num.items():
        hist.plotratio(
            h.integrate("region", region_num),
            h.integrate("region", region_den),
            ax=ax,
            error_opts=error_opts,
            label=label,
            clear=False,
        )

    ax.legend()
    ax.set_xlabel(r"Offline METNo$\mu$ [GeV]", horizontalalignment='right', x=1)
    ax.set_ylabel('Efficiency', verticalalignment='bottom', y=0.9)

    ax.set_xlim(80, 280)

    ax.grid(True, which='major')
    ax.set_ylim(bottom=0)
    
    ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')

    hep.cms.label(year="2022", paper=True, llabel=" Preliminary", rlabel=r"$34.3 \ fb^{-1}$, 2022 (13.6 TeV)")
    hep.cms.text()

    outpath = pjoin(outdir, "metnomu_turnon_comparison.pdf")
    fig.savefig(outpath)
    plt.close(fig)


def main():
    args = parse_cli()
    inpath = args.inpath
    acc = dir_archive(inpath)

    # Output directory to save plots
    outtag = os.path.basename(inpath.rstrip('/'))
    outdir = f'./output/{outtag}/latest'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Turn-on regions + initial parameter guesses for the fit 
    regions_fit_guesses = {
        'tr_jet' : (500, 50),
        'tr_ht' : (1050, 25),
        'tr_metnomu' : (200, 20),
        'tr_metnomu_filterhf' : (200, 20),
    }

    # Determine the fit function based on CLI argument
    if args.fit_func == 'sigmoid':
        fit_func = sigmoid
    else:
        fit_func = error_func

    for region, fit_init in tqdm(regions_fit_guesses.items(), desc='Plotting turn-ons'):
        if not re.match(args.region, region):
            continue
        
        plot_turnons_for_different_runs(acc, 
            outdir, 
            fit_init=fit_init,
            fit_func=fit_func,
            region=region,
            plot_fit=False,
        )

    # List of datasets to overlay in the same plot
    datasets = [
        {"regex": "Muon.*2022[CD]", "label": "2022C+D"},
        {"regex": "Muon.*2022E", "label": "2022E"},
        {"regex": "Muon.*2022F", "label": "2022F"},
        {"regex": "Muon.*2022G", "label": "2022G"},
    ]

    # Eta-separated plots for leading jet eta (PFJet500)
    try:
        plot_turnons_by_eta(acc, outdir, region='tr_jet', datasets=datasets)
    except KeyError:
        print('Skipping eta-split turn-on plots.')

    try:
        plot_eta_efficiency(acc, outdir, region='tr_jet', datasets=datasets)
    except KeyError:
        print('Skipping eta-based efficiency plots.')

    try:
        plot_turnons_with_without_water_leak(acc, outdir, dataset='Muon.*2022E')
    except KeyError:
        print('Skipping water-leak plots.')

    # L1 vs HLT turn-on plotting
    try:
        plot_l1_vs_hlt_HT1050(acc, outdir, dataset='Muon.*2022E.*')
    except KeyError:
        print('Skipping L1 vs HLT turn-on plots.')

    # PU=60 study
    try:
        compare_turnons_with_PU60_fill(acc, outdir, region='tr_jet')
        compare_turnons_with_PU60_fill(acc, outdir, region='tr_ht')
        compare_turnons_with_PU60_fill(acc, outdir, region='tr_metnomu')
    except KeyError:
        print('Skipping PU=60 plots.')

    # Efficiency vs Nvtx plots for MET/METNoMu triggers
    plot_efficiency_vs_nvtx(acc, outdir, distribution='recoil_npvgood', region='tr_metnomu', dataset='Muon.*2022.*')
    
    # plot_efficiency_vs_nvtx(acc, outdir, distribution='met_npvgood', region='tr_met')
    # plot_efficiency_vs_nvtx(acc, outdir, distribution='recoil_npvgood', region='tr_metnomu_filterhf')
    # plot_efficiency_vs_nvtx(acc, outdir, distribution='recoil_npvgood', region='tr_metnomu_L1ETMHF100')

    # plot_turnon_wrt_nvtx(acc, outdir, distribution='met_npvgood', region='tr_met', dataset='Muon.*2022.*')
    plot_turnon_wrt_nvtx(acc, outdir, distribution='recoil_npvgood', region='tr_metnomu', dataset='Muon.*2022.*')

    # Turn-on comparisons between two regions
    regions_to_compare = {
        'tr_metnomu' : 'METNoMu120',
        'tr_metnomu_L1ETMHF100' : 'METNoMu120 + L1ETMHF100',
    }
    # compare_turnons(acc, outdir, dataset='Muon.*2022[FG].*', regions=regions_to_compare)

    # 
    # METNoMu turn-on comparison for different thresholds
    # 
    regions_num = {
        f"tr_metnomu{thresh}_filterhf_num" : f"METNo$\\mu$ > {thresh} GeV" for thresh in [110,120,130,140]
    }
    # compare_metnomu_turnon_for_different_thresh(
    #     acc, 
    #     outdir,
    #     regions_num=regions_num,
    #     region_den="tr_metnomu_filterhf_den",
    #     dataset="Muon.*2022.*"
    # )

if __name__ == '__main__':
    main()