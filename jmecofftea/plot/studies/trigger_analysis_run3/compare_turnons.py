#!/usr/bin/env python3

import os
import sys
import re
import argparse
import numpy as np
import mplhep as hep

from matplotlib import pyplot as plt
from coffea import hist
from klepto.archives import dir_archive
from tqdm import tqdm

from jmecofftea.plot.style import trigger_names, binnings, markers, trigger_labels

pjoin = os.path.join

# Distributions to look at for each trigger
DISTRIBUTIONS = {
    'tr_met' : 'recoil',
    'tr_metnomu' : 'recoil',
    'tr_metnomu_filterhf' : 'recoil',
    'tr_jet' : 'ak4_pt0',
    'tr_ht' : 'ht',
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath", help="Path to the input coffea accumulator.")
    parser.add_argument("-d", "--datasets", nargs="+", help="The datasets to compare, each argument should be a regular expression.")
    parser.add_argument("-l", "--labels", nargs="+", help="The labels for the datasets to compare.")
    parser.add_argument('--cms-style', action='store_true', help='Use the CMS style in turn-on plots.')

    args = parser.parse_args()

    if not args.labels or not args.datasets:
        raise IOError("Datasets and labels must be specified, please see --help.")

    # Sanity check
    if len(args.datasets) != len(args.labels):
        raise IOError("Number of datasets and labels must match.")

    return args


def compare_turnons(acc, outdir, datasets, labels, region, use_cms_style=False):
    """
    Compare turn-ons for the given datasets.
    """
    distribution = DISTRIBUTIONS[region]
    acc.load(distribution)
    h = acc[distribution]

    if distribution in binnings():
        new_ax = binnings()[distribution]
        h = h.rebin(new_ax.name, new_ax)
    
    # Get the histograms for numerator and denominator
    h_num_all = h.integrate('region', f'{region}_num')
    h_den_all = h.integrate('region', f'{region}_den')

    error_opts = markers("data")

    # CMS plot styling
    if use_cms_style:
        plt.style.use(hep.style.CMS)
        error_opts["markersize"] = 14

    fig, ax = plt.subplots()

    for dataset, label in zip(datasets, labels):
        h_num = h_num_all.integrate("dataset", re.compile(dataset))
        h_den = h_den_all.integrate("dataset", re.compile(dataset))

        hist.plotratio(
            h_num,
            h_den,
            ax=ax,
            label=label,
            error_opts=error_opts,
            clear=False
        )

    ax.set_xlabel(binnings()[distribution].label, horizontalalignment='right', x=1)
    ax.set_ylabel('Efficiency', verticalalignment='bottom', y=0.9)

    ax.legend()
    ax.grid(True, which="major")

    # CMS plot styling
    if use_cms_style:
        hep.cms.label(year="2022", paper=True, llabel=" Preliminary", rlabel=r"(13.6 TeV)")
        hep.cms.text()

        ax.text(0.8, 0.05, trigger_labels()[region],
            fontsize=24,
            ha='center',
            va='bottom',
            transform=ax.transAxes,
        )

    else:
        ax.text(1,1,trigger_names()[region],
            fontsize=10,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

        ax.set_xlabel(binnings()[distribution].label, fontsize=14)
        ax.set_ylabel('Efficiency', fontsize=14)

    ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')
    ax.set_ylim(0,1.5)

    # Save plot
    outpath = pjoin(outdir, f"turnon_comparison_{region}.pdf")
    fig.savefig(outpath)
    plt.close(fig)


def main():
    args = parse_cli()

    acc = dir_archive(args.inpath)

    # Output directory to save plots
    outtag = os.path.basename(args.inpath.rstrip('/'))
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Triggers to compare across datasets
    regions = [
        "tr_jet",
        "tr_ht",
        "tr_metnomu",
        "tr_metnomu_filterhf",
    ]

    for region in tqdm(regions, desc="Plotting efficiencies"):
        compare_turnons(acc, outdir, args.datasets, args.labels, region, use_cms_style=args.cms_style)


if __name__ == "__main__":
    main()