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
    'tr_met' : 'met',
    'tr_metnomu' : 'recoil',
    'tr_metnomu_filterhf' : 'recoil',
    'tr_jet' : 'ak4_pt0',
    'tr_ht' : 'ht',
}


def validate_cli_args(args):
    """
    Validate command line interface arguments.
    """
    # Cannot specify both datasets and region tags.
    if args.tags and args.datasets:
        raise IOError("Specifying both -t and -d is not supported (please see --help).")

    # The case where different datasets are specified
    if args.datasets:
        # Number of datasets and labels must match.
        if len(args.datasets) != len(args.labels):
            raise IOError("Number of datasets and labels must match.")
        
    # No datasets or tags are specified, this is an error
    if not args.datasets and not args.tags:
        raise IOError("Region tags (-t) or datasets (-d) must be specified, please see --help.")


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath", help="Path to the input coffea accumulator.")
    parser.add_argument("-d", "--datasets", nargs="+", help="The datasets to compare, each argument should be a regular expression. Cannot be specified together with -t.")
    parser.add_argument("-l", "--labels", nargs="+", help="The labels for the datasets to compare.")
    parser.add_argument("-t", "--tags", nargs="+", help="Different region tags to compare efficiencies for. Cannot be specified together with -d.")
    parser.add_argument('--cms-style', action='store_true', help='Use the CMS style in turn-on plots.')

    args = parser.parse_args()

    # Validate command line arguments
    validate_cli_args(args)

    return args


def get_histogram(acc, region):
    """
    From the passed accumulator, obtain the histogram and return it.
    """
    distribution = DISTRIBUTIONS[region]
    acc.load(distribution)
    h = acc[distribution]

    # Rebinning (if necessary)
    if distribution in binnings():
        new_ax = binnings()[distribution]
        h = h.rebin(new_ax.name, new_ax)

    return h


def compare_turnons_for_diff_datasets(acc, outdir, datasets, labels, region, use_cms_style=False):
    """
    Compare turn-ons for the given datasets.
    """
    distribution = DISTRIBUTIONS[region]
    h = get_histogram(acc, region)

    # Get the histograms for numerator and denominator
    h_num = h.integrate('region', f'{region}_num')
    h_den = h.integrate('region', f'{region}_den')

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


def compare_turnons_for_diff_regions(acc, outdir, region_tags, base_region, dataset, use_cms_style=False):
    """
    Compare turn-ons for different region tags. 
    """
    distribution = DISTRIBUTIONS[base_region]
    h = get_histogram(acc, base_region).integrate("dataset", re.compile(dataset))
    
    error_opts = markers("data")
    # CMS plot styling
    if use_cms_style:
        plt.style.use(hep.style.CMS)
        error_opts["markersize"] = 14

    # Plot efficiencies for each region tag
    fig, ax = plt.subplots()

    for region_tag in region_tags:
        # Get the histograms for numerator and denominator regions
        h_num = h.integrate("region", f"{base_region}_num_{region_tag}")
        h_den = h.integrate("region", f"{base_region}_den_{region_tag}")

        hist.plotratio(
            h_num,
            h_den,
            ax=ax,
            label=region_tag,
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

            ax.text(0.8, 0.05, trigger_labels()[base_region],
                fontsize=24,
                ha='center',
                va='bottom',
                transform=ax.transAxes,
            )

        else:
            ax.text(1,1,trigger_names()[base_region],
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
        outpath = pjoin(outdir, f"turnon_comparison_region_tags_{base_region}.pdf")
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
        "tr_met",
        "tr_metnomu",
        "tr_metnomu_filterhf",
    ]

    # Compare different datasets
    if args.datasets:
        for region in tqdm(regions, desc="Plotting efficiencies"):
            compare_turnons_for_diff_datasets(acc, outdir, args.datasets, args.labels, region, use_cms_style=args.cms_style)

    # Compare different regions
    elif args.tags:
        for base_region in tqdm(regions, desc="Plotting efficiencies"):
            compare_turnons_for_diff_regions(acc, outdir, args.tags, base_region, dataset="Muon.*2023.*", use_cms_style=args.cms_style)


if __name__ == "__main__":
    main()