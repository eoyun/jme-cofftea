#!/usr/bin/env python

import os
import sys
import re
import argparse

from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive

from bucoffea.plot.util import (
    merge_datasets, 
    merge_extensions, 
    scale_xs_lumi, 
    get_dataset_tag
    )

pjoin = os.path.join

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to the merged input accumulator.')
    parser.add_argument('-d', '--distribution', help='Name of the distribution to plot.', default='cnn_score')
    parser.add_argument('-o', '--overlay', help='Name of the distribution to overlay the plot.', default='mjj')
    parser.add_argument('--dataset', help='Name of the dataset the plot.', default='ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX_2017')

    args = parser.parse_args()
    return args

def plot_overlayed(acc, args):
    """
    Make an overlayed plot of a given variable (args.distribution), overlayed with another
    variable (args.overlay). Currently we only support distribution = cnn_score and overlay = mjj.

    Will look at the contents for the signal region by default, and args.dataset specifies
    the dataset to look at. 
    """
    acc.load(args.variable)
    h = acc[args.variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Look at the signal region
    region = 'sr_vbf_no_veto_all'
    h = h.integrate('region', region).integrate('dataset', args.dataset)

    # Rebin the mjj axis
    new_mjj_bins = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200, 500, 1000, 7500])
    h = h.rebin('mjj', new_mjj_bins)

    fig, ax = plt.subplots()
    hist.plot1d(
        h,
        ax=ax,
        overlay='mjj',
    )

    ax.text(0,1,get_dataset_tag(args.dataset),
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes,
    )

    outpath = pjoin(args.outdir, f'{args.variable}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved to: {outpath}')


def main():
    args = parse_cli()
    acc = dir_archive(args.inpath)
    acc.load('sumw')

    outtag = os.path.basename(args.inpath.rstrip('/'))
    outdir = f'./output/{outtag}/overlayed'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    args.outdir = outdir

    # Deduct the 2D histogram name to plot
    if args.distribution == 'cnn_score' and args.overlay == 'mjj':
        args.variable = 'cnn_score_mjj'
    else:
        raise NotImplementedError(f'Overlay plot not implemented for: {args.distribution}, {args.overlay}')

    plot_overlayed(acc, args)

if __name__ == '__main__':
    main()