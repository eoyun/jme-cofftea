#!/usr/bin/env python

import os
import sys
import re
import argparse
import numpy as np

from coffea import hist
from matplotlib import colors, pyplot as plt
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
    parser.add_argument('inpath', help='Path to the merged accumulator input.')
    parser.add_argument('-v', '--variable', help='Name of the 2D histogram to plot.')
    parser.add_argument('-x', '--xaxis', help='Name of the axis to plot on the x-axis.')
    parser.add_argument('-d', '--dataset', help='Regular expression specifying the dataset name to plot.')
    parser.add_argument('-r', '--region', help='Name of the region to integrate over.', default='sr_vbf')
    args = parser.parse_args()
    return args

def rebin(h: hist.Hist) -> hist.Hist:
    """Rebin several axes of the histogram if necessary."""
    new_bins = {
        'vpt' : hist.Bin("vpt",r"$p_{T}^{V}$ (GeV)", 50, 0, 2000)
    }

    for axis in h.axes():
        if axis.name in new_bins:
            h = h.rebin(axis.name, new_bins[axis.name])
    
    return h

def plot_2d_histogram(args):
    """Plots and saved the two-dimensional histogram."""
    acc = args.acc
    acc.load(args.variable)
    h = acc[args.variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('dataset', re.compile(args.dataset)).integrate('region', args.region)

    h = rebin(h)

    # Make the plot
    fig, ax = plt.subplots()
    hist.plot2d(h,
        ax=ax,
        xaxis=args.xaxis,
        patch_opts={'norm': colors.LogNorm(1e-3,1e3)},
    )

    ax.text(0,1,get_dataset_tag(args.dataset),
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes,
    )

    # Save figure
    outpath = pjoin(args.outdir, f'{args.variable}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')


def main():
    args = parse_cli()
    acc = dir_archive(args.inpath)
    acc.load('sumw')

    args.acc = acc

    outtag = os.path.basename(args.inpath.rstrip('/'))
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    args.outdir = outdir

    plot_2d_histogram(args)

if __name__ == '__main__':
    main()
