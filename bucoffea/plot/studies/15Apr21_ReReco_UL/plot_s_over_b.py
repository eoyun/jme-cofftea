#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def plot_s_over_b(acc, outtag, hfestimatefile=None, distribution='mjj', region='sr_vbf_no_veto_all', year=2017):
    '''Plot S/sqrt(B) for the signal region.'''
    if not hfestimatefile:
        raise RuntimeError('Provide an HF estimate file!')
    
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h,acc)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if distribution == 'mjj':
        mjj_bins = [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.]
        mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', mjj_bins)
        h = h.rebin('mjj', mjj_ax)
    
    bkg = re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL_Pt_FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}')
    signal = re.compile(f'VBF_HToInvisible.*M125.*{year}')

    h = h.integrate('region', region)
    h_bkg = h.integrate('dataset', bkg)
    h_signal = h.integrate('dataset', signal)

    # Add the HF noise contribution
    hf_contribution = hfestimatefile[f'qcd_estimate_mjj_{year}'].values

    total_bkg = h_bkg.values()[()] + hf_contribution

    # Compute the ratio per bin
    ratio = h_signal.values()[()] / np.sqrt(total_bkg)
    xcenters = h_signal.axis(distribution).centers()
    xedges = h_signal.axis(distribution).edges()

    fig, ax = plt.subplots()
    plot_opts = {
        'linestyle' : '',
        'marker' : 'o',
    }
    ax.plot(xcenters, ratio, **plot_opts)

    ax.set_ylabel(r'S/$\sqrt{B}$')
    ax.set_xlabel(r'$M_{jj} \ (GeV)$')

    ax.set_ylim(0,10)

    outdir = f'./output/{outtag}/s_over_b'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f's_over_b_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    
    print(f'File saved: {outpath}')

    # Save to output ROOT file if one is provided
    outputrootpath = pjoin(outdir, 's_over_b.root')
    outputrootfile = uproot.recreate(outputrootpath)
    outputrootfile[f's_over_b_{year}'] = (ratio, xedges)

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw_pileup')
    acc.load('nevents')

    outtag = re.findall('merged_.*', inpath)[0].replace('/','')

    hfestimatefile = uproot.open(f'./output/{outtag}/qcd_estimate/hf_qcd_estimate.root')

    plot_s_over_b(acc, outtag, hfestimatefile=hfestimatefile)

if __name__ == '__main__':
    main()