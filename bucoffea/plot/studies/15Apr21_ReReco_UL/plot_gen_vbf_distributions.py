#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import warnings
import numpy as np
import mplhep as hep

from matplotlib import pyplot as plt
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio, URTH1
from klepto.archives import dir_archive
from tqdm import tqdm
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

def plot_distribution(acc, outtag, region, dataset, outrootfile, distribution='gen_mjj'):
    '''Plot Z(ll) / W(lv) GEN-level ratios for both years'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h, scale_lumi=False)
    h = merge_datasets(h)

    if distribution == 'gen_mjj':
        mjj_bins = list(np.arange(200,2000,200)) + [2000,2500,3000,3500,5000]
        mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', mjj_bins)
        h = h.rebin('mjj', mjj_ax)

    h = h.integrate('region', region)

    xlabels = {
        'ak4_pt0' : r'Leading Jet $p_T \ (GeV)$',
        'ak4_pt1' : r'Trailing Jet $p_T \ (GeV)$',
    }

    for year in [2017, 2018]:
        datasetregex = {
            'QCDZll' : re.compile(f'DYJetsToLL_Pt.*FXFX.*{year}'),
            'EWKZll' : re.compile(f'EWKZ2Jets.*ZToLL.*{year}'),
            'QCDWlv' : re.compile(f'WJetsToLNu_Pt.*FXFX.*{year}'),
            'EWKWlv' : re.compile(f'EWKW.*Jets.*WToLNu.*{year}'),
        }

        fig, ax = plt.subplots()

        hist.plot1d(h[datasetregex[dataset]],
            ax=ax,
            binwnorm=1,
            overlay='dataset'
            )

        ax.set_yscale('log')
        
        if 'dilepton' in distribution:
            ax.set_ylim(1e-1,1e5)
        elif 'mjj' in distribution:
            ax.set_ylim(1e-5,1e3)
        else:
            ax.set_ylim(1e-3,1e5)

        ax.text(0,1,dataset,
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        ax.text(1,1,year,
            fontsize=14,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

        if distribution in xlabels.keys():
            ax.set_xlabel(xlabels[distribution])

        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outpath = pjoin(outdir, f'{dataset}_{distribution}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        # Save to output ROOT file
        _h = h[datasetregex[dataset]].integrate('dataset')
        xedges = _h.axes()[0].edges()
        sumw, sumw2 = _h.values(sumw2=True)[()]

        # Overflow and underflow

        sumw = np.r_[0,sumw,0]
        sumw2 = np.r_[0,sumw2,0]

        outrootfile[f'{dataset}_{distribution}_{year}'] = URTH1(xedges, sumw=sumw, sumw2=sumw2)

def plot_ratios(rootfile, outtag, distribution='gen_mjj'):
    '''Plot Z/W ratios for 2017 and 2018.'''
    sumw = {
        2017: {},
        2018: {},
    }
    sumw2 = {
        2017: {},
        2018: {},
    }

    for year in [2017, 2018]:
        for dataset in ['QCDZll', 'EWKZll', 'QCDWlv', 'EWKWlv']:
            sumw[year][f'{dataset}_{distribution}'] = rootfile[f'{dataset}_{distribution}_{year}'].values
            sumw2[year][f'{dataset}_{distribution}'] = rootfile[f'{dataset}_{distribution}_{year}'].variances
            xedges = rootfile[f'{dataset}_{distribution}_{year}'].edges

        # Combine QCD + EWK V
        sumw[year][f'QCD_EWKZll_{distribution}'] = sumw[year][f'QCDZll_{distribution}'] + sumw[year][f'EWKZll_{distribution}']
        sumw[year][f'QCD_EWKWlv_{distribution}'] = sumw[year][f'QCDWlv_{distribution}'] + sumw[year][f'EWKWlv_{distribution}']

        sumw2[year][f'QCD_EWKZll_{distribution}'] = sumw2[year][f'QCDZll_{distribution}'] + sumw2[year][f'EWKZll_{distribution}']
        sumw2[year][f'QCD_EWKWlv_{distribution}'] = sumw2[year][f'QCDWlv_{distribution}'] + sumw2[year][f'EWKWlv_{distribution}']

    fig, ax, rax = fig_ratio()
    r_2017 = sumw[2017][f'QCD_EWKZll_{distribution}'] / sumw[2017][f'QCD_EWKWlv_{distribution}']
    r_2018 = sumw[2018][f'QCD_EWKZll_{distribution}'] / sumw[2017][f'QCD_EWKWlv_{distribution}']
    
    sumw2_num_2017 = sumw2[2017][f'QCD_EWKZll_{distribution}']
    r_err_2017 = np.abs(
        hist.poisson_interval(r_2017, sumw2_num_2017 / sumw[2017][f'QCD_EWKWlv_{distribution}']**2) - r_2017
    )

    sumw2_num_2018 = sumw2[2018][f'QCD_EWKZll_{distribution}']
    r_err_2018 = np.abs(
        hist.poisson_interval(r_2018, sumw2_num_2018 / sumw[2018][f'QCD_EWKWlv_{distribution}']**2) - r_2018
    )

    hep.histplot(r_2017, xedges, yerr=r_err_2017, ax=ax, label='2017')
    hep.histplot(r_2018, xedges, yerr=r_err_2018, ax=ax, label='2018')

    ax.legend(title='Year')
    ax.set_ylabel('Ratio')
    ax.set_xlabel(r'$M_{jj} \ (GeV)$')

    ax.text(0,1,r'$Z(\ell\ell) \ / \ W(\ell\nu)$',
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes
    )

    ax.text(1,1,'GEN',
        fontsize=14,
        ha='right',
        va='bottom',
        transform=ax.transAxes
    )

    # 2017/2018 ratio
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
    }

    rr = r_2017 / r_2018
    rr_err = r_err_2017 / r_2018

    xcenters = (xedges[:-1] + xedges[1:]) / 2.0
    
    rax.errorbar(xcenters, rr, yerr=rr_err, **data_err_opts)

    rax.grid(True)
    rax.set_ylim(0.6,1.4)
    rax.set_ylabel('2017 / 2018')
    rax.set_xlabel(r'$M_{jj} \ (GeV)$')

    rax.axhline(0.9, xmin=0, xmax=1, color='green', ls='--', label='10%')
    rax.axhline(1.1, xmin=0, xmax=1, color='green', ls='--')

    rax.legend()
    
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'ratio.pdf')
    fig.savefig(outpath)
    plt.close(fig)

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)

    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/','')

    datasets_regions = {
        'QCDZll' : 'cr_z_vbf',
        'EWKZll' : 'cr_z_vbf',
        'QCDWlv' : 'cr_w_vbf',
        'EWKWlv' : 'cr_w_vbf',
    }

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outrootpath = pjoin(outdir, 'vjets_yields.root')
    outrootfile = uproot.recreate(outrootpath)

    distributions = [
        'gen_mjj', 
        'ak4_pt0',
        'ak4_pt1',
        'dilepton_mass',
    ]

    for dataset, region in tqdm(datasets_regions.items()):
        for distribution in distributions:
            try:
                plot_distribution(acc, 
                    outtag, 
                    dataset=dataset, 
                    region=region,
                    distribution=distribution,
                    outrootfile=outrootfile
                    )
            except AssertionError:
                print(f'Skipping: {dataset}, {distribution}')
                continue

    plot_ratios(outrootfile, outtag)

if __name__ == '__main__':
    main()