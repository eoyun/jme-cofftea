#!/usr/bin/env python

import os
import sys
import re
import warnings
import numpy as np
import mplhep as hep

from matplotlib import pyplot as plt
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from klepto.archives import dir_archive
from scipy.optimize import curve_fit
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

ylims = {
    'ak4_eta0' : (0,1),
    'ak4_eta1' : (0,1.5),
    'ak4_pt0' : (0,1),
    'ak4_pt1' : (0,1),
    'mjj' : (0,2),
}

xlabels = {
    'ak4_eta0' : r'Leading Jet $\eta$',
    'ak4_eta1' : r'Trailing Jet $\eta$',
    'ak4_pt0' : r'Leading Jet $p_T \ (GeV)$',
    'ak4_pt1' : r'Trailing Jet $p_T \ (GeV)$',
    'mjj' : r'$M_{jj} \ (GeV)$',
}

def linear(x, a, b):
    return a*x + b

def mjj_bins():
    return list(np.arange(200,3000,200)) + [3000,3500]

def plot_ratio(acc, outtag, distribution='mjj', tag='electrons', regions=['cr_2e_vbf', 'cr_1e_vbf'], year=2017):
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if 'ak4_pt' in distribution:
        new_ax = hist.Bin('jetpt',r'Jet $p_{T}$ (GeV)',list(range(80,600,20)) + list(range(600,1000,20)) )
        h = h.rebin('jetpt', new_ax)

    # Mjj binning
    if distribution == 'mjj':
        mjj_ax = hist.Bin("mjj", r"$M_{jj}$ (GeV)", mjj_bins())
        h = h.rebin('mjj', mjj_ax)

    h_num = h.integrate('region', regions[0])
    h_den = h.integrate('region', regions[1])

    data = {
        'cr_1m_vbf' : f'MET_{year}',
        'cr_2m_vbf' : f'MET_{year}',
        'cr_1e_vbf' : f'EGamma_{year}',
        'cr_2e_vbf' : f'EGamma_{year}',
    }

    mc = {
        'cr_1m_vbf' : re.compile(f'(EWKW.*|EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|DYJetsToLL_Pt_FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}'),
        'cr_1e_vbf' : re.compile(f'(EWKW.*|EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|DYJetsToLL_Pt_FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}'),
        'cr_2m_vbf' : re.compile(f'(EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|DYJetsToLL_Pt_FXFX.*).*{year}'),
        'cr_2e_vbf' : re.compile(f'(EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|DYJetsToLL_Pt_FXFX.*).*{year}'),
    }

    minor_bkg = {
        'cr_1m_vbf' : re.compile(f'(EWKW.*|Top_FXFX.*|Diboson.*).*{year}'),
        'cr_1e_vbf' : re.compile(f'(EWKW.*|Top_FXFX.*|Diboson.*).*{year}'),
        'cr_2m_vbf' : re.compile(f'(Top_FXFX.*|Diboson.*).*{year}'),
        'cr_2e_vbf' : re.compile(f'(Top_FXFX.*|Diboson.*).*{year}'),
    }

    h_data_num = h_num.integrate('dataset', data[regions[0]])
    h_data_den = h_den.integrate('dataset', data[regions[1]])
    
    h_minorbkg_num = h_num.integrate('dataset', minor_bkg[regions[0]])
    h_minorbkg_den = h_den.integrate('dataset', minor_bkg[regions[1]])

    h_mc_num = h_num.integrate('dataset', mc[regions[0]])
    h_mc_den = h_den.integrate('dataset', mc[regions[1]])

    # Ratio: data - minor bkg / MC
    sumw_data_num, sumw2_data_num = h_data_num.values(sumw2=True)[()]
    sumw_data_den, _ = h_data_den.values(sumw2=True)[()]

    sumw_minorbkg_num, sumw2_minorbkg_num = h_minorbkg_num.values(sumw2=True)[()]
    sumw_minorbkg_den, _ = h_minorbkg_den.values(sumw2=True)[()]

    data_num = sumw_data_num - sumw_minorbkg_num
    data_den = sumw_data_den - sumw_minorbkg_den

    sumw2_num = sumw2_data_num - sumw2_minorbkg_num

    r_data = data_num / data_den

    r_data_err = np.abs(
        hist.poisson_interval(r_data, sumw2_num / data_den**2) - r_data
    )

    sumw_mc_num, sumw2_mc_num = h_mc_num.values(sumw2=True)[()]
    sumw_mc_den, _ = h_mc_den.values(sumw2=True)[()]

    r_mc = sumw_mc_num / sumw_mc_den
    
    r_mc_err = np.abs(
        hist.poisson_interval(r_mc, sumw2_mc_num / sumw_mc_den**2) - r_mc
    )

    if distribution == 'mjj':
        axname = 'mjj'
    elif 'ak4_eta' in distribution:
        axname = 'jeteta'
    elif 'ak4_pt' in distribution:
        axname = 'jetpt'
    else:
        raise RuntimeError(f'Check distribution: {distribution}')
    
    xedges = h_mc_num.axis(axname).edges()
    xcenters = h_mc_num.axis(axname).centers()

    r_data[np.isnan(r_data) | np.isinf(r_data)] = -1.
    r_mc[np.isnan(r_mc) | np.isinf(r_mc)] = -1.

    fig, ax, rax = fig_ratio()

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }

    hep.histplot(r_data, 
        xedges, 
        yerr=r_data_err, 
        ax=ax, 
        histtype='errorbar', 
        label='Data / Data', 
        **data_err_opts
        )
        
    hep.histplot(r_mc, 
        xedges, 
        yerr=r_mc_err, 
        ax=ax, 
        label='MC / MC'
        )

    ax.legend()
    ax.set_ylim(ylims[distribution])

    ylabels = {
        'electrons' : r'$Z(ee) \ / \ W(e\nu)$',
        'muons' : r'$Z(\mu\mu) \ / \ W(\mu\nu)$',
    }

    ax.set_ylabel(ylabels[tag])

    ax.text(1,1,year,
        fontsize=14,
        ha='right',
        va='bottom',
        transform=ax.transAxes
    )

    rr = r_data / r_mc
    rr_err = r_data_err / r_mc
    rax.errorbar(xcenters, rr, yerr=rr_err, **data_err_opts)

    if distribution == 'mjj':
        # Fit the double ratio with a linear function
        sigma = 0.5 * np.abs(rr_err[0] + rr_err[1])
    
        popt, _ = curve_fit(linear, xdata=xcenters, ydata=rr, sigma=sigma, p0=[5e-4,1])
    
        rax.plot(xcenters, linear(xcenters, *popt), color='red', label=f'{popt[0]:.5f}*x + {popt[1]:.3f}')

        rax.legend()

    rax.set_xlabel(xlabels[distribution])
    rax.set_ylabel('Data / MC')
    rax.set_ylim(0.5,5)
    rax.grid(True)

    if distribution != 'mjj':
        rax.axhline(1, xmin=0, xmax=1, color='red')
    else:
        rax.axhline(1, xmin=0, xmax=1, color='blue', linestyle='--')

    outdir = f'./output/{outtag}/fine_mjj_binning'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'ratio_{distribution}_{tag}_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)

    acc.load('sumw')
    acc.load('sumw_pileup')
    acc.load('nevents')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    tag_regions = {
        'electrons' : ['cr_2e_vbf', 'cr_1e_vbf'],
        'muons' : ['cr_2m_vbf', 'cr_1m_vbf'],
    }

    for tag, regions in tag_regions.items():
        for distribution in ['mjj']:
            plot_ratio(acc, outtag, distribution=distribution, tag=tag, regions=regions)

if __name__ == '__main__':
    main()