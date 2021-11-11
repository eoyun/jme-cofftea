#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import mplhep as hep
import matplotlib

# Use a different backend for matplotlib, otherwise HEP styling doesn't work for some reason
matplotlib.use('tkagg')

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from coffea import hist
from coffea.hist import poisson_interval
from bucoffea.plot.util import merge_datasets, merge_extensions, merge_years, scale_xs_lumi, fig_ratio, lumi
from bucoffea.plot.plotter import legend_labels, legend_labels_IC, colors, colors_IC
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

# Use default CMS styling
plt.style.use(hep.style.CMS)
font = { 'size' : 16 }
matplotlib.rc('font', **font)

def pretty_eta_label(etaslice):
    return f'${etaslice.start:.2f} < |\\eta_{{j0}}| < {etaslice.stop:.2f}$' 

def preprocess(h, acc, etaslice, year):
    h = merge_extensions(h, acc)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if year == 'combined':
        h = merge_years(h)

    # Integrate out the eta slice
    h = h.integrate('jeteta', etaslice)
    return h

def get_qcd_estimation_for_etaslice(h, outtag, year, etaslice=slice(3, 3.25), fformat='pdf', logy=False):
    '''Get QCD estimation for the given leading jet eta slice.'''
    # QCD CR
    region = 'cr_vbf_qcd'
    h = h.integrate('region', region)

    if year in [2017, 2018]:
        data = f'MET_{year}'
        mc = re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL_Pt.*FXFX.*|WJetsToLNu_Pt.*FXFX.*).*{year}')
    elif year == 'combined':
        data = re.compile('MET.*')
        mc = re.compile('(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL_Pt.*FXFX.*|WJetsToLNu_Pt.*FXFX.*).*')

    h_data = h.integrate('dataset', data)
    h_mc = h.integrate('dataset', mc)

    # Data - MC gives the estimation in SR (TF pre-applied)
    h_mc.scale(-1)
    h_data.add(h_mc)

    fig, ax = plt.subplots()
    hist.plot1d(h_data, ax=ax, binwnorm=1)

    if logy:
        ax.set_yscale('log')
        ax.set_ylim(1e-4,1e4)
    ax.set_ylabel('HF Estimation / Bin Width')

    ax.get_legend().remove()
    ax.yaxis.set_ticks_position('both')

    ax.text(0.,1.,pretty_eta_label(etaslice),
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes
    )

    ax.text(1.,1.,year,
        fontsize=14,
        ha='right',
        va='bottom',
        transform=ax.transAxes
    )

    outdir = f'./output/{outtag}/dphitkpf'
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    etatag = f'{str(etaslice.start).replace(".", "_")}_{str(etaslice.stop).replace(".","_")}'

    outpath = pjoin(outdir, f'hf_estimation_eta_{etatag}_{year}.{fformat}')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

    # Return the histogram containing the QCD template
    return h_data

def plot_signal_over_total_bkg(h_signal, h_mc, h_qcd, outtag, year):
    '''Plot signal in the given eta slice divided by the total background.'''
    sumw_signal = h_signal.values()[()]
    sumw_bkg = h_mc.values()[()] + h_qcd.values()[()]

    ratio = sumw_signal / sumw_bkg
    xcenters =  h_signal.axis('dphi').centers()

    

    fig, ax = plt.subplots()
    opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }

    ax.plot(xcenters, ratio, label='Signal/Bkg', **opts)

    ax.set_xlabel(r'$\Delta\phi_{TK,PF}$')
    ax.set_ylabel('S / B Ratio')
    ax.set_xlim(0,np.pi)
    ax.set_ylim(0,0.3)

    avg = np.average(ratio)
    ax.axhline(avg, xmin=0, xmax=1, color='red', lw=2, label=f'Average S/B: {avg*100:.2f}%')

    ax.legend()
 
    hep.cms.label(ax=ax, 
            llabel="", # Just "CMS" label on the left hand side
            lumi=lumi(year) if year in [2017, 2018] else 101, # Combined luminosity = 101 fb^-1
            year=year if year in [2017, 2018] else None, 
            )

    outdir = f'./output/{outtag}/dphitkpf'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, 's_over_b.pdf')
    fig.savefig(outpath)
    plt.close(fig)

def plot_dphitkpf(acc, outtag, year, region='sr_vbf', distribution='dphitkpf_ak4_eta0', etaslice=slice(3, 3.25), fformat='pdf', logy=False, print_lastbin_yields=False, ic_theme=True, plot_pretty_eta_label=True):
    '''Plot dphitkpf distribution in data and MC in a stack plot, for the given eta slice for the leading jet.'''
    acc.load(distribution)
    h = preprocess(acc[distribution], acc, etaslice, year)

    # Get the QCD template
    h_qcd = get_qcd_estimation_for_etaslice(h, outtag, year, etaslice=etaslice, fformat=fformat)

    h = h.integrate('region', region)

    if year in [2017, 2018]:
        data = f'MET_{year}'
        mc = re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL_Pt.*FXFX.*|WJetsToLNu_Pt.*FXFX.*).*{year}')
    elif year == 'combined':
        data = re.compile('MET.*')
        mc = re.compile('(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL_Pt.*FXFX.*|WJetsToLNu_Pt.*FXFX.*).*')
    
    datasets = list(map(str, h[mc].identifiers('dataset')))
    
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
    }
    
    h_data = h.integrate('dataset', data)

    # Stack plot for MC
    plot_info = {
        'label' : [],
        'sumw' : [],
    }

    fig, ax, rax = fig_ratio()
    for dataset in datasets:
        sumw = h[mc].integrate('dataset', dataset).values()[()]
        if (sumw == 0.).all():
            continue
        plot_info['label'].append(dataset)
        plot_info['sumw'].append(sumw)

    # Add the QCD contribution!
    sumw_qcd = h_qcd.values()[()]
    sumw_qcd[sumw_qcd < 0] = 0.
    if not (sumw_qcd == 0).all():
        plot_info['label'].insert(0, 'HF noise')
        plot_info['sumw'].insert(0, sumw_qcd)

    xedges = h_data.axis('dphi').edges()

    hist.plot1d(h_data, ax=ax, binwnorm=1, error_opts=data_err_opts)

    hep.histplot(plot_info['sumw'], xedges, 
        ax=ax,
        label=plot_info['label'], 
        histtype='fill',
        binwnorm=1,
        stack=True
        )

    if year in [2017, 2018]:
        signal = re.compile(f'VBF_HToInvisible.*withDipoleRecoil.*{year}')
    elif year == 'combined':
        signal = re.compile(f'VBF_HToInvisible.*withDipoleRecoil.*')

    signal_line_opts = {
        'linestyle': '--',
        'linewidth': 2.,
        'color': (1,0,1),
    }

    if print_lastbin_yields:
        print(f'HF estimate: {sumw_qcd[-1]:.3f}')
        print(f'Data: {h_data.values()[()][-1]:.3f}')
        print(f'Signal: {h[signal].integrate("dataset").values()[()][-1]:.3f}')
        print(f'MC bkg: {h.integrate("dataset", mc).values()[()][-1]:.3f}')

    hist.plot1d(
        h[signal],
        ax=ax,
        overlay='dataset',
        line_opts=signal_line_opts,
        binwnorm=1,
        clear=False
    )

    plot_signal_over_total_bkg(
        h[signal].integrate('dataset'),
        h[mc].integrate('dataset'),
        h_qcd,
        outtag,
        year
    )

    if logy:
        ax.set_yscale('log')
        ax.set_ylim(1e-2,1e6)

    ax.set_ylabel('Events / Bin Width')
    ax.yaxis.set_ticks_position('both')

    colors_to_look = colors_IC if ic_theme else colors
    legend_labels_to_look = legend_labels_IC if ic_theme else legend_labels

    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label == 'None':
            handle.set_label('Data')
        for datasetregex, new_label in legend_labels_to_look.items():
            col = None
            edgecol = None
            if re.match(datasetregex, label):
                handle.set_label(new_label)
            for k, v in colors_to_look.items():
                if re.match(k, label):
                    col = (v[0]/255, v[1]/255, v[2]/255, 0.8)
                    edgecol = (v[0]/255, v[1]/255, v[2]/255, 0.99)
                    break

            if col and edgecol:
                handle.set_color(col)
                handle.set_linestyle('-')
                handle.set_linewidth(1.5)
                handle.set_edgecolor(edgecol)


    ax.legend(handles=handles, ncol=2, prop={'size' : 11})

    # CMS label & text
    hep.cms.label(ax=ax, 
            llabel="", # Just "CMS" label on the left hand side
            rlabel="$101 \\ \mathrm{fb^{-1}} \\ (13 \\ TeV)$",
            )
    
    if plot_pretty_eta_label:
        ax.text(0.95,0.15,pretty_eta_label(etaslice),
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

    # Plot ratio
    h_mc = h[mc].integrate('dataset', mc)

    sumw_data, sumw2_data = h_data.values(sumw2=True)[()]
    sumw_mc = h_mc.values()[()]
    # Add the QCD contribution to the MC
    sumw_mc = sumw_mc + sumw_qcd

    r = sumw_data / sumw_mc
    rerr = np.abs(poisson_interval(r, sumw2_data / sumw_mc**2) - r)

    r[np.isnan(r) | np.isinf(r)] = 0.
    rerr[np.isnan(rerr) | np.isinf(rerr)] = 0.

    hep.histplot(
        r,
        xedges,
        yerr=rerr,
        ax=rax,
        histtype='errorbar',
        **data_err_opts
    )

    ax.set_xlabel('$\\Delta\\phi \\ (rad)$')

    rax.set_xlabel('$\\Delta\\phi \\ (rad)$')
    rax.set_ylabel('Data / MC')
    rax.set_ylim(0.5,1.5)
    loc1 = MultipleLocator(0.2)
    loc2 = MultipleLocator(0.1)
    rax.yaxis.set_major_locator(loc1)
    rax.yaxis.set_minor_locator(loc2)

    rax.yaxis.set_ticks_position('both')

    sumw_denom, sumw2_denom = h_mc.values(sumw2=True)[()]

    unity = np.ones_like(sumw_denom)
    denom_unc = poisson_interval(unity, sumw2_denom / sumw_denom ** 2)
    opts = {"step": "post", "facecolor": (0, 0, 0, 0.3), "linewidth": 0}
    
    rax.fill_between(
        xedges,
        np.r_[denom_unc[0], denom_unc[0, -1]],
        np.r_[denom_unc[1], denom_unc[1, -1]],
        label='Bkg. uncert.',
        **opts
    )

    rax.legend(loc='upper left')

    rax.axhline(1., xmin=0, xmax=1, color=(0,0,0,0.4), ls='--')
    
    outdir = f'./output/{outtag}/dphitkpf'
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    etatag = f'{str(etaslice.start).replace(".", "_")}_{str(etaslice.stop).replace(".","_")}'
    
    outpath = pjoin(outdir, f'data_mc_eta_{etatag}_{year}.{fformat}')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw_pileup')
    acc.load('nevents')

    outtag = re.findall('merged_.*', inpath)[0].replace('/','')

    etaslices = [
        # slice(0, 2.5),
        # slice(2.5, 3),
        slice(3, 3.25),
        # slice(3.25, 5),
    ]
    
    for year in ['combined']:
        for etaslice in etaslices:
            for fformat in ['pdf']:
                plot_dphitkpf(acc, outtag, year=year, etaslice=etaslice, fformat=fformat)

if __name__ == '__main__':
    main()