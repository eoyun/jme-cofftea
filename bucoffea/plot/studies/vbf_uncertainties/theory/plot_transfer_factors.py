#!/usr/bin/env python

import os
import sys
import re
import argparse
import numpy as np
import mplhep as hep

from matplotlib import pyplot as plt
from coffea import hist
from tqdm import tqdm
from klepto.archives import dir_archive
from pprint import pprint

from bucoffea.plot.util import (
    merge_extensions, 
    merge_datasets, 
    scale_xs_lumi, 
    rebin_histogram,
    fig_ratio,
    )

pjoin = os.path.join

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to the merged accumulator input.')
    parser.add_argument('-v', '--variable', help='The variable to plot the uncertainties for.', default='cnn_score', choices=['cnn_score','mjj'])
    args = parser.parse_args()
    return args

def get_uncertainty_tag(unc: str) -> str:
    mapping = {
        "unc_(zoverw|goverz)_nlo_muf_down" : r"$\mu_F$ down",
        "unc_(zoverw|goverz)_nlo_muf_up" : r"$\mu_F$ up",
        "unc_(zoverw|goverz)_nlo_mur_down" : r"$\mu_R$ down",
        "unc_(zoverw|goverz)_nlo_mur_up" : r"$\mu_R$ up",
        "unc_(zoverw|goverz)_nlo_pdf_down" : r"$PDF$ down",
        "unc_(zoverw|goverz)_nlo_pdf_up" : r"$PDF$ up",
    }
    for regex, label in mapping.items():
        if re.match(regex, unc):
            return label

    raise ValueError(f"Cannot find tag for uncertainty: {unc}")

def plot_transfer_factors(acc, 
        transfer_factors: dict,
        distribution: str,
        outdir: str,
    ) -> None:
    """
    Plot the set of transfer factors as a function of the given variable.
    """
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = rebin_histogram(h, distribution)

    for name, transfer_factor in tqdm(transfer_factors.items(), desc="Plotting transfer factors"):
        h_num = h \
            .integrate("dataset", transfer_factor["num"]["dataset"]) \
            .integrate("region",  transfer_factor["num"]["region"])
        
        h_den = h \
            .integrate("dataset", transfer_factor["denom"]["dataset"]) \
            .integrate("region",  transfer_factor["denom"]["region"])


        data_err_opts = {
            'linestyle':'-',
            'elinewidth': 1,
            'marker' : 'o',
        }

        if "variations" in transfer_factor:
            fig, ax, rax = fig_ratio()
        else:
            fig, ax = plt.subplots()
        
        # Plot theory variations on the transfer factor
        if "variations" in transfer_factor:
            dist = f'{distribution}_unc'
            acc.load(dist)
            h_unc = acc[dist]

            h_unc = merge_extensions(h_unc, acc, reweight_pu=False)
            scale_xs_lumi(h_unc)
            h_unc = merge_datasets(h_unc)
            h_unc = rebin_histogram(h_unc, distribution)
            
            h_varied_den = h_unc \
                .integrate("dataset", transfer_factor["denom"]["dataset"]) \
                .integrate("region",  transfer_factor["denom"]["region"])


            nominal_ratio = h_num.values()[()] / h_den.values()[()]

            for unc in transfer_factor["variations"]:
                hist.plotratio(
                    h_num,
                    h_varied_den.integrate("uncertainty", unc),
                    ax=ax,
                    unc='num',
                    error_opts=data_err_opts,
                    label=get_uncertainty_tag(unc),
                    clear=False,
                )

                # Ratio of ratios
                # varied_ratio = h_varied_num.integrate("uncertainty",unc).values()[()] / h_den.values()[()]
                varied_ratio = h_num.values()[()] / h_varied_den.integrate("uncertainty", unc).values()[()]
                
                double_ratio = varied_ratio / nominal_ratio
                rax.plot(
                    h_num.axes()[0].centers(),
                    double_ratio,
                    marker='o',
                    ls='',
                )

            ax.legend(ncol=2, title="Uncertainty")
            
            rax.set_ylim(0.8,1.2)
            rax.grid(True)
            rax.set_xlabel("CNN score")
            rax.set_ylabel("Ratio")

        # Plot the nominal transfer factor
        hist.plotratio(
            h_num, h_den,
            ax=ax,
            unc="num",
            error_opts=data_err_opts,
            clear=False,
        )

        if "ylim" in transfer_factor:
            ax.set_ylim(*transfer_factor["ylim"])
        
        ax.set_xlim(0,1)
        ax.grid(True)
        ax.set_ylabel('Transfer Factor')

        ax.text(0,1,name,
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes,
        )

        outpath = pjoin(outdir, f"{name}.pdf")
        fig.savefig(outpath)
        plt.close(fig)


def main():
    args = parse_cli()

    acc = dir_archive(args.inpath)
    acc.load('sumw')

    outtag = os.path.basename(args.inpath.rstrip('/'))

    # Output directory to save plots
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Define the transfer factors to plot
    transfer_factors = {
        "wlv_over_zvv" : {
            "num"   : {"dataset" : "WJetsToLNu_Pt-FXFX_2017",                  "region" : "sr_vbf"},
            "denom" : {"dataset" : "ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX_2017", "region" : "sr_vbf"},
            "variations" : [
                "unc_zoverw_nlo_muf_down",
                "unc_zoverw_nlo_muf_up",
                "unc_zoverw_nlo_mur_down",
                "unc_zoverw_nlo_mur_up",
                "unc_zoverw_nlo_pdf_down",
                "unc_zoverw_nlo_pdf_up",
            ],
            "ylim" : (0,1),
        },
        "gjets_over_zvv" : {
            "num"   : {"dataset" : "GJets_DR-0p4_HT_MLM_2017",                 "region" : "cr_g_vbf"},
            "denom" : {"dataset" : "ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX_2017", "region" : "sr_vbf"},
            "variations" : [
                "unc_goverz_nlo_muf_down",
                "unc_goverz_nlo_muf_up",
                "unc_goverz_nlo_mur_down",
                "unc_goverz_nlo_mur_up",
                "unc_goverz_nlo_pdf_down",
                "unc_goverz_nlo_pdf_up",
            ],
            "ylim" : (0,1),
        },
        "zmumu_over_zvv" : {
            "num"    : {"dataset" : "DYJetsToLL_Pt_FXFX_2017",                  "region" : "cr_2m_vbf"},
            "denom"  : {"dataset" : "ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX_2017", "region" : "sr_vbf"},
        },
    }

    plot_transfer_factors(acc, 
        transfer_factors, 
        distribution=args.variable,
        outdir=outdir,
        )

if __name__ == '__main__':
    main()
