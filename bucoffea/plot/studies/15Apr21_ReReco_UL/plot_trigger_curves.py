#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np

from bucoffea.helpers import sigmoid3
from matplotlib import pyplot as plt

pjoin = os.path.join

def main():
    outdir = './output/met_trigger'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outrootpath = pjoin(outdir, 'met_trigger_turnons.root')

    outrootfile = uproot.recreate(outrootpath)
    recoil_pt = np.linspace(0,1000,201)

    # Inclusive parametrization of MET trigger turn-ons!
    data_params = {
        2017 : (0.047, 165.573, 0.989),
        2018 : (0.045, 175.975, 0.991)
    }

    mc_params = {
        2017 : (0.048, 150.315, 0.992),
        2018 : (0.049, 158.786, 0.992)
    }

    for year in [2017, 2018]:
        t_data = sigmoid3(recoil_pt, *data_params[year])
        t_mc = sigmoid3(recoil_pt, *mc_params[year])

        fig, ax = plt.subplots()
        ax.plot(recoil_pt, t_data, label='Data')
        ax.plot(recoil_pt, t_mc, label='MC')

        ax.legend()

        outfile = pjoin(outdir, f'trigger_eff_{year}.pdf')
        fig.savefig(outfile)
        plt.close(fig)

        # Save to output ROOT file
        outrootfile[f'met_trigger_turnon_data_{year}'] = (t_data[:-1], recoil_pt)
        outrootfile[f'met_trigger_turnon_mc_{year}'] = (t_mc[:-1], recoil_pt)

if __name__ == "__main__":
    main()