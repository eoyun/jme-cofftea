#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np

from tabulate import tabulate
from pprint import pprint

def compute_total_jes_unc(infile, dataset, year):
    '''Given the file with JES uncertainties and the dataset name, compute the total JEC
    unc at low and high mjj.'''
    keys = [k.decode('utf-8').replace(';1','') for k in infile.keys()]
    uncs = list(filter(lambda x: re.match(f'{dataset}.*', x) and str(year) in x and 'jer' not in x, keys))
    upvars = list(filter(lambda x: 'Up' in x, uncs))
    downvars = list(filter(lambda x: 'Down' in x, uncs))

    total_up = {
        'low_mjj' : 0,
        'high_mjj' : 0,
    }
    total_down = {
        'low_mjj' : 0,
        'high_mjj' : 0,
    }

    for uvar, dvar in zip(upvars, downvars):
        h_upunc = infile[uvar]
        h_downunc = infile[dvar]

        total_up['low_mjj'] += (h_upunc.values[0]-1)**2
        total_up['high_mjj'] += (h_upunc.values[-1]-1)**2
        total_down['low_mjj'] += (h_downunc.values[0]-1)**2
        total_down['high_mjj'] += (h_downunc.values[-1]-1)**2

    total_up['low_mjj'] = np.sqrt(total_up['low_mjj'])
    total_up['high_mjj'] = np.sqrt(total_up['high_mjj'])
    total_down['low_mjj'] = np.sqrt(total_down['low_mjj'])
    total_down['high_mjj'] = np.sqrt(total_down['high_mjj'])

    pprint(total_up)
    pprint(total_down)

def main():
    inpath = 'vbf_shape_jes_uncs_jer_smeared.root'
    infile = uproot.open(inpath)

    for year in [2017, 2018]:
        for dataset in ['VBF_HToInvisible', 'ZJetsToNuNu']:
            print(dataset, year)
            compute_total_jes_unc(infile, dataset=dataset, year=year)


if __name__ == '__main__':
    main()