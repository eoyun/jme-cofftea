#!/usr/bin/env python

import os
import sys
import re
import csv
import uproot
import numpy as np
from glob import glob
from tqdm import tqdm
from pprint import pprint

pjoin = os.path.join

def dump_run_event_lumi(inpath, region, year, outdir):
    outfile = pjoin(outdir, f'{region}_{year}.csv')
    
    infile = uproot.open(inpath)
    df = infile[region].pandas.df()

    # Convert run,event,lumi to CSV
    for item in ['run','event','lumi']:
        df[item] = df[item].astype(int)

    df.to_csv(outfile, index=False, columns=['run','lumi','event','mjj'])
    print(f'File saved: {outfile}')

def main():
    indir = sys.argv[1]

    regions_files = {
        'sr_vbf'    : pjoin(indir,'tree_MET_combined_{year}.root'),
        'cr_1m_vbf' : pjoin(indir,'tree_MET_combined_{year}.root'),
        'cr_2m_vbf' : pjoin(indir,'tree_MET_combined_{year}.root'),
        'cr_1e_vbf' : pjoin(indir,'tree_EGamma_combined_{year}.root'),
        'cr_2e_vbf' : pjoin(indir,'tree_EGamma_combined_{year}.root'),
        'cr_g_vbf'  : pjoin(indir,'tree_EGamma_combined_{year}.root'),
    }
    
    # Set output directory based on the input directory
    outdir = indir.replace('input', 'output')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        for region, inpath in regions_files.items():
            dump_run_event_lumi(inpath.format(year=year), region, year, outdir)
    
if __name__ == '__main__':
    main()