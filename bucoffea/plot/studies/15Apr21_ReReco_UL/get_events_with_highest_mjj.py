#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
import pandas as pd

pjoin = os.path.join

def dump_events_with_highest_mjj(tree, numevents=20):
    df = tree.pandas.df(['run','event','lumi','mjj']).astype({
        'event' : 'int32',
        'run' : 'int32',
        'lumi' : 'int32',
    })
    
    # Sort with mjj
    df.sort_values(
        by='mjj',
        ascending=False,
        inplace=True
        )

    df.reset_index(inplace=True, drop=True)

    outdir = './output/09Dec21'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, "high_mjj_events_MET_2018.csv")
    df.head(numevents).to_csv(outpath, index=False)

def main():
    infile = sys.argv[1]
    tree = uproot.open(infile)['sr_vbf']

    dump_events_with_highest_mjj(tree)

if __name__ == '__main__':
    main()