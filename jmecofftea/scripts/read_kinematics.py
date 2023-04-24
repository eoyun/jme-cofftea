#!/usr/bin/env python

import os
import sys

from coffea.util import load
from collections import defaultdict
from tabulate import tabulate

pjoin = os.path.join


def main():
    """
    Read data from "kinematics" accumulator, and print out a table,
    where each entry corresponds to a single event.
    """
    inpath = sys.argv[1]
    acc = load(inpath)

    # Read the accumulator
    info = acc['kinematics']
    num_events = len(info['event'])

    # Construct the table
    table = defaultdict(list)

    # List of column names to read from the "kinematics" accumulator
    columns = [
        'ak4_pt0',
        'ak4_eta0',
        'ak4_phi0',
        'ak4_nhf0',
        'ak4_nef0',
        'ak4_chf0',
        'ak4_cef0',
        'ak4_mufrac0',
        'mu_pt0',
        'mu_eta0',
        'mu_phi0',
        'mu_tightId0',
    ]

    for i in range(num_events):
        table['event'].append(info['event'][i])
        for column in columns:
            table[column].append(info[column][i][0])
    
    print(tabulate(table, headers="keys"))

if __name__ == '__main__':
    main()


