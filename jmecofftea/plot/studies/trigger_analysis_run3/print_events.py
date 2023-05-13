#!/usr/bin/env python

import os
import sys
import csv
import json
import numpy as np

from klepto.archives import dir_archive

pjoin = os.path.join

def print_run_lumi_event(acc, region, outdir):
    """
    Prepare a table of (run,lumi,event) records for events 
    in a given analysis region.
    """
    # Load arrays into memory
    quantities = ["selected_runs", "selected_lumis", "selected_events"]
    for q in quantities:
        acc.load(q)

    data = []
    for run,lumi,event in zip(acc["selected_runs"][region], acc["selected_lumis"][region], acc["selected_events"][region]):
        data.append({
            "run" : int(run),
            "lumi" : int(lumi),
            "event" : int(event),
        })

    # Dump to JSON file
    outpath = pjoin(outdir, f"events_{region}.json")

    with open(outpath, "w+") as f:
        json.dump(data, f, indent=4)

    print(f"Events are saved to: {outpath}")


def main():
    inpath = sys.argv[1]
    
    acc = dir_archive(inpath)

    # Output directory to save output
    outtag = os.path.basename(inpath.rstrip('/'))
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print_run_lumi_event(acc, region="tr_fail_ht1050", outdir=outdir)


if __name__ == "__main__":
    main()
