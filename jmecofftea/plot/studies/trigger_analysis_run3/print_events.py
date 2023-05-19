#!/usr/bin/env python

import os
import sys
import csv
import numpy as np

from klepto.archives import dir_archive

pjoin = os.path.join

def print_events_to_csv(acc, region, outdir):
    """
    Save (run,lumi,event) data for events in an analysis region to
    an output CSV file.
    """
    # Load arrays into memory
    quantities = ["selected_runs", "selected_lumis", "selected_events"]
    for q in quantities:
        acc.load(q)

    events = []
    events.append(["Run", "Lumi", "Event"])

    for run,lumi,event in zip(acc["selected_runs"][region], acc["selected_lumis"][region], acc["selected_events"][region]):
        events.append([int(run), int(lumi), int(event)])

    # Dump to CSV file
    outpath = pjoin(outdir, f"events_{region}.csv")

    with open(outpath, "w+") as f:
        writer = csv.writer(f)

        for event in events:
            writer.writerow(event)

    print(f"Events are saved to: {outpath}")


def main():
    inpath = sys.argv[1]
    
    acc = dir_archive(inpath)

    # Output directory to save output
    outtag = os.path.basename(inpath.rstrip('/'))
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Save (run,lumi,event) to CSV file
    print_events_to_csv(acc, region="tr_fail_ht1050", outdir=outdir)


if __name__ == "__main__":
    main()
