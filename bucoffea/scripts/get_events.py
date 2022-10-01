#!/usr/bin/env python

import os
import sys

from coffea.util import load

pjoin = os.path.join

def main():
    """
    Get the list of events in a specific region, and write them into
    an output events.txt file, in "run:lumi:event" format.
    """
    inpath = sys.argv[1]
    acc = load(inpath)

    # Name of the region having the saved events
    name = 'tr_jet_fail_jet500_high_ak4_pt0'

    runs = acc['selected_runs'][name]
    lumis = acc['selected_lumis'][name]
    events = acc['selected_events'][name]

    outfile = 'events.txt'
    
    with open(outfile, 'w+') as f:
        for run,lumi,event in zip(runs,lumis,events):
            f.write(f'{run}:{lumi}:{event}\n')

if __name__ == '__main__':
    main()


