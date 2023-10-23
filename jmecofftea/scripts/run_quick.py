#!/usr/bin/env python

from jmecofftea.helpers.dataset import extract_year
from jmecofftea.processor.executor import run_uproot_job_nanoaod
from jmecofftea.helpers.cutflow import print_cutflow
from coffea.util import save
import coffea.processor as processor
import argparse
import os

def parse_commandline():

    parser = argparse.ArgumentParser()
    parser.add_argument('processor', type=str, help='The processor to be run. (monojet or vbfhinv)')
    args = parser.parse_args()

    return args

def main():
    # 
    # Define the mapping between dataset name and the corresponding list of files we want to run on.
    # 
    fileset = {
        "Muon0_2023B" : [
            "root://cmsxrootd.fnal.gov//store/data/Run2023B/Muon0/NANOAOD/PromptNanoAODv11p9_v1-v2/2810000/1074b310-64e3-4ad4-90f0-443a3c80ad37.root"
        ],
       "Muon0_2023C" : [
           "root://cmsxrootd.fnal.gov//store/data/Run2023C/Muon0/NANOAOD/PromptNanoAODv11p9_v1-v1/70000/9d003698-9b74-40b5-b34c-24c33f4b8bef.root"
       ],
       "Muon0_2023D": [
           "root://cmsxrootd.fnal.gov//store/data/Run2023D/Muon0/NANOAOD/PromptReco-v1/000/369/956/00000/05056be2-5638-4f7f-b504-59365c0e570d.root"
       ],
    }

    # years = list(set(map(extract_year, fileset.keys())))
    # assert(len(years)==1)

    args = parse_commandline()
    processor_class = args.processor

    # Currently, we only support hlt processor.
    if args.processor == 'hlt':
        from jmecofftea.hlt.hltProcessor import hltProcessor
        processorInstance = hltProcessor()
    if args.processor == 'hlt_dijet':
        from jmecofftea.hlt.hltProcessor_dijet import hltProcessor
        processorInstance = hltProcessor()
    else:
        raise ValueError(f"Unknown value given for the processor argument: {args.processor}")

    for dataset, filelist in fileset.items():
        newlist = []
        for file in filelist:
            if file.startswith("/store/"):
                newlist.append("root://cms-xrd-global.cern.ch//" + file)
            else: newlist.append(file)
        fileset[dataset] = newlist
    path = './dijet'
    if not os.path.exists(path) :
        os.mkdir(path)
    for dataset, filelist in fileset.items():
        print(f"Running on dataset: {dataset}")
        print(f"Number of files: {len(filelist)}")
        tmp = {dataset:filelist}
        output = run_uproot_job_nanoaod(tmp,
                                    treename='Runs' if args.processor=='sumw' else 'Events',
                                    processor_instance=processorInstance,
                                    executor=processor.futures_executor,
                                    executor_args={'workers': 4, 'flatten': True},
                                    chunksize=500000,
                                    )
        save(output, f"./dijet/{processor_class}_{dataset}.coffea")
        # Debugging / testing output
        # debug_plot_output(output)
        print_cutflow(output, outfile=f'./dijet/{processor_class}_cutflow_{dataset}.txt')

if __name__ == "__main__":
    main()
