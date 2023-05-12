#!/usr/bin/env python

from jmecofftea.helpers.dataset import extract_year
from jmecofftea.processor.executor import run_uproot_job_nanoaod
from jmecofftea.helpers.cutflow import print_cutflow
from coffea.util import save
import coffea.processor as processor
import argparse

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
        "JetMET_2022C" : [
            "root://cmsxrootd.fnal.gov//store/data/Run2022C/JetMET/NANOAOD/PromptNanoAODv10-v1/30000/11d5498c-2a79-4848-b22e-558e2850a81c.root"
        ],
        "Muon_2022F" : [
            "root://cmsxrootd.fnal.gov//store/data/Run2022F/Muon/NANOAOD/PromptNanoAODv10_v1-v2/2530000/36220bbd-e1a5-4f68-a606-0b1da6682e41.root"
        ],
        "Muon0_2023B" : [
            "root://cmsxrootd.fnal.gov//store/data/Run2023B/Muon0/NANOAOD/PromptNanoAODv11p9_v1-v1/30000/76bdd785-b80d-4607-8dee-dc2aa806cc0b.root"
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
    else:
        raise ValueError(f"Unknown value given for the processor argument: {args.processor}")

    for dataset, filelist in fileset.items():
        newlist = []
        for file in filelist:
            if file.startswith("/store/"):
                newlist.append("root://cms-xrd-global.cern.ch//" + file)
            else: newlist.append(file)
        fileset[dataset] = newlist

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
        save(output, f"{processor_class}_{dataset}.coffea")
        # Debugging / testing output
        # debug_plot_output(output)
        print_cutflow(output, outfile=f'{processor_class}_cutflow_{dataset}.txt')

if __name__ == "__main__":
    main()
