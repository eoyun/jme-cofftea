#!/usr/bin/env python

from bucoffea.helpers.dataset import extract_year
from bucoffea.processor.executor import run_uproot_job_nanoaod
from bucoffea.helpers.cutflow import print_cutflow
from coffea.util import save
import coffea.processor as processor
import argparse

def parse_commandline():

    parser = argparse.ArgumentParser()
    parser.add_argument('processor', type=str, help='The processor to be run. (monojet or vbfhinv)')
    args = parser.parse_args()

    return args

def main():
    fileset = {
        "JetMET_2022C" : [
            "root://cmsxrootd.fnal.gov//store/data/Run2022C/JetMET/NANOAOD/PromptNanoAODv10-v1/30000/11d5498c-2a79-4848-b22e-558e2850a81c.root"
        ],
        "Muon_2022F" : [
            "root://cmsxrootd.fnal.gov//store/data/Run2022F/Muon/NANOAOD/PromptNanoAODv10_v1-v2/2530000/36220bbd-e1a5-4f68-a606-0b1da6682e41.root"
        ],
        "Muon_2022G" : [
            "root://cmsxrootd.fnal.gov//store/data/Run2022G/Muon/NANOAOD/PromptNanoAODv10_v1-v1/2820000/1959f286-7f5e-492c-97ac-5c34ae05010b.root"
        ],
    }

    years = list(set(map(extract_year, fileset.keys())))
    assert(len(years)==1)

    args = parse_commandline()
    processor_class = args.processor

    if processor_class == 'monojet':
        from bucoffea.monojet import monojetProcessor
        processorInstance = monojetProcessor()
    elif processor_class == 'vbfhinv':
        from bucoffea.vbfhinv import vbfhinvProcessor
        processorInstance = vbfhinvProcessor()
    elif processor_class == 'lhe':
        from bucoffea.gen.lheVProcessor import lheVProcessor
        processorInstance = lheVProcessor()
    elif args.processor == 'purity':
        from bucoffea.photon_purity import photonPurityProcessor
        processorInstance = photonPurityProcessor()
    elif args.processor == 'sumw':
        from bucoffea.gen import mcSumwProcessor
        processorInstance = mcSumwProcessor()
    elif args.processor == 'gen':
        from bucoffea.gen.genVbfProcessor import genVbfProcessor
        processorInstance = genVbfProcessor()
    elif args.processor == 'hlt':
        from bucoffea.hlt.hltProcessor import hltProcessor
        processorInstance = hltProcessor()

    for dataset, filelist in fileset.items():
        newlist = []
        for file in filelist:
            if file.startswith("/store/"):
                newlist.append("root://cms-xrd-global.cern.ch//" + file)
            else: newlist.append(file)
        fileset[dataset] = newlist

    for dataset, filelist in fileset.items():
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
