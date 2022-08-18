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

    rootfiles0 = []
    path = "/eos/user/c/cericeci/Run3/data/"
    for i in range(40):
        rootfiles0.append(path + "SingleMuon_C_" + str(i+1) + ".root")
        #rootfiles0.append(path + "MET_C_" + str(i+1) + ".root")

    rootfiles0.remove(path + "SingleMuon_C_16.root")

    fileset = {
        #"dimuon_mass-SingleMuon-2017C" : rootfiles0,
        #"MET-all_tightid_withtrig-2017C" : rootfiles0,
        #"trigger-turnon-SingleMuon-2017C" : rootfiles0,
        "SingleMuon-2017C" : rootfiles0,
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
