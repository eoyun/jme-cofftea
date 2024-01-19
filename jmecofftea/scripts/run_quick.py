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
    parser.add_argument('output', type=str, help='output directory')
    args = parser.parse_args()

    return args

def main():
    # 
    # Define the mapping between dataset name and the corresponding list of files we want to run on.
    # 
    fileset = {
       #"JetMET0_2023B" : [
       #    "root://cluster142.knu.ac.kr//store/user/yeo/coffea/JetMET/060eed3c-d114-4135-b3f6-2cc6c8cf4c19.root"
       #],
       #"JetMET0_2023B" : [
       #    "root://xrootd-cms.infn.it//store/data/Run2023B/JetMET0/NANOAOD/22Sep2023-v1/2540000/060eed3c-d114-4135-b3f6-2cc6c8cf4c19.root"
       #],
       #"JetMET0_2023C" : [
       #    "root://xrootd-cms.infn.it//store/data/Run2023C/JetMET0/NANOAOD/22Sep2023_v1-v1/30000/1c01b41c-efc4-4170-91f2-a6e866a60ea7.root"
       #],
       #"JetMET0_2023D": [
       #    "root://xrootd-cms.infn.it//store/data/Run2023D/JetMET0/NANOAOD/22Sep2023_v1-v1/2530000/90b4ce31-2fb1-4822-8ca4-aef2c302761d.root"
       #],
       "JetMET0_2023D" : [
           "root://cluster142.knu.ac.kr//store/user/yeo/coffea/JetMET/90b4ce31-2fb1-4822-8ca4-aef2c302761d.root"
       ],
    }

    # years = list(set(map(extract_year, fileset.keys())))
    # assert(len(years)==1)

    args = parse_commandline()
    print(args)
    output = args.output
    processor_class = args.processor
    outdir = './output/'+output
    if not os.path.exists(outdir) :
        os.mkdir(outdir)

    # Currently, we only support parse_commandlinehlt processor.
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
        save(output, f"{outdir}/{processor_class}_{dataset}.coffea")
        # Debugging / testing output
        # debug_plot_output(output)
        print_cutflow(output, outfile=f'{processor_class}_cutflow_{dataset}.txt')

if __name__ == "__main__":
    main()
