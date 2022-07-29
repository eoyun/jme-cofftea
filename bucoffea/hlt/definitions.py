import re
import copy

import coffea.processor as processor
import numpy as np
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray

Hist = hist.Hist
Bin = hist.Bin
Cat = hist.Cat

def hlt_accumulator():
    """
    Returns an accumulator, mapping each histogram name to the relevant hist.Hist object.
    """
    # Axis definitions for histograms
    # Categorical axes
    dataset_ax = Cat("dataset", "Primary dataset")
    region_ax = Cat("region", "Selection region")

    # Numerical axes
    jet_pt_ax = Bin("jetpt", r"Jet $p_{T}$ (GeV)", 50, 0, 1000)
    jet_eta_ax = Bin("jeteta", r"Jet $\eta$", 50, -5, 5)

    # Histogram definitions
    items = {}
    items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)

    # Return the accumulator of histograms
    return processor.dict_accumulator(items)

def setup_candidates(df):
    """
    Set up physics candidates as JaggedCandidateArray data structures, 
    from the given dataframe.
    """
    pass

def hlt_regions():
    """
    Returns the following mapping:
    Region name -> List of cuts for the region
    """
    pass