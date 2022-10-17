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
    jet_abseta_ax = Bin("jeteta", r"Jet $\eta$", 50, 0, 5)
    jet_phi_ax = Bin("jetphi", "Jet $\phi$", 50, -3.14, 3.14)
    dimu_mass_ax = Bin("dimumass", "Dimuon Mass (GeV)", 50, 60, 120) 
    recoil_ax = Bin("recoil", "Recoil (GeV)", 50, 0, 1000)
    met_ax = Bin("met", "MET (GeV)", 50, 0, 1000)
    ht_ax = Bin("ht", r"$H_{T}$ (GeV)", 200, 0, 4000)
    frac_ax = Bin('frac','Fraction', 50, 0, 1)

    # Histogram definitions
    items = {}
    items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["dimu_mass"] = Hist("Counts", dataset_ax, region_ax, dimu_mass_ax)
    items["recoil"] = Hist("Counts", dataset_ax, region_ax, recoil_ax)
    items["met"] = Hist("Counts", dataset_ax, region_ax, met_ax)
    items["ht"] = Hist("Counts", dataset_ax, region_ax, ht_ax)

    items["ak4_chf0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nhf0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_mufrac0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)

    items["ak4_abseta0_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_abseta_ax, jet_pt_ax)

    # Keep track of events that fail PFJet500
    items['selected_runs'] = processor.defaultdict_accumulator(list)  
    items['selected_lumis'] = processor.defaultdict_accumulator(list)  
    items['selected_events'] = processor.defaultdict_accumulator(list)  

    items['kinematics'] = processor.defaultdict_accumulator(list)

    # Return the accumulator of histograms
    return processor.dict_accumulator(items)

def setup_candidates(df, cfg):
    """
    Set up physics candidates as JaggedCandidateArray data structures, 
    from the given dataframe.
    """
    ak4 = JaggedCandidateArray.candidatesfromcounts(
        df['nJet'],
        pt=df['Jet_pt'],
        eta=df['Jet_eta'],
        abseta=np.abs(df['Jet_eta']),
        phi=df['Jet_phi'],
        mass=np.zeros_like(df['Jet_pt']),
        tightIdLepVeto=(df['Jet_jetId'] & 4) == 4, # bitmask: 1 = loose, 2 = tight, 3 = tight + lep veto
        cef=df['Jet_chEmEF'],
        chf=df['Jet_chHEF'],
        nef=df['Jet_neEmEF'],
        nhf=df['Jet_neHEF'],
        mufrac=df['Jet_muEF'],
        setaeta=df['Jet_hfsigmaEtaEta'],
        sphiphi=df['Jet_hfsigmaPhiPhi'],
        hfcentralstripsize=df['Jet_hfcentralEtaStripSize'],
    )

    met_pt = df['MET_pt']
    met_phi = df['MET_phi']
 
    #muons
    muons = JaggedCandidateArray.candidatesfromcounts(
        df['nMuon'],
        pt=df['Muon_pt'],
        eta=df['Muon_eta'],
        abseta=np.abs(df['Muon_eta']),
        phi=df['Muon_phi'],
        mass=0 * df['Muon_pt'],
        charge=df['Muon_charge'],
        looseId=df['Muon_looseId'],
        tightId=df['Muon_tightId'],
        iso=df["Muon_pfRelIso04_all"],
        dxy=df['Muon_dxy'],
        dz=df['Muon_dz'],
        globalmu = df['Muon_isGlobal'],
        pfcand = df['Muon_isPFcand']
    ) 

    #HLT muons
    hltMuons = JaggedCandidateArray.candidatesfromcounts(
        df['nTrigObj'],
        pt=df['TrigObj_pt'],
        eta=df['TrigObj_eta'],
        abseta=np.abs(df['TrigObj_eta']),
        phi=df['TrigObj_phi'],
        mass=0
    )

    # Pre-filter: All muons must be at least loose
    muons = muons[muons.looseId \
                    & (muons.iso < cfg.MUON.CUTS.LOOSE.ISO) \
                    & (muons.pt > cfg.MUON.CUTS.LOOSE.PT) \
                    & (muons.abseta < cfg.MUON.CUTS.LOOSE.ETA)
                    ]

    #HLT Muon matching
    #hltMuons = hltMuons[hltMuons.id == 13]
    #muons = muons[muons.match(hltMuons, deltaRCut = 0.4)]

    #electrons
    electrons = JaggedCandidateArray.candidatesfromcounts(
        df['nElectron'],
        pt=df['Electron_pt'],
        eta=df['Electron_eta'],
        abseta=np.abs(df['Electron_eta']),
        etasc=df['Electron_eta']+df['Electron_deltaEtaSC'],
        absetasc=np.abs(df['Electron_eta']+df['Electron_deltaEtaSC']),
        phi=df['Electron_phi'],
        mass=0 * df['Electron_pt'],
        charge=df['Electron_charge'],
        #looseId=(df[cfg.ELECTRON.BRANCH.ID]>=1),
        #tightId=(df[cfg.ELECTRON.BRANCH.ID]==4),
        dxy=np.abs(df['Electron_dxy']),
        dz=np.abs(df['Electron_dz']),
        barrel=np.abs(df['Electron_eta']+df['Electron_deltaEtaSC']) <= 1.4442
    )

    # All electrons must be at least loose
    pass_dxy = (electrons.barrel & (np.abs(electrons.dxy) < cfg.ELECTRON.CUTS.LOOSE.DXY.BARREL)) \
    | (~electrons.barrel & (np.abs(electrons.dxy) < cfg.ELECTRON.CUTS.LOOSE.DXY.ENDCAP))

    pass_dz = (electrons.barrel & (np.abs(electrons.dz) < cfg.ELECTRON.CUTS.LOOSE.DZ.BARREL)) \
    | (~electrons.barrel & (np.abs(electrons.dz) < cfg.ELECTRON.CUTS.LOOSE.DZ.ENDCAP))

    electrons = electrons[(electrons.pt>cfg.ELECTRON.CUTS.LOOSE.PT) \
                                    & (electrons.absetasc<cfg.ELECTRON.CUTS.LOOSE.ETA) \
                                    & pass_dxy \
                                    & pass_dz
                                    ]
    #taus
    taus = JaggedCandidateArray.candidatesfromcounts(
        df['nTau'],
        pt=df['Tau_pt'],
        eta=df['Tau_eta'],
        abseta=np.abs(df['Tau_eta']),
        phi=df['Tau_phi'],
        mass=0 * df['Tau_pt'],
        #decaymode=df[cfg.TAU.BRANCH.ID],
        decaymode=df['Tau_idDecayModeOldDMs'],
        iso=df['Tau_idDeepTau2017v2p1VSjet']
    )

    taus = taus[ (taus.decaymode) \
                & (taus.pt > cfg.TAU.CUTS.PT)\
                & (taus.abseta < cfg.TAU.CUTS.ETA) \
                & ((taus.iso&2)==2)]

    #photons
    if cfg.PHOTON.BRANCH.ID in df.keys():
        PHOTON_BRANCH_ID = cfg.PHOTON.BRANCH.ID
    else:
        PHOTON_BRANCH_ID = cfg.PHOTON.BRANCH.IDV7
    photons = JaggedCandidateArray.candidatesfromcounts(
        df['nPhoton'],
        pt=df['Photon_pt'],
        eta=df['Photon_eta'],
        abseta=np.abs(df['Photon_eta']),
        phi=df['Photon_phi'],
        mass=0*df['Photon_pt'],
        #looseId= (df[PHOTON_BRANCH_ID]>=1) & df['Photon_electronVeto'],
        #mediumId=(df[PHOTON_BRANCH_ID]>=2) & df['Photon_electronVeto'],
        r9=df['Photon_r9'],
        barrel=df['Photon_isScEtaEB'],
    )
    photons = photons[(photons.pt > cfg.PHOTON.CUTS.LOOSE.pt) \
              & (photons.abseta < cfg.PHOTON.CUTS.LOOSE.eta)
              ]


    return met_pt, met_phi, ak4, muons, electrons, taus, photons

def hlt_regions():
    """
    Returns the following mapping:
    Region name -> List of cuts for the region
    """
    regions = {}

    common_cuts = [
        'leadak4_id', 
        'lumi_mask',
        # 'filt_met',
        ]

    # W(mu nu) + jet selection for METNoMu trigger measurement
    metnomu_cuts = common_cuts + [
        'leadak4_pt_eta', 
        'at_least_one_tight_mu', 
        'one_muon',
        'HLT_IsoMu27',
        'muon_pt>30',
    ]

    regions['tr_metnomu_num'] = metnomu_cuts + ['HLT_PFMETNoMu120']
    regions['tr_metnomu_den'] = metnomu_cuts

    regions['tr_metnomu110_filterhf_num'] = metnomu_cuts + ['HLT_PFMETNoMu110_FilterHF']
    regions['tr_metnomu110_filterhf_den'] = metnomu_cuts

    # regions['tr_jet_num'] = common_cuts + ['HLT_PFJet500']
    # regions['tr_jet_den'] = common_cuts

    # PFJet500 trigger num and denom regions with energy fraction cuts
    # regions['tr_jet_num_energy_frac'] = common_cuts + ['leadak4_energy_frac'] + ['HLT_PFJet500']
    # regions['tr_jet_den_energy_frac'] = common_cuts + ['leadak4_energy_frac']

    return regions
