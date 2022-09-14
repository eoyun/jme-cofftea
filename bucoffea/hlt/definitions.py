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
    jet_phi_ax = Bin("jetphi", "Jet $\phi$", 50, -3.14, 3.14)
    dimu_mass_ax = Bin("dimumass", "Dimuon Mass (GeV)", 50, 60, 120) 
    trigger_turnon_ax = Bin("turnon", "METnoMu120 (GeV)", 50, 0, 1000)
    met_ax = Bin("MET", "MET (GeV)", 50, 0, 1000)

    # Histogram definitions
    items = {}
    items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["dimu_mass"] = Hist("Counts", dataset_ax, region_ax, dimu_mass_ax)
    items["trigger_turnon"] = Hist("Counts", dataset_ax, region_ax, trigger_turnon_ax)
    items["met"] = Hist("Counts", dataset_ax, region_ax, met_ax)

    #keep track of events that failt metmht trigger
    items['selected_events'] = processor.defaultdict_accumulator(list)  

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
        looseId=(df['Jet_jetId']&2) == 2, # bitmask: 1 = loose, 2 = tight, 3 = tight + lep veto
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
        iso=df["Muon_pfRelIso04_all"],
        tightId=df['Muon_tightId'],
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
    muons = muons[(muons.iso < cfg.MUON.CUTS.LOOSE.ISO) \
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

    lepton_veto = ['veto_ele', 'veto_pho']

    #regions['my_regions'] = ['leadak4_pt_eta', 'leadak4_id']
    #regions['trigger w/o filter'] = ['leadak4_pt_eta', 'leadak4_id', 'mftmht_trig', 'at_least_one_tight_mu', 'dimuon_mass', 'dimuon_charge', 'two_muons', 'lumi_mask']
    #regions['trigger w/ filter'] = ['leadak4_pt_eta', 'leadak4_id', 'mftmht_clean_trig', 'at_least_one_tight_mu', 'dimuon_mass', 'dimuon_charge', 'two_muons', 'lumi_mask']

    #Z(mu mu) turn on regions
    #regions['clean turn on numerator'] = ['leadak4_pt_eta', 'leadak4_id', 'clean_mu', 'at_least_one_tight_mu', 'two_muons', 'dimuon_mass', 'dimuon_charge', 'HLT_IsoMu27', 'muon_pt>30', 'veto_ele', 'veto_pho', 'lumi_mask', 'filt_met', 'calo_diff', 'mftmht_clean_trig']
    #regions['turn on denominator'] = ['leadak4_pt_eta', 'leadak4_id', 'clean_mu', 'at_least_one_tight_mu', 'two_muons', 'dimuon_mass', 'dimuon_charge','HLT_IsoMu27', 'muon_pt>30', 'veto_ele', 'veto_pho', 'lumi_mask', 'filt_met', 'calo_diff']
    #regions['turn on numerator'] = ['leadak4_pt_eta', 'leadak4_id', 'clean_mu', 'at_least_one_tight_mu', 'two_muons', 'dimuon_mass', 'dimuon_charge', 'HLT_IsoMu27', 'muon_pt>30', 'veto_ele', 'veto_pho', 'lumi_mask', 'filt_met', 'calo_diff', 'mftmht_trig']

    #W(mu nu) turn on regions
    #regions['test'] = ['leadak4_pt_eta', 'leadak4_id', 'at_least_one_tight_mu', 'one_muon', 'clean_mu', 'HLT_IsoMu27'] + lepton_veto
    #regions['test w/ lumi'] = ['leadak4_pt_eta', 'leadak4_id', 'at_least_one_tight_mu', 'one_muon', 'clean_mu', 'HLT_IsoMu27', 'lumi_mask'] + lepton_veto
    #regions['test w/ lumi, filt_met'] = ['leadak4_pt_eta', 'leadak4_id', 'at_least_one_tight_mu', 'one_muon', 'clean_mu', 'HLT_IsoMu27', 'lumi_mask', 'filt_met'] + lepton_veto 
    #regions['test w/lumi, filt_met, calodiff'] = ['leadak4_pt_eta', 'leadak4_id', 'at_least_one_tight_mu', 'one_muon', 'clean_mu', 'HLT_IsoMu27', 'lumi_mask', 'filt_met', 'calo_diff'] + lepton_veto
    #regions['test w/lumi, filt, calod, met_trig'] = ['leadak4_pt_eta', 'leadak4_id', 'at_least_one_tight_mu', 'one_muon', 'clean_mu', 'HLT_IsoMu27', 'lumi_mask', 'filt_met', 'calo_diff', 'mftmht_trig'] + lepton_veto
    regions['clean turn on numerator'] = ['leadak4_pt_eta', 'leadak4_id', 'clean_mu', 'at_least_one_tight_mu', 'one_muon', 'lumi_mask', 'HLT_IsoMu27', 'muon_pt>30', 'filt_met', 'calo_diff', 'mftmht_clean_trig'] + lepton_veto
    regions['turn on denominator'] = ['leadak4_pt_eta', 'leadak4_id', 'clean_mu', 'at_least_one_tight_mu', 'one_muon', 'lumi_mask', 'HLT_IsoMu27', 'muon_pt>30', 'filt_met', 'calo_diff'] + lepton_veto
    regions['turn on numerator'] = ['leadak4_pt_eta', 'leadak4_id', 'clean_mu', 'at_least_one_tight_mu', 'one_muon', 'lumi_mask', 'HLT_IsoMu27', 'muon_pt>30', 'filt_met', 'calo_diff', 'mftmht_trig'] + lepton_veto
    #regions['failing metmht'] = ['leadak4_pt_eta', 'leadak4_id', 'clean_mu', 'at_least_one_tight_mu', 'one_muon', 'lumi_mask', 'HLT_IsoMu27', 'muon_pt>30', 'filt_met', 'calo_diff', 'fail_metmht_trig', 'recoil>250'] + letpon_veto

    #Jet Phi w/ w/o muon cuts
    #regions['no muon cleaning'] = ['leadak4_pt_eta', 'leadak4_id']
    #regions['muon cleaning'] = ['leadak4_pt_eta', 'leadak4_id', 'clean_mu', 'at_least_one_tight_mu', 'one_muon', 'veto_ele', 'veto_pho', 'lumi_mask', 'HLT_IsoMu27', 'muon_pt>30', 'filt_met', 'calo_diff']

    #W(e nu) turn on regions
    #cr_1e_cuts = ['trig_ele','one_electron', 'at_least_one_tight_el', 'veto_muo', 'calo_diff', 'filt_met', 'hlt_ele']
    #regions['clean turn on numerator'] = ['leadak4_pt_eta', 'leadak4_id', 'veto_pho', 'lumi_mask', 'mftmht_clean_trig'] + cr_1e_cuts
    #regions['turn on denominator'] = ['leadak4_pt_eta', 'leadak4_id', 'veto_pho', 'lumi_mask'] + cr_1e_cuts
    #regions['turn on numerator'] = ['leadak4_pt_eta', 'leadak4_id', 'veto_pho', 'lumi_mask', 'mftmht_trig'] + cr_1e_cuts

    return regions
