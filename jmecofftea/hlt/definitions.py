import re
import copy

import coffea.processor as processor
import numpy as np

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray, JaggedTLorentzVectorArray

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
    nvtx_ax = Bin('nvtx','Number of vertices',100,-0.5,99.5)

    # Histogram definitions
    items = {}
    items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["sub_ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["sub_ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["sub_ak4_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["delta_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["dimu_mass"] = Hist("Counts", dataset_ax, region_ax, dimu_mass_ax)
    items["recoil"] = Hist("Counts", dataset_ax, region_ax, recoil_ax)
    items["met"] = Hist("Counts", dataset_ax, region_ax, met_ax)
    items["ht"] = Hist("Counts", dataset_ax, region_ax, ht_ax)

    items["ak4_chf0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nhf0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_mufrac0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)

    items["ak4_abseta0_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_abseta_ax, jet_pt_ax)

    # PU-related plots
    items["met_npv"] = Hist("Counts", dataset_ax, region_ax, met_ax, nvtx_ax)
    items["met_npvgood"] = Hist("Counts", dataset_ax, region_ax, met_ax, nvtx_ax)
    items["recoil_npv"] = Hist("Counts", dataset_ax, region_ax, recoil_ax, nvtx_ax)
    items["recoil_npvgood"] = Hist("Counts", dataset_ax, region_ax, recoil_ax, nvtx_ax)

    # Keep track of events that pass specific regions
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
    # AK4 PF PUPPI jets
    # If we are going to manually apply different JECs, take the raw pt from NanoAOD.
    # Otherwise, get the jet pt straight out of NanoAOD.
    ak4 = JaggedCandidateArray.candidatesfromcounts(
        df['nJet'],
        pt=df['Jet_pt']*(1-df['Jet_rawFactor']) if cfg.JECS.OFFLINE.APPLY else df['Jet_pt'],
        eta=df['Jet_eta'],
        abseta=np.abs(df['Jet_eta']),
        phi=df['Jet_phi'],
        mass=np.zeros_like(df['Jet_pt']),
        tightIdLepVeto=(df['Jet_jetId'] & 4) == 4, # bitmask: 1 = loose, 2 = tight, 3 = tight + lep veto
        area=df['Jet_area'],
        cef=df['Jet_chEmEF'],
        chf=df['Jet_chHEF'],
        nef=df['Jet_neEmEF'],
        nhf=df['Jet_neHEF'],
        mufrac=df['Jet_muEF'],
        setaeta=df['Jet_hfsigmaEtaEta'],
        sphiphi=df['Jet_hfsigmaPhiPhi'],
        hfcentralstripsize=df['Jet_hfcentralEtaStripSize'],
    )

    # Offline MET, by default we use PUPPI.
    met_pt = df['PuppiMET_pt']
    met_phi = df['PuppiMET_phi']
 
    # Muons
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

    # Pre-filter: All muons must be at least loose
    muons = muons[muons.looseId \
                    & (muons.iso < cfg.MUON.CUTS.LOOSE.ISO) \
                    & (muons.pt > cfg.MUON.CUTS.LOOSE.PT) \
                    & (muons.abseta < cfg.MUON.CUTS.LOOSE.ETA)
                    ]

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

def hlt_regions(cfg):
    """
    Returns the following mapping:
    Region name -> List of cuts for the region
    """
    regions = {}

    common_cuts = [
        'leadak4_id', 
        'subleadak4_id', 
        #'lumi_mask',
        #'at_least_one_tight_mu', 
        #'one_muon',
        #'HLT_IsoMu27',
        #'muon_pt>30',
        # 'filt_met',
        ]

    regions['tr_jet_num'] = common_cuts + ['HLT_PFJet80']
    regions['tr_jet_40'] = common_cuts + ['HLT_PFJet40']
    regions['tr_jet_60'] = common_cuts + ['HLT_PFJet60']
    regions['tr_jet_80'] = common_cuts + ['HLT_PFJet80']
    regions['tr_jet_140'] = common_cuts + ['HLT_PFJet140']
    regions['tr_jet_200'] = common_cuts + ['HLT_PFJet200']
    regions['tr_jet_260'] = common_cuts + ['HLT_PFJet260']
    regions['tr_jet_320'] = common_cuts + ['HLT_PFJet320']
    regions['tr_jet_400'] = common_cuts + ['HLT_PFJet400']
    regions['tr_jet_450'] = common_cuts + ['HLT_PFJet450']
    regions['tr_jet_500'] = common_cuts + ['HLT_PFJet500']
   # regions['tr_dijet'] = common_cuts + ['dijet']
    regions['tr_jet_den'] = common_cuts

    # Jet500 regions where the leading jet is in water leak region
    # vs. NOT in the water leak region
    if cfg.STUDIES.WATER_LEAK:
        regions['tr_jet_water_leak_veto_num'] = copy.deepcopy(regions['tr_jet_num'])
        regions['tr_jet_water_leak_veto_num'].append('ak4_not_in_water_leak')

        regions['tr_jet_water_leak_veto_den'] = copy.deepcopy(regions['tr_jet_den'])
        regions['tr_jet_water_leak_veto_den'].append('ak4_not_in_water_leak')

        regions['tr_jet_water_leak_num'] = copy.deepcopy(regions['tr_jet_num'])
        regions['tr_jet_water_leak_num'].append('ak4_in_water_leak')

        regions['tr_jet_water_leak_den'] = copy.deepcopy(regions['tr_jet_den'])
        regions['tr_jet_water_leak_den'].append('ak4_in_water_leak')

    # Additional jet requirement for the HT and MET triggers
    cuts_for_ht_met = common_cuts + ['leadak4_pt_eta']

    regions['tr_ht_num'] = cuts_for_ht_met# + ['HLT_PFHT1050']
    regions['tr_ht_den'] = cuts_for_ht_met

    regions['tr_met_num'] = cuts_for_ht_met# + ['HLT_PFMET120']
    regions['tr_met_den'] = cuts_for_ht_met

    regions['tr_metnomu_num'] = cuts_for_ht_met# + ['HLT_PFMETNoMu120']
    regions['tr_metnomu_den'] = cuts_for_ht_met

    regions['tr_metnomu_filterhf_num'] = cuts_for_ht_met# + ["HLT_PFMETNoMu120_FilterHF"]
    regions['tr_metnomu_filterhf_den'] = cuts_for_ht_met

    # Studies for the L1 turn-on for HT1050
    if cfg.STUDIES.L1_TURNON:
        regions['tr_l1_ht_num'] = cuts_for_ht_met# + ['L1_pass_HT1050']
        regions['tr_l1_ht_den'] = cuts_for_ht_met

    regions['tr_fail_ht1050'] = cuts_for_ht_met# + ['offline_ht_gt_1050', 'fail_PFHT1050']

    if cfg.STUDIES.HIGH_PU_FILL:
        regions['tr_metnomu_highpu_num'] = cuts_for_ht_met# + ['HLT_PFMETNoMu120', 'pu60_fill']
        regions['tr_metnomu_highpu_den'] = cuts_for_ht_met# + ['pu60_fill']

        regions['tr_ht_highpu_num'] = cuts_for_ht_met# + ['HLT_PFHT1050', 'pu60_fill']
        regions['tr_ht_highpu_den'] = cuts_for_ht_met# + ['pu60_fill']

        regions['tr_jet_highpu_num'] = cuts_for_ht_met# + ['HLT_PFJet40', 'pu60_fill']
        regions['tr_jet_highpu_den'] = cuts_for_ht_met# + ['pu60_fill']

    # Tracker BPIX issue. We will look at regions where:
    # 1. The offline leading jet is inside the impacted region
    # 2. The offline leading jet is NOT inside the impacted region
    # For the purposes of this study, all the events considered are taken during the issue
    # (i.e., there is a constraint on the run number.)
    if cfg.STUDIES.TRK_BPIX_ISSUE:
        regions_to_clone = [
            'tr_jet',
            'tr_ht',
            'tr_met',
            'tr_metnomu',
            'tr_metnomu_filterhf',
        ]

        for base_region in regions_to_clone:
            # Regions where the leading jet is in the impacted tracker region
            regions[f"{base_region}_ak4_in_bad_bpix_num"] = copy.deepcopy(regions[f"{base_region}_num"])
            regions[f"{base_region}_ak4_in_bad_bpix_num"].append("ak4_in_bad_trk")
            regions[f"{base_region}_ak4_in_bad_bpix_num"].append("bpix_issue")
            
            regions[f"{base_region}_ak4_in_bad_bpix_den"] = copy.deepcopy(regions[f"{base_region}_den"])
            regions[f"{base_region}_ak4_in_bad_bpix_den"].append("ak4_in_bad_trk")
            regions[f"{base_region}_ak4_in_bad_bpix_den"].append("bpix_issue")

            # Regions where the leading jet is NOT in the impacted tracker region
            regions[f"{base_region}_ak4_not_in_bad_bpix_num"] = copy.deepcopy(regions[f"{base_region}_num"])
            regions[f"{base_region}_ak4_not_in_bad_bpix_num"].append("ak4_not_in_bad_trk")
            regions[f"{base_region}_ak4_not_in_bad_bpix_num"].append("bpix_issue")

            regions[f"{base_region}_ak4_not_in_bad_bpix_den"] = copy.deepcopy(regions[f"{base_region}_den"])
            regions[f"{base_region}_ak4_not_in_bad_bpix_den"].append("ak4_not_in_bad_trk")
            regions[f"{base_region}_ak4_not_in_bad_bpix_den"].append("bpix_issue")


    # Regions with specific run ranges: The idea is to copy the numerator and denominator
    # region for each trigger and apply the run range cut on top.
    for label, run_range in cfg.RUN.RANGES.items():
        regions_to_clone = [
	    'tr_jet_40',
	    'tr_jet_60',
	    'tr_jet_80',
	    'tr_jet_140',
	    'tr_jet_200',
	    'tr_jet_260',
	    'tr_jet_320',
	    'tr_jet_400',
	    'tr_jet_450',
	    'tr_jet_500',
            'tr_jet_num',
            'tr_jet_den',
            'tr_ht_num',
            'tr_ht_den',
            'tr_met_num',
            'tr_met_den',
            'tr_metnomu_num',
            'tr_metnomu_den',
            'tr_metnomu_filterhf_num',
            'tr_metnomu_filterhf_den',
        ]

        run_min, run_max = run_range

        for base_region in regions_to_clone:
            regions[f"{base_region}_{label}"] = copy.deepcopy(regions[base_region])
            regions[f"{base_region}_{label}"].append(f"cut_{label}")

    return regions
