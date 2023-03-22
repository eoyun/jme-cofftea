import copy
import coffea.processor as processor
import re
import numpy as np
from dynaconf import settings as cfg

from coffea.lumi_tools import LumiMask

from bucoffea.hlt.definitions import hlt_accumulator, hlt_regions, setup_candidates
from bucoffea.helpers import bucoffea_path, recoil, metnomu, mask_and, mask_or, object_overlap
from coffea.lumi_tools import LumiMask
from bucoffea.helpers.dataset import extract_year
from bucoffea.helpers.paths import bucoffea_path
from bucoffea.helpers.jme import get_jme_correctors

class hltProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = hlt_accumulator()

    @property
    def accumulator(self):
        return self._accumulator

    def _configure(self, df=None):
        cfg.DYNACONF_WORKS="merge_configs"
        cfg.MERGE_ENABLED_FOR_DYNACONF = True
        cfg.SETTINGS_FILE_FOR_DYNACONF = bucoffea_path("config/hlt.yaml")

        # Reload config based on year
        if df:
            dataset = df['dataset']
            self._year = extract_year(dataset)
            df["year"] = self._year
            cfg.ENV_FOR_DYNACONF = "default"
        else:
            cfg.ENV_FOR_DYNACONF = "default"
        cfg.reload()

    def process(self, df):
        if not df.size:
            return self.accumulator.identity()
        dataset = df['dataset']

        self._configure(df)

        met_pt, met_phi, ak4, muons, electrons, taus, photons = setup_candidates(df, cfg)

        # JECs
        jme_correctors = get_jme_correctors()

        rho = ak4.pt.ones_like() * df["Rho_fixedGridRhoFastjetAll"]

        # Apply the proper JECs, pre or post HCAL for data
        # The getCorrection() call below changes ak4.pt IN PLACE (dangerous!)
        if re.match(".*2022[CDE]", df["dataset"]):
            jme_correctors["preHCAL"].getCorrection(JetPt=ak4.pt, JetEta=ak4.eta, Rho=rho, JetA=ak4.area)
        elif re.match(".*2022[FG]", df["dataset"]):
            jme_correctors["postHCAL"].getCorrection(JetPt=ak4.pt, JetEta=ak4.eta, Rho=rho, JetA=ak4.area)

        # Implement selections
        selection = processor.PackedSelection()

        pass_all = np.ones(df.size)==1
        selection.add('inclusive', pass_all)

        # Create mask for events with good lumis (using the golden JSON)
        if df["year"] == 2022:
            json = bucoffea_path("data/json/Cert_Collisions2022_355100_362760_Golden.json")
        elif df["year"] == 2018:
            json = bucoffea_path("data/json/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt")
        else:
            raise RuntimeError(f"Golden JSON filtering not implemented for year: {df['year']}")

        lumi_mask = LumiMask(json)(df['run'], df['luminosityBlock'])
        selection.add('lumi_mask', lumi_mask)

        # Requirements on the leading jet
        leadak4_index = ak4.pt.argmax()
        leadak4_pt_eta = (ak4.pt.max() > cfg.AK4.PT) & (ak4.abseta[leadak4_index] < cfg.AK4.ABSETA)
        selection.add('leadak4_pt_eta', leadak4_pt_eta.any())
        
        selection.add("leadak4_pt_cut", ak4.pt.max() > 300)

        # Compute HT, follow the computation recipe of HLT_PFHT1050
        ht = ak4[(ak4.pt > cfg.HT.JETPT) & (ak4.abseta < cfg.HT.ABSETA)].pt.sum()

        # Tight ID on leading AK4 jet
        selection.add('leadak4_id', (ak4.tightIdLepVeto[leadak4_index].any()))
        
        # Requirement on hadronic energy fractions of the jet (for jets within tracker range)
        has_track = ak4[leadak4_index].abseta <= 2.5
        energy_frac_good = has_track * ((ak4[leadak4_index].chf > cfg.AK4.CHF) & (ak4[leadak4_index].nhf < cfg.AK4.NHF)) + ~has_track
        selection.add('leadak4_energy_frac', energy_frac_good.any())

        # Selection for leading jet - whether it is within the water leak region or not
        leading_ak4_in_water_leak = ((ak4[leadak4_index].eta > 1.4) & (ak4[leadak4_index].eta < 2.2) & \
            (ak4[leadak4_index].phi > 1.8) & (ak4[leadak4_index].phi < 2.6))

        selection.add('ak4_not_in_water_leak', ~leading_ak4_in_water_leak.any())
        selection.add('ak4_in_water_leak', leading_ak4_in_water_leak.any())

        # Trigger requirements
        selection.add('HLT_PFMET120', df['HLT_PFMET120_PFMHT120_IDTight'])
        selection.add('HLT_PFMETNoMu120', df['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight'])
        selection.add('HLT_IsoMu27', df['HLT_IsoMu27'])
        
        # L1 seed requirements
        selection.add('L1_ETMHF100', df['L1_ETMHF100'])

        # HF-filtered METNoMu120 trigger - available starting from 2022 data taking
        if df['year'] == 2022:        
            selection.add('HLT_PFMETNoMu110_FilterHF', df['HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF'])
            selection.add('HLT_PFMETNoMu120_FilterHF', df['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF'])
            selection.add('HLT_PFMETNoMu130_FilterHF', df['HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_FilterHF'])
            selection.add('HLT_PFMETNoMu140_FilterHF', df['HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_FilterHF'])
        else:
            selection.add('HLT_PFMETNoMu110_FilterHF', ~pass_all)
            selection.add('HLT_PFMETNoMu120_FilterHF', ~pass_all)
            selection.add('HLT_PFMETNoMu130_FilterHF', ~pass_all)
            selection.add('HLT_PFMETNoMu140_FilterHF', ~pass_all)

        # Jet+HT triggers
        selection.add('HLT_PFJet500', df['HLT_PFJet500'])
        selection.add('HLT_PFHT1050', df['HLT_PFHT1050'])

        # L1 requirement for HT1050 (only for 2022 era datasets)
        if df['year'] == 2022 and cfg.STUDIES.L1_TURNON:
            l1_seeds = [
                'L1_HTT120er',
                'L1_HTT160er',
                'L1_HTT200er',
                'L1_HTT255er',
                'L1_HTT280er',
                'L1_HTT280er_QuadJet_70_55_40_35_er2p5',
                'L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3',
                'L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3',
                'L1_HTT320er',
                'L1_HTT360er',
                'L1_ETT2000',
                'L1_HTT400er',
                'L1_HTT450er',
            ]
            
            l1_pass_ht1050 = ~pass_all

            for seed in l1_seeds:
                l1_pass_ht1050 |= df[seed]
        
            selection.add('L1_pass_HT1050', l1_pass_ht1050)
        
        else:
            selection.add('L1_pass_HT1050', ~pass_all)

        # For events failing PFJet500 with a high leading jet pt
        selection.add('leadak4_high_pt', (ak4.pt.max() > 600))
        selection.add('fail_HLT_PFJet500', ~df['HLT_PFJet500'])

        # HF jet cuts
        high_pt_ak4 = ak4[ak4.pt>80]
        seta_minus_sphi = high_pt_ak4.setaeta - high_pt_ak4.sphiphi

        selection.add('seta_minus_sphi', (seta_minus_sphi < 0.02).all())
        selection.add('central_strip_size', (high_pt_ak4.hfcentralstripsize < 3).all())

        df['is_tight_muon'] = muons.tightId \
                      & (muons.iso < cfg.MUON.CUTS.TIGHT.ISO) \
                      & (muons.pt > cfg.MUON.CUTS.TIGHT.PT) \
                      & (muons.abseta < cfg.MUON.CUTS.TIGHT.ETA)

        dimuons = muons.distincts()
        dimuon_charge = dimuons.i0['charge'] + dimuons.i1['charge']

        # Single Muon CR
        leadmuon_index=muons.pt.argmax()
        selection.add('one_muon', muons.counts==1)
        selection.add('muon_pt>30', muons.pt.max() > cfg.MUON.CUTS.TIGHT.PT)
        selection.add('at_least_one_tight_mu', df['is_tight_muon'].any())

        # Recoil
        df['recoil_pt'], df['recoil_phi'] = metnomu(met_pt, met_phi, muons)
        selection.add('recoil>250', df['recoil_pt'] > cfg.RECOIL.PT)

        for run_num in cfg.RUN.COMPARE:
            run_mask = df['run'] < run_num
            selection.add(f'run_bf_{run_num}', run_mask)
            selection.add(f'run_af_{run_num}', ~run_mask)

        #Electron veto
        selection.add('veto_ele', electrons.counts==0)

        #Tau veto
        selection.add('veto_tau', taus.counts==0)

        #Photon Veto
        selection.add('veto_pho', photons.counts==0)

        # MET filters
        selection.add('filt_met', mask_and(df, cfg.FILTERS.DATA)) 
        
        # Delta(PFMET, CaloMET)
        df["dPFCaloCR"] = (met_pt - df["CaloMET_pt"]) / df['recoil_pt']
        selection.add('calo_diff', np.abs(df["dPFCaloCR"]) < cfg.DPFCALO)

        # Selection to get high PU (=60) fill
        selection.add("pu60_fill", (df["run"] >= 362613) & (df["run"] <= 362618))

        # Fill histograms
        output = self.accumulator.identity()

        # Save kinematics for specific events
        if cfg.RUN.KINEMATICS.SAVE:
            for event in cfg.RUN.KINEMATICS.EVENTS:
                event_mask = df['event'] == event

                if not event_mask.any():
                    continue

                output['kinematics']['event'] += [event]

                output['kinematics']['ak4_pt0'] += [ak4[leadak4_index][event_mask].pt]            
                output['kinematics']['ak4_eta0'] += [ak4[leadak4_index][event_mask].eta]            
                output['kinematics']['ak4_phi0'] += [ak4[leadak4_index][event_mask].phi]            
                output['kinematics']['ak4_tightId0'] += [ak4[leadak4_index][event_mask].looseId]            
                
                output['kinematics']['ak4_nhf0'] += [ak4[leadak4_index][event_mask].nhf]
                output['kinematics']['ak4_nef0'] += [ak4[leadak4_index][event_mask].nef]
                output['kinematics']['ak4_chf0'] += [ak4[leadak4_index][event_mask].chf]
                output['kinematics']['ak4_cef0'] += [ak4[leadak4_index][event_mask].cef]
                output['kinematics']['ak4_mufrac0'] += [ak4[leadak4_index][event_mask].mufrac]

                output['kinematics']['mu_pt0'] += [muons[leadmuon_index][event_mask].pt]
                output['kinematics']['mu_eta0'] += [muons[leadmuon_index][event_mask].eta]
                output['kinematics']['mu_phi0'] += [muons[leadmuon_index][event_mask].phi]
                output['kinematics']['mu_tightId0'] += [muons[leadmuon_index][event_mask].tightId]

        regions = hlt_regions(cfg)
	
        for region, cuts in regions.items():
            # Only run on the regions we want to run
            if not re.match(cfg.RUN.REGIONS, region):
                continue

            mask = selection.all(*cuts)

            if cfg.RUN.SAVE.PASSING:
                output['selected_runs'][region] += list(df['run'][mask])
                output['selected_lumis'][region] += list(df['luminosityBlock'][mask])
                output['selected_events'][region] += list(df['event'][mask])

            def ezfill(name, **kwargs):
                """Helper function to make filling easier."""
                output[name].fill(
                    region=region, 
                    dataset=dataset, 
                    **kwargs
                    )

            ezfill('ak4_eta0',   jeteta=ak4[leadak4_index].eta[mask].flatten())
            ezfill('ak4_phi0',   jetphi=ak4[leadak4_index].phi[mask].flatten())
            ezfill('ak4_pt0',    jetpt=ak4[leadak4_index].pt[mask].flatten())
            ezfill('recoil',     recoil=df['recoil_pt'][mask])                      
            ezfill('met',        met=met_pt[mask])
            ezfill('ht',         ht=ht[mask])

            ezfill('ak4_abseta0_pt0',   jeteta=ak4[leadak4_index].abseta[mask].flatten(), jetpt=ak4[leadak4_index].pt[mask].flatten())

            # PU plots -> Number of vertices vs. MET/METNoMu
            ezfill('met_npv',          met=met_pt[mask],   nvtx=df["PV_npvs"][mask])
            ezfill('met_npvgood',      met=met_pt[mask],   nvtx=df["PV_npvsGood"][mask])
            ezfill('recoil_npv',       recoil=df["recoil_pt"][mask],  nvtx=df["PV_npvs"][mask])
            ezfill('recoil_npvgood',   recoil=df["recoil_pt"][mask],  nvtx=df["PV_npvsGood"][mask])

            pt_ratio = (ak4[mask].pt / ak4[mask].ptnano).flatten()
            ezfill('ak4_pt_jec_over_nano',   ratio=pt_ratio)

            if 'fail_jet500' in region:
                ezfill('ak4_chf0',     frac=ak4[leadak4_index].chf[mask].flatten())
                ezfill('ak4_nhf0',     frac=ak4[leadak4_index].nhf[mask].flatten())
                ezfill('ak4_mufrac0',  frac=ak4[leadak4_index].mufrac[mask].flatten())

        return output

    def postprocess(self, accumulator):
        return accumulator
