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

class hltProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = hlt_accumulator()

    @property
    def accumulator(self):
        return self._accumulator

    def _configure(self, df=None):
        cfg.DYNACONF_WORKS="merge_configs"
        cfg.MERGE_ENABLED_FOR_DYNACONF = True
        cfg.SETTINGS_FILE_FOR_DYNACONF = bucoffea_path("config/vbfhinv.yaml")

        # Reload config based on year
        if df:
            dataset = df['dataset']
            self._year = extract_year(dataset)
            df["year"] = self._year
            #cfg.ENV_FOR_DYNACONF = f"era{self._year}"
            cfg.ENV_FOR_DYNACONF = "era2017"
        else:
            cfg.ENV_FOR_DYNACONF = f"default"
        cfg.reload()

    def process(self, df):
        if not df.size:
            return self.accumulator.identity()
        dataset = df['dataset']

        self._configure(df)

        met_pt, met_phi, ak4, muons, electrons, taus, photons = setup_candidates(df, cfg)

        # Implement selections
        selection = processor.PackedSelection()

        # Create mask for events with good lumis (using the golden JSON)
        json = bucoffea_path("data/json/Cert_Collisions2022_355100_357900_Golden.json") #era C+D json
        lumi_mask = LumiMask(json)(df['run'], df['luminosityBlock'])
        selection.add('lumi_mask', lumi_mask)
        
        #index of leading (highest energy Jet)
        leadak4_index = ak4.pt.argmax()

        #require that lead jet > 40 GeV and |eta| < 4.5
        leadak4_pt_eta = (ak4.pt.max() > 40) & (ak4.abseta[leadak4_index] < 4.5)
        selection.add('leadak4_pt_eta', leadak4_pt_eta.any())
        
        ht = ak4[ak4.pt>20].pt.sum()

        # Tight ID on leading AK4 jet
        selection.add('leadak4_id', (ak4.looseId[leadak4_index].any()))
        
        # Requirement on hadronic energy fractions of the jet (for jets within tracker range)
        has_track = ak4[leadak4_index].abseta <= 2.5
        energy_frac_good = has_track * ((ak4[leadak4_index].chf > 0.1) & (ak4[leadak4_index].nhf < 0.8)) + ~has_track
        selection.add('leadak4_energy_frac', energy_frac_good.any())

        # Trigger requirements
        selection.add('HLT_PFMETNoMu120', df['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight'])
        selection.add('HLT_PFMETNoMu120_FilterHF', df['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF'])
        selection.add('HLT_IsoMu27', df['HLT_IsoMu27'])

        selection.add('HLT_PFJet500', df['HLT_PFJet500'])
        selection.add('HLT_PFHT1050', df['HLT_PFHT1050'])

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

        # Dimuon CR
        leadmuon_index=muons.pt.argmax()
        selection.add('at_least_one_tight_mu', df['is_tight_muon'].any())
        selection.add('dimuon_mass', ((dimuons.mass > cfg.SELECTION.CONTROL.DOUBLEMU.MASS.MIN) \
                                    & (dimuons.mass < cfg.SELECTION.CONTROL.DOUBLEMU.MASS.MAX)).any())
        selection.add('dimuon_charge', (dimuon_charge==0).any())
        selection.add('two_muons', muons.counts==2)
        
        #Single Muon CR
        selection.add('one_muon', muons.counts==1)
        selection.add('muon_pt>30', muons.pt.max() > 30)

        #Recoil
        df['recoil_pt'], df['recoil_phi'] = metnomu(met_pt, met_phi, muons)
        selection.add('recoil>250', df['recoil_pt'] > 250)

        run_mask = df['run'] < 356800
        selection.add('run_bf_356800', run_mask)
        selection.add('run_af_356800', ~run_mask)

        #Electron veto
        selection.add('veto_ele', electrons.counts==0)

        #Tau veto
        selection.add('veto_tau', taus.counts==0)

        #Photon Veto
        selection.add('veto_pho', photons.counts==0)

        #MET filters
        selection.add('filt_met', mask_and(df, cfg.FILTERS.DATA)) 
        df["dPFCaloCR"] = (met_pt - df["CaloMET_pt"]) / df['recoil_pt']
        selection.add('calo_diff', np.abs(df["dPFCaloCR"]) < 0.5)

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

        regions = hlt_regions()
	
        for region, cuts in regions.items():

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

            if 'fail_jet500' in region:
                ezfill('ak4_chf0',     frac=ak4[leadak4_index].chf[mask].flatten())
                ezfill('ak4_nhf0',     frac=ak4[leadak4_index].nhf[mask].flatten())
                ezfill('ak4_mufrac0',  frac=ak4[leadak4_index].mufrac[mask].flatten())

            # with open('eventlist.txt', 'a') as f:
            #     for event in output['selected_events'][region]:
            #         f.write('\n' + str(event))

        # Return the output accumulator once the histograms are filled
        return output

    def postprocess(self, accumulator):
        return accumulator
