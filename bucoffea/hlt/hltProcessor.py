import copy
import coffea.processor as processor
import re
import numpy as np
from dynaconf import settings as cfg

from coffea.lumi_tools import LumiMask

from bucoffea.hlt.definitions import hlt_accumulator, hlt_regions, setup_candidates
from bucoffea.helpers import bucoffea_path, recoil, mask_and
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
            cfg.ENV_FOR_DYNACONF = f"era{self._year}"
        else:
            cfg.ENV_FOR_DYNACONF = f"default"
        cfg.reload()

    def process(self, df):
        if not df.size:
            return self.accumulator.identity()
        dataset = df['dataset']

        self._configure(df)

        met_pt, met_phi, ak4, muons, electrons, photons = setup_candidates(df, cfg)

        #accumulate events that fail metmht trigger in W(munu) selection
        #items['selected_events'] = processor.defaultdict_accumulator(list)

        # Implement selections
        selection = processor.PackedSelection()

        # Create mask for events with good lumis (using the golden JSON)
        json = bucoffea_path("data/json/Cert_Collisions2022_355100_356175_Golden.json")
        lumi_mask = LumiMask(json)(df['run'], df['luminosityBlock'])
        selection.add('lumi_mask', lumi_mask)
        
        #index of leading (highest energy Jet)
        leadak4_index = ak4.pt.argmax()

        #require that lead jet > 40 GeV and |eta| < 4.5
        leadak4_pt_eta = (ak4.pt.max() > 40) & (ak4.abseta[leadak4_index] < 4.5)
        selection.add('leadak4_pt_eta', leadak4_pt_eta.any())
        
        #require that lead jet has loose ID
        selection.add('leadak4_id', (ak4.looseId[leadak4_index].any()))
        
        #require that mftmht_trig and mftmht_clean_trig are triggered
        selection.add('mftmht_trig', df['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight'])
        selection.add('mftmht_clean_trig', df['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF'])
        selection.add('fail_metmht_trig', df['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight'] == 0)
        selection.add('HLT_IsoMu27', df['HLT_IsoMu27'])

        df['is_tight_muon'] = (muons.iso < cfg.MUON.CUTS.TIGHT.ISO) \
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

        #Single Electron CR
        trig_ele = mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE_BACKUP) | mask_or(df, cfg.TRIGGERS.ELECTRON.SINGLE)
        selection.add('trig_ele', trig_ele)
        selection.add('one_electron', electrons.counts==1)
        selection.add('at_least_one_tight_el', df['is_tight_electron'].any())
        selection.add('hlt_ele', df['HLT_Ele35_WPTight_Gsf'] | df['HLT_Ele115_CaloIdVT_GsfTrkIdT'])
        
        #Recoil
        df['recoil_pt'], df['recoil_phi'] = recoil(met_pt, met_phi, electrons, muons, photons)
        recoil_pt, recoil_phi = df['recoil_pt'], df['recoil_phi']i
        selection.add('recoil>250', recoil_pt > 250)

        #Electron veto
        selection.add('veto_ele', electrons.counts==0)

        #Photon Veto
        selection.add('veto_pho', photons.counts==0)

        #MET filters
        selection.add('filt_met', mask_and(df, cfg.FILTERS.DATA)) 
        df["dPFCaloCR"] = (met_pt - df["CaloMET_pt"]) / recoil_pt
        selection.add('calo_diff', np.abs(df["dPFCaloCR"]) < 0.5)

        # Fill histograms
        output = self.accumulator.identity()

        regions = hlt_regions()
	
        for region, cuts in regions.items():

            mask = selection.all(*cuts)
            #keep track of event numbers that pass this mask
            #output['selected_events'][region] += list(df['event'][mask])

            def ezfill(name, **kwargs):
                """Helper function to make filling easier."""

                if not ('dataset' in kwargs):
                    kwargs['dataset'] = dataset

                output[name].fill(region=region, **kwargs)

            #w_leadak4 = weight_shape(ak4[leadak4_index].eta[mask], region_weights.partial_weight(exclude=exclude)[mask]
            #if filling histogram with simulated data can weight the data in ezill with parameter [weight]
            w_leadak4 = 1
            #ezfill('ak4_eta0',   jeteta=ak4[leadak4_index].eta[mask].flatten())
            #ezfill('ak4_pt0',    jetpt=ak4[leadak4_index].pt[mask].flatten())
            #ezfill('ak4_phi0',   jetphi=ak4[leadak4_index].phi[mask].flatten())
            #ezfill('dimu_mass', dimumass=dimuons.mass[mask].flatten())
            ezfill('trigger_turnon', turnon=recoil_pt[mask])                      
            #ezfill('met', MET=met_pt[mask])

        #with open('fail10.txt', 'a') as f:
            #for event in output['selected_events'][region]:
                #f.write('\n' + str(event))

        # Return the output accumulator once the histograms are filled
        return output

    def postprocess(self, accumulator):
        return accumulator
