import coffea.processor as processor
import numpy as np
from coffea import hist

from bucoffea.helpers import min_dphi_jet_met, dphi
from bucoffea.helpers.dataset import (is_lo_g, is_lo_g_ewk, is_lo_w, is_lo_z, is_lo_w_ewk, is_lo_z_ewk,
                                      is_nlo_g,is_nlo_g_ewk, is_nlo_w, is_nlo_z)
from bucoffea.helpers.gen import (fill_gen_v_info, islep, isnu,
                                  setup_dressed_gen_candidates,
                                  find_gen_dilepton,
                                  setup_gen_candidates,
                                  setup_lhe_cleaned_genjets)

from .lheVProcessor import vbf_selection

Hist = hist.Hist
Bin = hist.Bin
Cat = hist.Cat

def is_ele_or_mu(pdg):
    abspdg = np.abs(pdg)
    return (abspdg==11) | (abspdg==13)

class genVbfProcessor(processor.ProcessorABC):
    def __init__(self):
        # Histogram setup
        dataset_ax = Cat("dataset", "Primary dataset")
        region_ax = Cat("region", "Selection region")

        mjj_ax = Bin("mjj",r"$m(jj)$ (GeV)", 75, 0, 7500)
        jet_pt_ax = Bin("jetpt",r"$p_{T}^{j}$ (GeV)", 50, 0, 2000)
        jet_eta_ax = Bin("jeteta", r"$\eta$", 50, -5, 5)

        dilepton_mass_ax = Bin("dilepton_mass", r"$M(\ell\ell)$ (GeV)", 100,50,150)

        items = {}
        items["gen_mjj"] = Hist("Counts", dataset_ax, region_ax, mjj_ax)
        
        items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
        items["ak4_pt1"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
        items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
        items["ak4_eta1"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
        
        items["dilepton_mass"] = Hist("Counts", dataset_ax, region_ax, dilepton_mass_ax)

        items['sumw'] = processor.defaultdict_accumulator(float)
        items['sumw2'] = processor.defaultdict_accumulator(float)

        self._accumulator = processor.dict_accumulator(items)

        self.regions = {}

        common_sel = [
            'two_jets',
            'leadak4_pt_eta',
            'trailak4_pt_eta',
            'hemisphere',
            'mindphijr',
            'detajj',
            'dphijj'
        ]

        self.regions['cr_z_vbf'] = common_sel + ['z_selection']
        self.regions['cr_w_vbf'] = common_sel + ['w_selection']

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
        dataset = df['dataset']

        genjets = setup_lhe_cleaned_genjets(df)
        dijet = genjets[:,:2].distincts()
        mjj = dijet.mass.max()

        gen = setup_gen_candidates(df)
        neutrinos = gen[(gen.status==1) & isnu(gen.pdg) & ((gen.flag&1)==1)]

        if is_lo_w(dataset) or is_nlo_w(dataset) or is_lo_w_ewk(dataset) or is_lo_z(dataset) or is_nlo_z(dataset) or is_lo_z_ewk(dataset):
            dressed = setup_dressed_gen_candidates(df)
            fill_gen_v_info(df, gen, dressed)

        # Z->ll (l=e,mu) decays
        if is_lo_z(dataset) or is_nlo_z(dataset) or is_lo_z_ewk(dataset):
            target = 91
            dilep_dress = find_gen_dilepton(dressed, 0)
            dilep_dress = dilep_dress[np.abs(dilep_dress.mass-target).argmin()]

        # W->lv (l=e,mu) decays
        elif is_lo_w(dataset) or is_nlo_w(dataset) or is_lo_w_ewk(dataset):
            target = 81

            dilep_dress = dressed.cross(neutrinos)
            dilep_dress = dilep_dress[ (
                                      ((np.abs(dilep_dress.i0.pdg)==11) & (np.abs(dilep_dress.i1.pdg)==12) ) \
                                    | ((np.abs(dilep_dress.i0.pdg)==13) & (np.abs(dilep_dress.i1.pdg)==14) ) \
                                    ) & (dilep_dress.i0.pdg*dilep_dress.i1.pdg< 0)
                                    ]
            dilep_dress = dilep_dress[np.abs(dilep_dress.mass-target).argmin()]

        tag = 'combined'
        
        selection = vbf_selection(df[f'gen_v_phi_{tag}'], dijet, genjets)
        nominal = df['Generator_weight']
        
        # Selections for the W and Z processes, using dressed leptons

        wmask = (dilep_dress.counts == 1) & \
                    ( ((np.abs(dilep_dress.i0.pdg)==11) & (np.abs(dilep_dress.i1.pdg)==12) ) \
                    | ((np.abs(dilep_dress.i0.pdg)==13) & (np.abs(dilep_dress.i1.pdg)==14) ) \
                    ) & (dilep_dress.i0.pdg * dilep_dress.i1.pdg < 0)

        zmask = (dilep_dress.counts == 1) & \
            (np.abs(dilep_dress.mass.max() - 91) < 30) & \
            (np.abs(dilep_dress.i0.pdg) == np.abs(dilep_dress.i1.pdg)) & \
            (dilep_dress.i0.pdg * dilep_dress.i1.pdg < 0)

        selection.add('w_selection', wmask.any())
        selection.add('z_selection', zmask.any())

        for region, cuts in self.regions.items():
            mask = selection.all(*cuts)

            def ezfill(name, **kwargs):
                """Helper function to make filling easier."""
                output[name].fill(
                                  dataset=dataset,
                                  region=region,
                                  **kwargs
                                  )

            ezfill('gen_mjj',     mjj=mjj[mask],     weight=nominal[mask])
            
            ezfill('ak4_pt0',     jetpt=dijet.i0.pt[mask].flatten(),     weight=nominal[mask])
            ezfill('ak4_pt1',     jetpt=dijet.i1.pt[mask].flatten(),     weight=nominal[mask])
            
            ezfill('ak4_eta0',    jeteta=dijet.i0.eta[mask].flatten(),     weight=nominal[mask])
            ezfill('ak4_eta1',    jeteta=dijet.i1.eta[mask].flatten(),     weight=nominal[mask])

            if region == 'cr_z_vbf':
                ezfill('dilepton_mass', dilepton_mass=dilep_dress.mass[mask].max(),    weight=nominal[mask])

        # Keep track of weight sum
        output['sumw'][dataset] +=  df['genEventSumw']
        output['sumw2'][dataset] +=  df['genEventSumw2']
        return output

    def postprocess(self, accumulator):
        return accumulator

