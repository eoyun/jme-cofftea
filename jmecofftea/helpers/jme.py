import numpy as np

from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector

from jmecofftea.helpers.paths import jmecofftea_path

def get_jme_correctors(jecs_tag):
    """
    Get jet corrector object having L1L2L3 corrections for the given JEC tag. 
    """
    ext = extractor()
    ext.add_weight_sets([
        f"* * {jmecofftea_path(f'data/jme/{jecs_tag}_L1FastJet_AK4PFPuppi.txt')}",
        f"* * {jmecofftea_path(f'data/jme/{jecs_tag}_L2Relative_AK4PFPuppi.txt')}",
        f"* * {jmecofftea_path(f'data/jme/{jecs_tag}_L2Residual_AK4PFPuppi.txt')}",
        f"* * {jmecofftea_path(f'data/jme/{jecs_tag}_L3Absolute_AK4PFPuppi.txt')}",
        f"* * {jmecofftea_path(f'data/jme/{jecs_tag}_L2L3Residual_AK4PFPuppi.txt')}",
    ])
    ext.finalize()
    
    evaluator = ext.make_evaluator()

    # Define correctors for different levels of JEC. These will be used to
    # correct the raw jet pt (AK4 PF PUPPI jets) from NanoAOD.
    correctors = {}

    correctors["L1L2L3"] = FactorizedJetCorrector(
        Winter22Run3_RunC_V2_DATA_L1FastJet_AK4PFPuppi  = evaluator[f"{jecs_tag}_L1FastJet_AK4PFPuppi"],
        Winter22Run3_RunC_V2_DATA_L2Relative_AK4PFPuppi = evaluator[f"{jecs_tag}_L2Relative_AK4PFPuppi"],
        Winter22Run3_RunC_V2_DATA_L3Absolute_AK4PFPuppi = evaluator[f"{jecs_tag}_L3Absolute_AK4PFPuppi"],
    )

    correctors["L2L3Res"] = FactorizedJetCorrector(
        Winter22Run3_RunC_V2_DATA_L2L3Residual_AK4PFPuppi = evaluator[f"{jecs_tag}_L2L3Residual_AK4PFPuppi"]
    )

    return correctors


def propagate_jecs_to_met(met_pt_orig, met_phi_orig, ak4_init_p4, ak4_corrected_p4):
    """
    Apply type-1 correction on MET using the corrected AK4 jets.
    """
    # Compute uncorrected x and y components of MET
    met_px_orig = met_pt_orig * np.cos(met_phi_orig)
    met_py_orig = met_pt_orig * np.sin(met_phi_orig)

    # Propagate the JECs to x and y components of MET
    met_px_corrected = met_px_orig - (ak4_init_p4.x - ak4_corrected_p4.x).sum()
    met_py_corrected = met_py_orig - (ak4_init_p4.y - ak4_corrected_p4.y).sum()

    # Get the corrected MET pt and MET phi
    met_pt_corrected = np.hypot(met_px_corrected, met_py_corrected)
    met_phi_corrected = np.arctan2(met_py_corrected, met_px_corrected)

    return met_pt_corrected, met_phi_corrected