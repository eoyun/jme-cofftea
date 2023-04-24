from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector

from jmecofftea.helpers.paths import bucoffea_path

def get_jme_correctors():
    ext = extractor()
    ext.add_weight_sets([
        f"* * {bucoffea_path('data/jme/Winter22Run3_RunC_V2_DATA_L1FastJet_AK4PFPuppi.txt')}",
        f"* * {bucoffea_path('data/jme/Winter22Run3_RunC_V2_DATA_L2Relative_AK4PFPuppi.txt')}",
        f"* * {bucoffea_path('data/jme/Winter22Run3_RunC_V2_DATA_L2Residual_AK4PFPuppi.txt')}",
        f"* * {bucoffea_path('data/jme/Winter22Run3_RunC_V2_DATA_L3Absolute_AK4PFPuppi.txt')}",
        f"* * {bucoffea_path('data/jme/Winter22Run3_RunC_V2_DATA_L2L3Residual_AK4PFPuppi.txt')}",
    ])
    ext.finalize()
    
    evaluator = ext.make_evaluator()

    # Define pre and post-HCAL update JEC correctors.
    correctors = {}

    correctors["L1L2L3"] = FactorizedJetCorrector(
        Winter22Run3_RunC_V2_DATA_L1FastJet_AK4PFPuppi = evaluator["Winter22Run3_RunC_V2_DATA_L1FastJet_AK4PFPuppi"],
        Winter22Run3_RunC_V2_DATA_L2Relative_AK4PFPuppi = evaluator["Winter22Run3_RunC_V2_DATA_L2Relative_AK4PFPuppi"],
        Winter22Run3_RunC_V2_DATA_L3Absolute_AK4PFPuppi = evaluator["Winter22Run3_RunC_V2_DATA_L3Absolute_AK4PFPuppi"],
    )

    correctors["L2L3Res"] = FactorizedJetCorrector(
        Winter22Run3_RunC_V2_DATA_L2L3Residual_AK4PFPuppi = evaluator["Winter22Run3_RunC_V2_DATA_L2L3Residual_AK4PFPuppi"]
    )

    correctors["L2Res"] = FactorizedJetCorrector(
        Winter22Run3_RunC_V2_DATA_L2Residual_AK4PFPuppi = evaluator["Winter22Run3_RunC_V2_DATA_L2Residual_AK4PFPuppi"]
    )

    return correctors