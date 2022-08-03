import copy
import coffea.processor as processor
import re
import numpy as np

from coffea.lumi_tools import LumiMask

from bucoffea.hlt.definitions import hlt_accumulator, hlt_regions, setup_candidates
from bucoffea.helpers.paths import bucoffea_path

class hltProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = hlt_accumulator()

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        if not df.size:
            return self.accumulator.identity()
        dataset = df['dataset']

        # Create mask for events with good lumis (using the golden JSON)
        json = bucoffea_path("data/json/Cert_Collisions2022_355100_356175_Golden.json")
        lumi_mask = LumiMask(json)(df['run'], df['luminosityBlock'])

        # Implement selections
        selection = processor.PackedSelection()

        # Fill histograms
        output = self.accumulator.identity()

        # Return the output accumulator once the histograms are filled
        return output

    def postprocess(self, accumulator):
        return accumulator

