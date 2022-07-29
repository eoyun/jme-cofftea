import copy
import coffea.processor as processor
import re
import numpy as np

from bucoffea.hlt.definitions import hlt_accumulator, hlt_regions, setup_candidates

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

        # Implement selections
        selection = processor.PackedSelection()

        # Fill histograms
        output = self.accumulator.identity()

        # Return the output accumulator once the histograms are filled
        return output

    def postprocess(self, accumulator):
        return accumulator

