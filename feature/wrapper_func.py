import numpy as np
from PIL import Image

import os
import sys
MODULE_PATH = os.path.abspath(f"{os.path.split(__file__)[0]}/..")
if sys.path[0] != MODULE_PATH: sys.path.insert(0, MODULE_PATH)

from feature import cfp
from feature import hcfp


def extract_cfp_feature(audio_path, harmonic=False, harmonic_num=6, **kwargs):
    """Wrapper of CFP/HCFP feature extraction"""
    
    if harmonic:
        spec, gcos, ceps, _ = hcfp.extract_hcfp(audio_path, harmonic_num=harmonic_num, **kwargs)
        return np.dstack([spec, gcos, ceps])

    z, spec, gcos, ceps, _ = cfp.extract_cfp(audio_path, **kwargs)
    return np.dstack([z.T, spec.T, gcos.T, ceps.T])