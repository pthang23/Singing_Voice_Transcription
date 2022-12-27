import abc
import numpy as np

import os
import sys
MODULE_PATH = os.path.abspath(f"{os.path.split(__file__)[0]}/..")
if sys.path[0] != MODULE_PATH: sys.path.insert(0, MODULE_PATH)

from constants import datasets as dset
from constants.midi import LOWEST_MIDI_NOTE


class BaseLabelExtraction(metaclass=abc.ABCMeta):
    """Base class for extract label information"""

    @classmethod
    @abc.abstractmethod
    def load_label(cls, label_path): 
        """Load the label file and parse information into ``Label`` class"""
        raise NotImplementedError

    @classmethod
    def extract_label(cls, label_path, t_unit=0.02):
        labels = cls.load_label(label_path)
        fs = round(1 / t_unit)

        max_time = max(label.end_time for label in labels)
        output = np.zeros((round(max_time * fs), 352))
        for label in labels:
            start_idx = round(label.start_time * fs)
            end_idx = round(label.end_time * fs)
            pitch = round((label.note - LOWEST_MIDI_NOTE) * 4)
            output[start_idx:end_idx, pitch] = 1
        return output


class VocalContourlabelExtraction(BaseLabelExtraction):
    """vocal contour datasets label extraction class"""
    @classmethod
    def load_label(cls, label_path):
        return dset.VocalContourStructure.load_label(label_path)
