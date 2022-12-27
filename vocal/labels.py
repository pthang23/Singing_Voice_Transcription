import abc
import numpy as np

import os
import sys
MODULE_PATH = os.path.abspath(f"{os.path.split(__file__)[0]}/..")
if sys.path[0] != MODULE_PATH: sys.path.insert(0, MODULE_PATH)

from constants import datasets as dset


class BaseLabelExtraction(metaclass=abc.ABCMeta):
    """Base class for extract label information"""

    @classmethod
    @abc.abstractmethod
    def load_label(cls, label_path):  # -> list[Label]
        """Load the label file and parse information into ``Label`` class"""
        raise NotImplementedError

    @classmethod
    def extract_label(cls, label_path, t_unit=0.02):
        """Extract SDT label"""

        label_list = cls.load_label(label_path)

        max_sec = max([ll.end_time for ll in label_list])
        num_frm = int(max_sec / t_unit) + 10  # Reserve additional 10 frames

        sdt_label = np.zeros((num_frm, 6))
        frm_per_sec = round(1 / t_unit)
        clip = lambda v: np.clip(v, 0, num_frm - 1)
        for label in label_list:
            act_range = range(
                round(label.start_time*frm_per_sec), round(label.end_time*frm_per_sec)  # noqa: E226
            )
            on_range = range(
                round(label.start_time*frm_per_sec - 2), round(label.start_time*frm_per_sec + 4)  # noqa: E226
            )
            off_range = range(
                round(label.end_time*frm_per_sec - 2), round(label.end_time*frm_per_sec + 4)  # noqa: E226
            )
            if len(act_range) == 0:
                continue

            sdt_label[clip(act_range), 0] = 1  # activation
            sdt_label[clip(on_range), 2] = 1  # onset
            sdt_label[clip(off_range), 4] = 1  # offset

        sdt_label[:, 1] = 1 - sdt_label[:, 0]
        sdt_label[:, 3] = 1 - sdt_label[:, 2]
        sdt_label[:, 5] = 1 - sdt_label[:, 4]
        return sdt_label


class VocalAlignLabelExtraction(BaseLabelExtraction):
    """Label extraction for vocal-semi datasets"""
    @classmethod
    def load_label(cls, label_path):
        return dset.VocalAlignStructure.load_label(label_path)


class UnlabeledLabelExtraction(BaseLabelExtraction):
    """Label extraction for unlabeled datasets"""
    @classmethod
    def load_label(cls, label_path):
        return dset.UnlabeledStructure.load_label(label_path)