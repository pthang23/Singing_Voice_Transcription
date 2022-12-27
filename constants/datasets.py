import sys
import os
import csv
import glob
from os.path import join as jpath
from shutil import copy

import pretty_midi
import numpy as np
import re

MODULE_PATH = os.path.abspath(jpath(os.path.split(__file__)[0], '..'))
if sys.path[0] != MODULE_PATH: sys.path.append(MODULE_PATH)

from base import Label
from utils import get_logger


logger = get_logger("Constant Datasets")


def _get_file_list(dataset_path, dirs, ext):
    files = []
    for _dir in dirs:
        files += glob.glob(os.path.join(dataset_path, _dir, "*" + ext))
    return files


class BaseStructure:
    """Defines the necessary attributes and common functions for each sub-dataset structure class"""

    # Is labeled or unlabeled dataset
    is_labeled = True

    # The URL for downloading the dataset.
    url = None

    # The extension of ground-truth files (e.g. .mid, .csv).
    label_ext = None

    # Record folders that contain trainig wav files
    train_wavs = None

    # Record folders that contain testing wav files
    test_wavs = None

    # Record folders that contains training labels
    train_labels = None

    # Records folders that contains testing labels
    test_labels = None

    @classmethod
    def _get_data_pair(cls, wavs, labels):
        label_path_mapping = {os.path.basename(label): label for label in labels}

        pair = []
        for wav in wavs:
            basename = os.path.basename(wav)
            label_name = cls._name_transform(basename).replace(".wav", cls.label_ext)
            label_path = label_path_mapping[label_name]
            assert os.path.exists(label_path)
            pair.append((wav, label_path))

        return pair
    
    # FIX
    @classmethod
    def _get_unlabeled_data_pair(cls, wavs):
        pair = []
        for wav in wavs:
            pair.append((wav))
        return pair

    @classmethod
    def get_train_data_pair(cls, dataset_path):
        """Get pair of training file and the coressponding label file path."""
        # print(cls.get_train_wavs(dataset_path))
        # print(cls.get_train_labels(dataset_path))
        return cls._get_data_pair(cls.get_train_wavs(dataset_path), cls.get_train_labels(dataset_path)) if cls.is_labeled else cls._get_unlabeled_data_pair(cls.get_train_wavs(dataset_path))

    @classmethod
    def get_test_data_pair(cls, dataset_path):
        """Get pair of testing file and the coressponding label file path."""
        return cls._get_data_pair(cls.get_test_wavs(dataset_path), cls.get_test_labels(dataset_path)) if cls.is_labeled else cls._get_unlabeled_data_pair(cls.get_test_wavs(dataset_path))

    @classmethod
    def _name_transform(cls, basename):
        # Transform the basename of wav file to the corressponding label file name.
        return basename

    @classmethod
    def get_train_wavs(cls, dataset_path):
        """Get list of complete train wav paths"""
        return _get_file_list(dataset_path, cls.train_wavs, ".wav")

    @classmethod
    def get_test_wavs(cls, dataset_path):
        """Get list of complete test wav paths"""
        return _get_file_list(dataset_path, cls.test_wavs, ".wav")

    @classmethod
    def get_train_labels(cls, dataset_path):
        """Get list of complete train label paths"""
        return _get_file_list(dataset_path, cls.train_labels, cls.label_ext)

    @classmethod
    def get_test_labels(cls, dataset_path):
        """Get list of complete test label paths"""
        return _get_file_list(dataset_path, cls.test_labels, cls.label_ext)

    @classmethod
    def load_label(cls, label_path):
        """Load and parse labels for the given label file path"""

        midi = pretty_midi.PrettyMIDI(label_path)
        labels = []
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                label = Label(
                    start_time=note.start,
                    end_time=note.end,
                    note=note.pitch,
                    velocity=note.velocity,
                    instrument=inst.program
                )
                if label.note == -1:
                    continue
                labels.append(label)
        return labels


class VocalAlignStructure(BaseStructure):
    """Constant settings of vocal-semi datasets"""

    url = None

    # Label extension for note-level transcription.
    label_ext = ".csv"

    # Folder to train wavs
    train_wavs = ["audios"]

    # Folder to train labels
    train_labels = ["labels"]

    # Folder to test wavs
    test_wavs = []

    # Folder to test labels
    test_labels = []

    @classmethod
    def load_label(cls, label_path):
        with open(label_path, "r") as lin:
            lines = lin.readlines()

        labels = []
        for line in lines:
            if not re.fullmatch("([\d\s\.]+,){2}[\d\s\.]+(,.+)*", line.strip()):
                continue

            onset, offset, note = [element.strip() for element in line.split(",")[:3]]

            labels.append(Label(
                start_time=float(onset),
                end_time=float(offset),
                note=round(float(note))
            ))
        return labels


class VocalContourStructure(BaseStructure):
    """Constant settings of vocal-semi datasets"""

    url = None

    # Label extension for note-level transcription.
    label_ext = ".csv"

    # Folder to train wavs
    train_wavs = ["audios"]

    # Folder to train labels
    train_labels = ["labels"]

    # Folder to test wavs
    test_wavs = []

    # Folder to test labels
    test_labels = []

    @classmethod
    def load_label(cls, label_path):
        with open(label_path, "r") as fin:
            lines = fin.readlines()

        labels = []
        t_unit = 256 / 44100  # ~= 0.0058 secs
        for line in lines:
            elems = line.strip().split(",")
            sec, hz = float(elems[0]), float(elems[1])
            if hz < 1e-10:
                continue
            note = float(pretty_midi.hz_to_note_number(hz))  # Convert return type of np.float64 to float
            end_t = sec + t_unit
            labels.append(Label(start_time=sec, end_time=end_t, note=note))

        return labels

class UnlabeledStructure(BaseStructure):
    """Constant settings of unlabeled datasets"""

    is_labeled = False

    url = None

    # Folder to train wavs
    train_wavs = ["audios"]

    # Folder to test wavs
    test_wavs = []