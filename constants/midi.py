import os
from os.path import join as jpath

from librosa import note_to_midi

MODULE_PATH = os.path.abspath(jpath(os.path.split(__file__)[0], '..'))

LOWEST_MIDI_NOTE = note_to_midi("A0")
HIGHEST_MIDI_NOTE = note_to_midi("C8")