import numpy as np
import pretty_midi
from scipy.stats import norm

import os
import sys
MODULE_PATH = os.path.abspath(f"{os.path.split(__file__)[0]}/..")
if sys.path[0] != MODULE_PATH: sys.path.insert(0, MODULE_PATH)

from utils import get_logger

logger = get_logger("Vocal Inference")


def _find_peaks(seq, ctx_len=2, threshold=0.5):
    # Discard the first and the last <ctx_len> frames.
    peaks = []
    for idx in range(ctx_len, len(seq) - ctx_len - 1):
        cur_val = seq[idx]
        if cur_val < threshold:
            continue
        if not all(cur_val > seq[idx - ctx_len:idx]):
            continue
        if not all(cur_val >= seq[idx + 1:idx + ctx_len + 1]):
            continue
        peaks.append(idx)
    return peaks

def _find_first_bellow_th(seq, threshold=0.5):
    activate = False
    for idx, val in enumerate(seq):
        if val > threshold:
            activate = True
        if activate and val < threshold:
            return idx
    return 0

def infer_interval(pred, ctx_len=2, threshold=0.5, min_dura=0.1, t_unit=0.02):
    """Infer the onset and offset time of notes from the raw prediction values"""

    on_peaks = _find_peaks(pred[:, 2], ctx_len=ctx_len, threshold=threshold)
    off_peaks = _find_peaks(pred[:, 4], ctx_len=ctx_len, threshold=threshold)
    if len(on_peaks) == 0 or len(off_peaks) == 0:
        return None

    # Clearing out offsets before first onset (since onset is more accurate)
    off_peaks = [idx for idx in off_peaks if idx > on_peaks[0]]

    on_peak_id = 0
    est_interval = []
    min_len = min_dura / t_unit
    while on_peak_id < len(on_peaks) - 1:
        on_id = on_peaks[on_peak_id]
        next_on_id = on_peaks[on_peak_id + 1]

        off_peak_id = np.where(np.array(off_peaks) >= on_id + min_len)[0]
        if len(off_peak_id) == 0:
            off_id = _find_first_bellow_th(pred[on_id:, 0], threshold=threshold)
        else:
            off_id = off_peaks[off_peak_id[0]]

        if on_id < next_on_id < off_id \
                and np.mean(pred[on_id:next_on_id, 1]) > np.mean(pred[on_id:next_on_id, 0]):
            # Discard current onset, since the duration between current and
            # next onset shows an inactive status.
            on_peak_id += 1
            continue

        if off_id > next_on_id:
            # Missing offset between current and next onset.
            if (off_id - next_on_id) < min_len:
                # Assign the offset after the next onset to the current onset.
                est_interval.append((on_id * t_unit, off_id * t_unit))
                on_peak_id += 1
            else:
                # Insert an additional offset.
                est_interval.append((on_id * t_unit, next_on_id * t_unit))
                on_peak_id += 1
        elif (off_id - on_id) >= min_len:
            # Normal case that one onset has a corressponding offset.
            est_interval.append((on_id * t_unit, off_id * t_unit))
            on_peak_id += 1
        else:
            # Do nothing
            on_peak_id += 1

    # Deal with the border case, the last onset peak.
    on_id = on_peaks[-1]
    off_id = _find_first_bellow_th(pred[on_id:, 0], threshold=threshold) + on_id
    if off_id - on_id >= min_len:
        est_interval.append((on_id * t_unit, off_id * t_unit))

    return np.array(est_interval)


def _conclude_freq(freqs, std=2, min_count=3):
    """Conclude the average frequency with gaussian distribution weighting"""

    # Expect freqs contains zero
    half_len = len(freqs) // 2
    prob_func = lambda x: norm(0, std).pdf(x - half_len)
    weights = [prob_func(idx) for idx in range(len(freqs))]
    avg_freq = 0
    count = 0
    total_weight = 1e-8
    for weight, freq in zip(weights, freqs):
        if freq < 1e-6:
            continue

        avg_freq += weight * freq
        total_weight += weight
        count += 1

    return avg_freq / total_weight if count >= min_count else 0


def infer_midi(interval, agg_f0, t_unit=0.02):
    """Inference the given interval and aggregated F0 to MIDI file"""
    
    fs = round(1 / t_unit)
    max_secs = max(record["end_time"] for record in agg_f0)
    total_frames = round(max_secs) * fs + 10
    flat_f0 = np.zeros(total_frames)
    for record in agg_f0:
        start_idx = int(round(record["start_time"] * fs))
        end_idx = int(round(record["end_time"] * fs))
        flat_f0[start_idx:end_idx] = record["frequency"]

    notes = []
    drum_notes = []
    skip_num = 0
    for onset, offset in interval:
        start_idx = int(round(onset * fs))
        end_idx = int(round(offset * fs))
        freqs = flat_f0[start_idx:end_idx]
        avg_hz = _conclude_freq(freqs)
        if avg_hz < 1e-6:
            skip_num += 1
            note = pretty_midi.Note(velocity=80, pitch=77, start=onset, end=offset)
            drum_notes.append(note)
            continue

        note_num = int(round(pretty_midi.hz_to_note_number(avg_hz)))
        if not (0 <= note_num <= 127):
            logger.warning("Caught invalid note number: %d (should be in range 0~127). Skipping.", note_num)
            skip_num += 1
            continue
        note = pretty_midi.Note(velocity=80, pitch=note_num, start=onset, end=offset)
        notes.append(note)

    if skip_num > 0:
        logger.warning("A total of %d notes are skipped due to lack of corressponding pitch information.", skip_num)

    inst = pretty_midi.Instrument(program=0)
    inst.notes += notes
    drum_inst = pretty_midi.Instrument(program=1, is_drum=True, name="Missing Notes")
    drum_inst.notes += drum_notes
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.append(inst)
    midi.instruments.append(drum_inst)
    return midi
