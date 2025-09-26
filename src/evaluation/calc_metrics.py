# file used since piano-roll representation is not useful, as f1 score note-wise is more accurate.
# This is a bother, since it is necessary to convert from piano roll and then use a separate library,
# so I vibe coded this file:
# note that onset metrics should be calculated note-wise, while activation should be calculated frame-wise.
# idk how to calculate velocity, I'm barely awake rn so I will figure that ts out later

# ok update, I figured out how to use the mir_eval library. Basically, it takes four parameters, one is 
# an array of timestamps for a particular note's onset, then its offset in the row below, one is the correspoding
# note's frequency (in hz), and there are two sets of the above arrays: one for ground truth, and one for estimations. 


import numpy as np
import mir_eval
import torch

def pianoroll_to_notes(pianoroll, fs):
    """
    Convert piano-roll to note events.
    
    Args:
        pianoroll: np.ndarray of shape (time, pitches), binary (0/1)
        fs: frames per second (inverse of hop size)
    
    Returns:
        intervals: np.ndarray of shape (n_notes, 2), onset/offset in seconds
        pitches: np.ndarray of shape (n_notes,), MIDI pitch numbers
    """
    intervals = []
    pitches = []
    (n_time, n_pitch) = pianoroll.shape

    for pitch in range(n_pitch):
        active = np.where(pianoroll[:, pitch] > 0)[0]
        if len(active) == 0:
            continue
        # find contiguous regions of activity
        splits = np.split(active, np.where(np.diff(active) > 1)[0] + 1)
        for segment in splits:
            onset = segment[0] / fs
            offset = (segment[-1] + 1) / fs
            intervals.append([onset, offset])
            pitches.append(pitch + 21)  # assuming index 0 = A0 = MIDI 21

    return np.array(intervals), np.array(pitches)

# TODO
# IMPORTANT: 
# a final post-processing step may merit higher f1 scores:
# apply a smoothing convolution to note onset matrix, then look for the onset at a peak. this 
# theoretically avoids the problem of double note onsets. A way to make a new loss function, which 
# takes this post-processing step into account may be a new innovation which can catapult performace

# calculate metrics for onset or for activation. IDK what to do for velocity lol, I'm just not gonna calc that for now
def get_prec_recall_f1(label_pianoroll: np.ndarray, pred_pianoroll: np.ndarray, tolerance: float, fs: float):

    # Convert ground truth and predictions to events
    ref_intervals, ref_pitches = pianoroll_to_notes(label_pianoroll, fs)
    est_intervals, est_pitches = pianoroll_to_notes(pred_pianoroll, fs)

    # Compute note-level scores with 50ms onset tolerance and offset matching
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=tolerance,
        offset_ratio=None       # None = don’t require offset matching (ignore linter complaints)
    )

    return precision, recall, f1

