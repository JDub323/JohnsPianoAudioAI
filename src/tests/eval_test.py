from ..evaluation.calc_metrics import tuple_pianoroll_to_notes
import numpy as np
import os
import torch
import mir_eval
from ..corpus.utils import NoteLabel_to_tensor
from ..corpus.datatypes import ProcessedAudioSegment, NoteLabels

def test_pianoroll_to_notes(data_dir: str = "./data/processed/test/"):
    data = None
    torch.serialization.add_safe_globals([ProcessedAudioSegment, NoteLabels])
    # search for some processed audio segment with a piano-roll torch.tensor dictionary 
    for filename in os.listdir(data_dir):
        if filename.endswith(".pt"):
            # download it 
            filepath = os.path.join(data_dir, filename)
            print("Opening:", filepath)
            data = torch.load(filepath)
            print("Loaded:", type(data))

    # get the piano roll 
    if not data:
        raise ValueError

    mytensor = NoteLabel_to_tensor(data[6].ground_truth) # data is a list, just take arbitrary element

    # send it to pianoroll_to_notes
    intervals, pitches = tuple_pianoroll_to_notes(mytensor, 44100./512)
    print(f"intervals: {intervals}")
    print(f"pitches : {pitches}")
    return intervals, pitches

def test_metric_calcs(data_dir: str = "./data/processed/test"):
    ref_intervals, ref_pitches = test_pianoroll_to_notes(data_dir)
    
    # random set of intervals I made
    est_intervals = np.array([[1.46285714, 1.82276644],
                 [1.46285714, 1.82276644],
                 [0.37151927, 0.39473923],
                 [1.45124717, 1.82276644],
                 [0.08126984, 0.1044898 ],
                 [0.17414966, 0.19736961],
                 [0.40634921, 1.17260771],
                 [1.78793651, 1.82276644],
                 [0.24380952, 0.2554195 ],
                 [0.30185941, 0.32507937],
                 [0.58049887, 1.49768707]])
    est_pitches = np.array([61, 67, 72, 72, 76, 78, 78, 79, 81, 84, 87])
    
    tolerance = .05
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=tolerance,
        offset_ratio=None       # None = don’t require offset matching (ignore linter complaints)
    )
    # breakpoint()
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")

if __name__ == '__main__':
    test_metric_calcs()
