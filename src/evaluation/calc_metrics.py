# file used since piano-roll representation is not useful, as f1 score note-wise is more accurate.
# This is a bother, since it is necessary to convert from piano roll and then use a separate library,
# so I vibe coded this file:
# note that onset metrics should be calculated note-wise, while activation should be calculated frame-wise.
# idk how to calculate velocity, I'm barely awake rn so I will figure that ts out later

# ok update, I figured out how to use the mir_eval library. Basically, it takes four parameters, one is 
# an array of timestamps for a particular note's onset, then its offset in the row below, one is the correspoding
# note's frequency (in hz), and there are two sets of the above arrays: one for ground truth, and one for estimations. 


from mir_eval.transcription import precision_recall_f1_overlap
import numpy as np
import mir_eval
import torch

# VERY IMPORTANT:
# Sometimes, there are note onsets which are not considered activations. That is, the note onset matrix 
# is active, but the note activation matrix is not (a contradiction). As the model gets more accurate, the 
# probability of this happening should go to 0. Until then, there are two solutions to this:
# 1 add all note onsets on activation matrix to activations as stocatto notes, then use the activation matrix 
# to detect onsets and offsets.
# 2 add all note onsets from activation matrix to the onset matrix, then use onsets to detect beginnings.
# I think the best solution is one where notes from both the activation and onsets are considered (although 
# this will lead to more false positives), and hope that the false positive rate goes to 0. I may change this.
# TODO: curry this TS so I can have many functions which are similar to this one
# torch.set_printoptions(threshold=float('inf'))
# def tuple_pianoroll_to_notes(pianoroll: torch.Tensor, fs):
#     # pianoroll must have torch tensor shape (3, 88, T). 2 in first dim. would also work, but why would you do that??
#     """
#     Convert piano-roll to note events.
#
#     Args:
#         pianoroll: torch.Tensor of shape (3,88,157), binary (0/1) for onset and velocity
#         fs: frames per second (inverse of hop size over sampling rate??)
#
#     Returns:
#         intervals: np.ndarray of shape (n_notes, 2), onset/offset in seconds
#         pitches: np.ndarray of shape (n_notes,), MIDI pitch numbers
#     """
#     intervals = []
#     pitches = []
#
#     ACTIV_MIN = 0.5 # defines the min value for an activation to be considered valid
#     ONSET_MIN = 0.7 # SAME
#
#     activation = pianoroll[0]
#     onset = pianoroll[1]
#     # now both of these have shape 88 by TIME
#
#     # add a 0 to the end of the list, so all notes have offsets. this will make activation be of shape (88, TIME + 1)
#     # make sure the tensors are on the same device
#     empty_vals = torch.zeros((activation.shape[0], 1), dtype=activation.dtype, device=activation.device) 
#     activation = torch.cat((activation, empty_vals), dim=1)
#
#     # for each key
#     for key in range(88):
#         onsets = torch.nonzero(onset[key] > ONSET_MIN) # find all note onset frames for this key
#         if len(onsets) == 0:
#             continue
#
#         # find regions of activity using activations (really the activations's offsets) rather than onsets
#         # get a 1d binary representation of the list of activations
#         activation_bin = torch.where(activation[key] > ACTIV_MIN, 1.0, 0.0)
#
#         # get all indices where the difference between two subsequent values is -1
#         offsets = torch.where(torch.diff(activation_bin) == -1)[0]
#         # notice this array contains the indices of all of the last times the notes are on, not the first frame they are off.
#         # thus, before I convert them to times, I will add 1 to every element of the torch array
#         offsets = offsets + 1
#
#         # finally, I need to pair note onset indices with the corresponding next note offset index
#         # for my purposes, I will disallow the capacity for a note to have an onset and offset on the same frame.
#         onset_idx = 0
#         offset_idx = 0
#
#         while onset_idx < len(onsets):
#             current_onset = onsets[onset_idx]
#
#             # Find the next offset that occurs after this onset
#             while offset_idx < len(offsets) and offsets[offset_idx] <= current_onset:
#                 offset_idx += 1
#
#             # If we found a valid offset, create the note
#             if offset_idx < len(offsets):
#                 current_offset = offsets[offset_idx]
#
#                 # Convert frames to seconds
#                 onset_time = current_onset.item() / fs
#                 offset_time = current_offset.item() / fs
#
#                 intervals.append([onset_time, offset_time])
#                 pitches.append(key + 21)  # MIDI pitch (key 0 = A0 = MIDI 21)
#
#                 onset_idx += 1
#             else:
#                 # No more offsets available, break
#                 break
#
#     return np.array(intervals), np.array(pitches)


def tuple_pianoroll_to_notes(pianoroll: torch.Tensor, fs):
    """
    Convert piano-roll to note events.
    
    Args:
        pianoroll: torch.Tensor of shape (3,88,T), binary (0/1) for onset/activation
        fs: frames per second
    
    Returns:
        intervals: np.ndarray of shape (n_notes, 2), onset/offset in seconds
        pitches: np.ndarray of shape (n_notes,), MIDI pitch numbers
    """
    # Convert to numpy immediately
    pianoroll_np = pianoroll.detach().cpu().numpy()
    
    intervals = []
    pitches = []
    ACTIV_MIN = 0.5
    ONSET_MIN = 0.7
    
    activation = pianoroll_np[0]
    onset = pianoroll_np[1]
    
    # Add padding for offset detection
    activation = np.pad(activation, ((0, 0), (0, 1)), constant_values=0)
    
    # for each key
    for key in range(88):
        # Find onset frames
        onsets = np.where(onset[key] > ONSET_MIN)[0]
        if len(onsets) == 0:
            continue
        
        # Get binary activation
        activation_bin = np.where(activation[key] > ACTIV_MIN, 1.0, 0.0)
        
        # Find offsets (where activation drops from 1 to 0)
        offsets = np.where(np.diff(activation_bin) == -1)[0] + 1
        
        # Pair onsets with offsets
        onset_idx = 0
        offset_idx = 0
        
        while onset_idx < len(onsets):
            current_onset = onsets[onset_idx]
            
            # Find the next offset after this onset
            while offset_idx < len(offsets) and offsets[offset_idx] <= current_onset:
                offset_idx += 1
            
            # If we found a valid offset, create the note
            if offset_idx < len(offsets):
                current_offset = offsets[offset_idx]
                
                # Convert frames to seconds
                onset_time = current_onset / fs
                offset_time = current_offset / fs
                
                intervals.append([onset_time, offset_time])
                pitches.append(key + 21)  # MIDI pitch (key 0 = A0 = MIDI 21)
                
                onset_idx += 1
            else:
                break
    
    return np.array(intervals), np.array(pitches)

# TODO
# IMPORTANT: 
# a final post-processing step may merit higher f1 scores:
# apply a smoothing convolution to note onset matrix, then look for the onset at a peak. this 
# theoretically avoids the problem of double note onsets. A way to make a new loss function, which 
# takes this post-processing step into account may be a new innovation which can catapult performace

# calculate metrics for onset or for activation. IDK what to do for velocity lol, I'm just not gonna calc that for now
# uses onset matrix to decide when notes are pressed, use activation matrix to decide when notes are released
# returns in the shape precision, recall, f1
def get_prec_recall_f1(label_pianoroll: torch.Tensor, pred_pianoroll: torch.Tensor, tolerance: float, fs: float):
    # the pianorolls are of shape (1129, 88, 157) = (clip, representation, keys, time). Representations are alphabetical order
    # make sure they have the same number of clips
    assert label_pianoroll.shape[0] == pred_pianoroll.shape[0]
    sample_count = label_pianoroll.shape[0]

    sum_metrics = [0.,0.,0.]

    # for each sample
    for i in range(sample_count):
        # Convert ground truth and predictions to events
        ref_intervals, ref_pitches = tuple_pianoroll_to_notes(label_pianoroll[i], fs)
        est_intervals, est_pitches = tuple_pianoroll_to_notes(pred_pianoroll[i], fs)

        # cannot pass intervals of size 0 to mir_eval. If either of these are 0, add 0 for all metrics, 
        # unless both are 0, when you add 1 since it is a perfect job guessing no notes
        if ref_pitches.size == 0 and est_pitches.size == 0:
            sum_metrics[0] += 1 
            sum_metrics[1] += 1 
            sum_metrics[2] += 1 
            continue
        if ref_pitches.size == 0 or est_pitches.size == 0:
            continue


        # Compute note-level scores with 50ms onset tolerance and offset matching
        precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=tolerance,
            offset_ratio=None       # None = don’t require offset matching (ignore linter complaints)
        )
        sum_metrics[0] += precision
        sum_metrics[1] += recall
        sum_metrics[2] += f1
    
    return sum_metrics[0] / sample_count, sum_metrics[1] / sample_count, sum_metrics[2] / sample_count

