from omegaconf import DictConfig
import torch
from pathlib import Path
from typing import Any
from pretty_midi import PrettyMIDI
from torch import norm_except_dim

from .MySharder import MySharder
from .DataAugmenter import DataAugmenter
from . import utils
from .datatypes import NoteLabels, ProcessedAudioSegment
import pandas as pd
import numpy as np
import os
import logging 
import pdb
from datetime import datetime

# TEST: make 100 rows, started the code at 2:25pm, started processing at 2:29pm
# looking like around 27 seconds per row on average. Done around 3:15pm or a bit before, so around 27 seconds to 
# process each row. With this speed, it will take approximately 9 HOURS. Consider multithreading. 

# NEW TEST: NOW ON THE WSL FILE SYSTEM, I will try to make 200 rows. started at 11:37am, with other tabs open,
# but my computer sounded like it was going to explode so I closed them (around 30 tabs of google, including one playing music) 
# on row 40 at 11:41 after 4 minutes. on WSL, looks like 10 rows per minute, or about 6 seconds per row on average. However, 
# things seem to be slowing down. It could have been that tensors were finally being uploaded, which takes a bit.
# done with row 100 at 11:48am, which is still roughly 10 rows per minute. just going on WSL lowers the time to process to 
# about 2 hours 20 minutes, which is about a 4x speedup for FREE :). 2 hours is managable, takes up ~17GB if entire dataset
# done with row 200 before 11:59am. 2.7GB data for 200 rows
# this is apx. 15.8% of the total corpus in around 20 minutes, which is quite nice. 
# command: python3 -m src.scripts.build_corpus dataset.row_limit=200
# other command: tensorboard --logdir=./outputs/run_2025-10-31_11-31-37/logs

# now I shuffle. I will see if it adds non-negligible time

# low-priority TODO: send audiomentations and fft calculations to gpu, parallelize to increase speed (it is hard)
def build_corpus(configs: DictConfig):
    """
    Function to build the corpus. Does the following operations:
    - deletes the processed data currently in the processed data directory
    - download each midi and wav file 
    - convert the midi file to "ground truth" labels (NoteLabels object)
    - add any augmentations to the .wav data to make it more realistic to real-world conditions
    - convert the .wav file to spectrogram 
    - save each segment as ProcessedAudioSegment object using torch.save()

    Possible issues:
    - potentially large overhead with I/O operations in the cloud from saving each tensor individually
    - untested splitting alignment with midi/spectrograms 
    
    Future changes:
    """
    print("CURRENT TIME:")
    print(datetime.now())
    # get the dataframe with all the files
    df_unproc = utils.get_raw_data_df(configs)

    # clean up the dataframe (get rid of any null rows, unneeded cols, etc.)
    # pneumonic: dataframe, unprocessed
    df_unproc = utils.clean_df(df_unproc, configs)
    if configs.verbose: utils.print_data(df_unproc)

    # delete the old processed data (if it exists)
    # update: this function now gets user confirmation before deleting a directory
    utils.clean_old_proc_dir(configs)

    # add the folder back
    utils.add_processed_data_folders(configs)

    # preprocess data: convert it all from .wav to whatever spectrogram representation I want.
    # also adds noise to the data which could be from artifacts from an iphone, iff that is 
    # what configs say to do

    # make a DataAugmenter object (custom, simple, uses audiomentations lib)
    my_augmenter = DataAugmenter(configs=configs)

    # make a sharder to handle the processed data's dataframe and uploads 
    sharder = MySharder(configs)

    # for every audio file 
    for index, row in df_unproc.iterrows():
        # update terminal 
        print(f'\rProcessing row {index + 1:04d} out of {len(df_unproc):04d}', end='')

        # check if passed the row limit
        if utils.passed_row_limit(configs, i=index): break

        # download the .wav file 
        song_wav = utils.download_wav(configs, filename=str(row['audio_filename']))
        silence = np.zeros(configs.dataset.samples_silence, dtype=song_wav.dtype)
        song_wav = np.concatenate((song_wav, silence))

        # download the midi file while I am here (if it exists)
        song_midi = utils.download_midi(configs, filename=str(row['midi_filename']))

        # calculate the total number of frames which will be in the entire song (excluding the padding on last frame)
        total_frames = utils.calc_midi_frame_count(configs, duration=len(song_wav)/configs.dataset.sample_rate)

        # find the number of frames per tensor
        frames_per_segment = utils.calc_midi_frame_count(configs)

        # convert the midi to a NoteLabels object happens in the following function
        # call a split function to return a list of tuples, each with an input tensor and 
        # the corresponding labels object
        segments = split_and_process(configs, wav=song_wav, midi=song_midi, 
                                     total_frames=total_frames, split_frames=frames_per_segment)

        # for each audio segment
        for index, segment in enumerate(segments):
            # separate the segment tuple
            wav, label = segment # segment is a tuple of .wav and of NoteLabels

            # augment the segment audio (make it a lower quality)
            augmented_wav, augmentations = my_augmenter.augment_data(wav)

            # compute the spectrogram or NMF 
            input_tensor = utils.get_input_vector(configs, augmented_wav)

            # calculate what the split should be (test, training, verification)
            split_type = calc_split(configs, row)

            # save the input tensor to wherever the split should be (use the seed for rng)
            # logging.info('Saving tensor') # log
            processed_segment = ProcessedAudioSegment(model_input=input_tensor, ground_truth=label)
            save_location = calc_save_location(configs, filename=str(row['audio_filename']), split_type=split_type, i=index)
            # torch.save(processed_segment, os.path.join(save_location[0], save_location[1]))

            # add the data processed to the sharder
            sharder.push(processed_segment, save_location[1], split_type, int(row['year']), augmentations)

    # tell the sharder that we are done giving it data, and it will upload the df and the rest of the 
    # unfilled shards
    sharder.done_adding_data()

    # shuffle shards 
    logging.info('Cleaning Corpus')
    clean_corpus(configs)

    # save the configs used to generate the data
    logging.info('Saving configs') # log
    utils.save_configs(configs)

    logging.info('Data processed successfully!') # log
    print("CURRENT TIME:")
    print(datetime.now())

# returns a tuple which needs to be joined with path.join. First is the path, second is the filename (w/o relative path)
def calc_save_location(configs, filename: str, split_type: str, i: int) -> tuple[str, str]:
    path = os.path.join(configs.dataset.export_root, split_type)
    assert(filename.endswith('.wav') or filename.endswith('.mp3'))
    new_filename = os.path.basename(filename.removesuffix('.wav').removesuffix('.mp3')) + f"_Section_{i:02d}.pt"
    return (path, new_filename)

# needs to return a tuple of a .wav and its corresponding NoteLabels object
def split_and_process(configs, wav: np.ndarray, midi: PrettyMIDI | None, total_frames: int, split_frames: int) -> list[tuple]:
    logging.info('Splitting track') # log

    # get the wav vector
    split_size_in_samples = configs.dataset.sample_rate * configs.dataset.segment_length
    wav_arr = []
    while len(wav) != 0:
        # pop the front of the wav file
        front_frame = wav[:split_size_in_samples]
        
        # make sure the front frame has the right size 
        if len(front_frame) < split_size_in_samples:
            front_frame = np.pad(front_frame, (0, split_size_in_samples - len(front_frame)), "constant")

        wav_arr.append(front_frame)
        wav = wav[split_size_in_samples:]

    # get truth tensor now can handle "None" as 'midi' if it doesn't exist, so it is fine to not check
    note_labels = utils.get_truth_tensor(configs, midi, frame_count=total_frames)

    # convert the truth tensor to a NoteLabels object
    label_arr = utils.split_truth_tensor(labels=note_labels, segment_frames=split_frames)

    # return a list of tuples 
    ret = list(zip(wav_arr, label_arr))
    logging.info(f'Split track into {len(ret)} segments') # log
    return ret

def calc_split(configs, row: pd.Series) -> str:
    if configs.dataset.use_default_splits:
        return str(row['split'])
    else:
        raise ValueError('Random split functionality not programmed yet')

# function to shuffle the corpus after it was built so dataloader can used cached shards
# "clean corpus" in case I would like to do other post-processing, like removing outliers
def clean_corpus(configs):
    # breakpoint()
    # allow the loading of my shards
    torch.serialization.add_safe_globals([ProcessedAudioSegment, NoteLabels])

    # gather variables
    ptd = os.path.join(configs.dataset.export_root, "train") # only need to shuffle the training dataset
    name = configs.dataset.processed_csv_name
    sz = configs.dataset.sharding.samples_per_shard

    # shuffle elements among each shard
    super_shuffle_shards(ptd, name, sz)

# path is a directory which houses a dataframe which points to all the shards, and all the shards
# shuffles items within each shard, then shuffles the shards
def shuffle_shards(path_to_data, df_name):
    # download the dataframe 
    dataframe_abs_name = os.path.join(path_to_data, df_name)
    df = pd.read_csv(dataframe_abs_name)

    # get a list of shard filenames 
    shards = sorted(Path.glob(Path(path_to_data), "*_shard*.pt")) # the star in beginning lets me shuffle anything

    # shuffle the shards themselves
    num_shards = len(shards)
    shard_perm = np.random.permutation(num_shards)
    shards = [shards[i] for i in shard_perm]

    # for each shard 
    for new_idx, shard_filename in enumerate(shards):
        # get the id 
        shard_id = int(shard_filename.stem.split("_shard")[1])

        # load the shard 
        shard = torch.load(shard_filename)
        n = len(shard)

        # shuffle the shard 
        perm = np.random.permutation(n)
        shard_shuffled = [shard[i] for i in perm]
        torch.save(shard_shuffled, shard_filename)
        
        # apply the same shuffle to the rows of the dataframe
        mask = df["shard_number"] == shard_id
        shard_rows = df.loc[mask].copy().reset_index(drop=True)
        shard_rows = shard_rows.iloc[perm].reset_index(drop=True)
        
        # apply the shard-level shuffle
        shard_rows["shard_number"] = new_idx

        # put the shuffled rows back into the dataframe
        df.loc[mask] = shard_rows.values

    # save the dataframe with the same filename (overwrite it)
    df.to_csv(dataframe_abs_name, index=False)
    logging.info(f"Shuffling complete. Updated dataframe saved to {dataframe_abs_name}")

# path is a directory which houses a dataframe which points to all the shards, and all the shards
# shard_len is size of all shards except last one
def super_shuffle_shards(path_to_data, df_name, shard_len):
    # download the dataframe 
    dataframe_abs_name = os.path.join(path_to_data, df_name)
    df = pd.read_csv(dataframe_abs_name)

    # get a list of shard filenames 
    shards = sorted(Path.glob(Path(path_to_data), "*_shard*.pt")) # the star in beginning lets me shuffle anything

    # get the total number of samples
    shard_info = [] # used for global index mapping
    global_idx = 0
    
    for shard_path in shards:
        shard_id = int(shard_path.stem.split("_shard")[1])
        shard_size = len(torch.load(shard_path))
        shard_info.append((shard_id, shard_size, global_idx, shard_path))
        global_idx += shard_size
    
    total_samples = global_idx
    # shard_info.sort(key=lambda x: x[2]) # sort shard info by global index.
    
    # create random permutation of data
    permutation = np.arange(total_samples)

    swaps = []
    for i in range(total_samples - 1):
        j = np.random.randint(i, total_samples)
        if i != j:
            swaps.append((i, j))
            permutation[i], permutation[j] = permutation[j], permutation[i]
    
    logging.info(f"Generated {len(swaps)} swaps")
    
    # Helper functions
    def swap_within_shard(shard, rows, local_i, local_j):
        """Swap two elements within the same shard and dataframe rows."""
        # Swap samples
        shard[local_i], shard[local_j] = shard[local_j], shard[local_i]
        
        # Swap dataframe rows (entire rows, preserving all columns)
        rows.iloc[local_i], rows.iloc[local_j] = rows.iloc[local_j].copy(), rows.iloc[local_i].copy()
        
        # Update shard_index to reflect new positions
        rows.at[local_i, 'shard_index'] = local_i
        rows.at[local_j, 'shard_index'] = local_j
    
    def swap_between_shards(shard_a, rows_a, local_a, shard_a_id,
                           shard_b, rows_b, local_b, shard_b_id):
        """Swap elements between two different shards and their dataframe rows."""
        # Swap samples
        shard_a[local_a], shard_b[local_b] = shard_b[local_b], shard_a[local_a]
        
        # Get the rows to swap
        row_a = rows_a.iloc[local_a].copy()
        row_b = rows_b.iloc[local_b].copy()
        
        # Update shard_number and shard_index for the swapped rows
        row_a['shard_number'] = shard_b_id
        row_a['shard_index'] = local_b
        row_b['shard_number'] = shard_a_id
        row_b['shard_index'] = local_a
        
        # Perform the swap
        rows_a.iloc[local_a] = row_b
        rows_b.iloc[local_b] = row_a

    # Group swaps by shard pairs
    shard_pair_swaps = {}
    for i, j in swaps:
        shard_i = i // shard_len
        shard_j = j // shard_len
        pair = tuple(sorted([shard_i, shard_j]))
        
        if pair not in shard_pair_swaps:
            shard_pair_swaps[pair] = []
        shard_pair_swaps[pair].append((i, j))

    
    # apply the swaps to the dataframe and shards
    # the number of swap_list should be equal to (n+1)n/2 (where n is the number of shards)
    for pair_idx, (pair, swap_list) in enumerate(shard_pair_swaps.items()):
        print(f"Shuffling swap list number {pair_idx}")
        shard_a_id, shard_b_id = pair
        
        # Get shard paths
        shard_a_path = next(sp for sid, ss, si, sp in shard_info if sid == shard_a_id)
        shard_a = torch.load(shard_a_path)
        same_shard = (shard_a_id == shard_b_id)
        if same_shard:
            shard_b = shard_a
            shard_b_path = shard_a_path
        else:
            shard_b_path = next(sp for sid, ss, si, sp in shard_info if sid == shard_b_id)
            shard_b = torch.load(shard_b_path)
        
        # Load dataframe rows
        mask_a = df["shard_number"] == shard_a_id
        rows_a = df.loc[mask_a].copy().reset_index(drop=True)
        if same_shard:
            rows_b = rows_a
        else:
            mask_b = df["shard_number"] == shard_b_id
            rows_b = df.loc[mask_b].copy().reset_index(drop=True)
        
        # for each index i, j to swap
        for i, j in swap_list:
            shard_i_id, local_i = (i // shard_len, i % shard_len)
            shard_j_id, local_j = (j // shard_len, j % shard_len)
            
            if same_shard:
                # Both in same shard
                swap_within_shard(shard_a, rows_a, local_i, local_j)
            else:
                # Between different shards
                if shard_i_id == shard_a_id:
                    swap_between_shards(shard_a, rows_a, local_i, shard_a_id,
                                       shard_b, rows_b, local_j, shard_b_id)
                else:
                    swap_between_shards(shard_b, rows_b, local_j, shard_b_id,
                                       shard_a, rows_a, local_i, shard_a_id)
        
        # save shards
        torch.save(shard_a, shard_a_path)
        if not same_shard:
            torch.save(shard_b, shard_b_path)
        
        # write dataframe rows back
        df.loc[mask_a] = rows_a.values
        if not same_shard:
            mask_b = df["shard_number"] == shard_b_id
            df.loc[mask_b] = rows_b.values
        
        logging.info(f"Processed shard pair {pair_idx+1}/{len(shard_pair_swaps)}: "
                    f"shards {shard_a_id}, {shard_b_id} ({len(swap_list)} swaps)")

        # if shard_a != shard_b:
        #     del shard_b
        # del shard_a 
    
    # Save the dataframe ONCE at the end
    df.to_csv(dataframe_abs_name, index=False)
    logging.info(f"Shuffling complete. Updated dataframe saved to {dataframe_abs_name}")
    
