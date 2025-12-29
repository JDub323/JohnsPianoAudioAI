from omegaconf import DictConfig
import gc
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

