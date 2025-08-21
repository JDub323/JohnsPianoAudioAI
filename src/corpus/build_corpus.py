from omegaconf import DictConfig
from typing import Any
from pretty_midi import PrettyMIDI
from .DataAugmenter import DataAugmenter
from . import utils
from src.corpus.datatypes import NoteLabels, ProcessedAudioSegment
import pandas as pd
import numpy as np
import os
import librosa
import pretty_midi.pretty_midi
import torch
import logging 


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
    - consider splitting the midi representation after converting to a 2d numpy array, since the alternative is buggy at best. 
    """
    # get the dataframe with all the files
    df_unproc = utils.get_raw_data_df(configs)

    # clean up the dataframe (get rid of any null rows, unneeded cols, etc.)
    # pneumonic: dataframe, unprocessed
    df_unproc = utils.clean_df(df_unproc, configs)
    if configs.verbose: utils.print_data(df_unproc)

    # delete the old processed data (if it exists)
    utils.clean_old_proc_dir(configs)

    # add the folder back
    utils.add_processed_data_folders(configs)

    # preprocess data: convert it all from .wav to whatever spectrogram representation I want.
    # also adds noise to the data which could be from artifacts from an iphone, iff that is 
    # what configs say to do

    # make a DataAugmenter object (custom, simple, uses audiomentations lib)
    my_augmenter = DataAugmenter(configs=configs)

    # make list of dicts for processed data, to be converted into a dataframe
    list_for_proc_df = [] # pneumonic: list for processed dataframe

    # for every audio file 
    for index, row in df_unproc.iterrows():
        # update terminal 
        print(f'\rProcessing row {index + 1:04d} out of {len(df_unproc):04d}', end='')

        # check if passed the row limit
        if utils.passed_row_limit(configs, i=index): break

        # download the .wav file 
        song_wav = utils.download_wav(configs, filename=row['audio_filename'])

        # download the midi file while I am here (if it exists)
        song_midi = utils.download_midi(configs, filename=row['midi_filename'])
        
        # call a split function to return a list of tuples, each with a .wav vector and 
        # the corresponding midi object
        segments = split_track(configs, wav=song_wav, midi=song_midi)

        # for each audio segment
        for index, segment in enumerate(segments):
            # separate the segment tuple
            wav, midi = segment

            # calculate the duration: 
            duration = utils.calc_duration(wav=wav, midi=midi, configs=configs)

            # augment the segment audio (make it a lower quality)
            # TODO IN func below:
            # pad the end of the audio segment (abrupt end. remember not to train with abrupt start)
            # this is a little off, since piano audio is slightly muffled before ending. Is it good enough?
            augmented_wav, augmentations = my_augmenter.augment_data(wav)

            # compute the spectrogram or NMF 
            input_tensor = utils.get_input_vector(configs, augmented_wav)

            # compute the ground truth tensors (NoteLabels object) 
            note_labels = utils.get_truth_tensor(configs, midi)
            
            # calculate what the split should be (test, training, verification)
            split_type = calc_split(configs, row)

            # save the input tensor to wherever the split should be (use the seed for rng)
            logging.info('Saving tensor') # log
            processed_segment = ProcessedAudioSegment(model_input=input_tensor, ground_truth=note_labels)
            save_location = calc_save_location(configs, filename=row['audio_filename'], split_type=split_type, i=index)
            torch.save(processed_segment, os.path.join(save_location[0], save_location[1]))

            # add the data to a list to be made into a dataframe
            add_segment_to_df(list_for_proc_df, name=save_location[1], split=split_type, i=index,
                              duration=duration, augs=augmentations, year=row['year'])

    # save the dataframe
    logging.info('Saving dataframe') # log
    loc = os.path.join(configs.dataset.export_root, configs.dataset.spreadsheet_name)
    df_proc = pd.DataFrame(list_for_proc_df).reset_index(drop=True)
    df_proc.to_csv(loc)

    # save the configs used to generate the data
    logging.info('Saving configs') # log
    utils.save_configs(configs)

    logging.info('Data processed successfully!') # log

# returns a tuple which needs to be joined with path.join. First is the path, second is the filename (w/o relative path)
def calc_save_location(configs, filename: str, split_type: str, i: int) -> tuple[str, str]:
    path = os.path.join(configs.dataset.export_root, split_type)
    assert(filename.endswith('.wav') or filename.endswith('.mp3'))
    new_filename = os.path.basename(filename.removesuffix('.wav').removesuffix('.mp3')) + f"_Section_{i:02d}.pt"
    return (path, new_filename)

def split_track(configs, wav: np.ndarray, midi) -> list[tuple]:
    logging.info('Splitting track') # log
    # get the wav vector
    split_size_in_samples = configs.dataset.sample_rate * configs.dataset.segment_length
    wav_arr = []
    while len(wav) != 0:
        # pop the front of the wav file
        wav_arr.append(wav[:split_size_in_samples])
        wav = wav[split_size_in_samples:]

    # get the midi vector
    if midi != None:
        midi_arr = utils.split_pretty_midi(midi, segment_length=configs.dataset.segment_length)

    # handle case where there is no midi file
    else: 
        midi_arr = [PrettyMIDI()] * len(wav_arr) # this duplicates the same reference, which is fine since they are just filler

    # logs
    logging.info(f'midi after being split, but before getting zipped up') 
    if configs.verbose: utils.print_midi(midi_arr[0])
    logging.info(f'length: {len(midi_arr)}')

    # return a list of tuples 
    ret = list(zip(wav_arr, midi_arr))
    logging.info(f'Split track into {len(ret)} segments') # log
    return ret

def add_segment_to_df(dict_list: list, name: str, split: str,
                      i: int, duration: float, year: int, augs: dict[str, dict[str, Any]]):
    
    # to handle the augmentations, I need to flatten the dictionary, then merge the dictionaries
    flat_augs = utils.flatten_dict(augs)
    
    # note: name must have gotten rid of the path which is normally stored in maestro dataset
    new_filename = os.path.join(split, name)
    new_row = {
            'split': split,
            'filename': new_filename,
            'duration': duration,
            'year': year,
            **flat_augs # merges the dictionaries flat_augs and new_row
            }

    dict_list.append(new_row)

def calc_split(configs, row: pd.Series) -> str:
    if configs.dataset.use_default_splits:
        return str(row['split'])
    else:
        raise ValueError('Random split functionality not programmed yet')

