from omegaconf import DictConfig
from pandas.core.indexes.base import PrettyDict
from src.corpus.utils import *
from src.corpus.augment_data import augment_data, generate_augmentations
from src.corpus.datatypes import AugmentationData, NoteLabels, ProcessedAudioSegment
import pandas as pd
import numpy as np
import os
import librosa
import pretty_midi.pretty_midi
import shutil 
import torch

def build_corpus(configs: DictConfig):
    # get the dataframe with all the files
    print('Downloading dataframe')
    csv_path = os.path.join(configs.dataset.data_root, configs.dataset.spreadsheet_name)
    df_unproc = pd.read_csv(csv_path)

    # clean up the dataframe (get rid of any null rows, unneeded cols, etc.)
    # pneumonic: dataframe, unprocessed
    print('Cleaning dataframe')
    df_unproc = clean_df(df_unproc, configs)
    print_data(df_unproc)

    # delete the old processed data (if it exists)
    if os.path.exists(configs.dataset.export_root):
        print(f"Deleting old processed data directory: {configs.dataset.export_root}")
        shutil.rmtree(configs.dataset.export_root)

    # add the folder back
    print('Creating directory for processed data')
    add_processed_data_folders(configs)

    # preprocess data: convert it all from .wav to whatever spectrogram representation I want.
    # also adds noise to the data which could be from artifacts from an iphone, iff that is 
    # what configs say to do

    # make df for processed data
    column_names = ['split', 'filename', 'duration', 'year', 'augmentations']
    df_proc = pd.DataFrame(columns=column_names) # pneumonic: dataframe, processed

    # for every audio file 
    for index, row in df_unproc.iterrows():
        # download the .wav file 
        song_wav = download_wav(configs, filename=row['audio_filename'])

        # download the midi file while I am here
        print('Downloading midi')
        path_to_midi = os.path.join(configs.dataset.data_root, row['midi_filename'])
        song_midi = pretty_midi.PrettyMIDI(path_to_midi)
        # print_midi(song_midi)
        # note that song_midi can either be a mono np.ndarray or stereo, depending on configs

        # call a split function to return a list of tuples, each with a .wav vector and 
        # the corresponding midi object
        print('Splitting track')
        segments = split_track(configs, wav=song_wav, midi=song_midi)

        # for each audio segment
        for index, segment in enumerate(segments):
            # separate the segment tuple
            wav, midi = segment

            # calculate the duration: 
            duration = calc_duration(wav=wav, midi=midi)

            # augment the segment audio (make it a lower quality)
            # TODO IN func below:
            # pad the end of the audio segment (abrupt end. remember not to train with abrupt start)
            # this is a little off, since piano audio is slightly muffled before ending. Is it good enough?
            print('Augmenting data')
            augmentations = generate_augmentations(configs)
            augmented_wav = augment_data(configs, wav, augmentations)

            # compute the spectrogram or NMF 
            print('Computing input tensor')
            input_tensor = get_input_vector(configs, augmented_wav)

            # compute the ground truth tensors (NoteLabels object) 
            print('Computing ground truth tensors')
            note_labels = get_truth_tensor(configs, midi)
            
            # calculate what the split should be (test, training, verification)
            split_type = calc_split(configs, row)

            # save the input tensor to wherever the split should be (use the seed for rng)
            # consider making this its own function
            print('Saving tensor')
            processed_segment = ProcessedAudioSegment(model_input=input_tensor, ground_truth=note_labels)
            save_location = calc_save_location(configs, filename=row['audio_filename'], split_type=split_type, i=index)
            torch.save(processed_segment, os.path.join(save_location[0], save_location[1]))

            # add the data to a dataframe which has metadata and filename
            # remember to make file names have relative paths
            add_segment_to_df(df_proc, name=row['audio_filename'], split=split_type, i=index,
                              duration=duration, augs=augmentations, year=row['year'])

    # save the dataframe
    print('Saving dataframe')
    df_proc.to_csv(configs.dataset.export_root)

    # save the configs used to generate the data
    print('Saving configs')
    save_configs(configs)

    print('Data processed successfully!')

def clean_df(df: pd.DataFrame, configs) -> pd.DataFrame:
    df = df.drop(columns=['canonical_composer', 'canonical_title']) # consider dropping the recommended split
    df = df.dropna()
    return df

# returns a tuple which needs to be joined with path.join. First is the path, second is the filename (w/o relative path)
def calc_save_location(configs, filename: str, split_type: str, i: int) -> tuple[str, str]:
    path = os.path.join(configs.dataset.export_root, split_type)
    assert(filename.endswith('.wav'))
    new_filename = os.path.basename(filename.removesuffix('.wav')) + f"_Section_{i:02d}.pt"
    return (path, new_filename)

# TODO: actually test this function
def get_input_vector(configs, wav: np.ndarray) -> torch.Tensor:
    # gather the configs 
    sr = configs.dataset.sample_rate
    hl = configs.dataset.transform.hop_length
    nb = configs.dataset.transform.n_bins
    fs = configs.dataset.transform.filter_scale
    wt = configs.dataset.transform.window_type
    ws = configs.dataset.transform.window_size
    s = configs.dataset.transform.scale
    fmin = configs.dataset.transform.f_min

    method = configs.dataset.transform.input_representation

    # take the correct transform:
    # log-mel spectrogram
    if method == 'mel':
        # compute complex power spectrogram 
        C = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=ws, hop_length=hl, window=wt, n_mels=nb, fmin=fmin)

        # convert the complex, power spectrogram to log-mel
        ret = complex_to_log_spectrogram(spectro=C) 

    # constant q transform 
    elif method == 'cqt':
        # compute the constant q transform
        C = librosa.cqt(y=wav, sr=sr, hop_length=hl, n_bins=nb, filter_scale=fs, window=wt, scale=s, fmin=fmin)

        # return var is 
        ret = complex_to_log_spectrogram(spectro=C)

    # non-negative matrix factorization
    elif method == 'nmf':
        raise ValueError('This functionality was not added yet')
    else:
        raise ValueError('Invalid input representation')


    # return the transform as a torch Tensor. The batch size will remain 1 here, will merge many tensors 
    # together before training. the number of channels will be configs.num_channels
    if configs.dataset.num_channels != 1:
        raise ValueError('I am too lazy to figure out how to get multiple channels in a torch tensor')
    return torch.from_numpy(ret)

def get_truth_tensor(configs, midi: pretty_midi.PrettyMIDI) -> NoteLabels:
    duration = midi.get_end_time()
    
    sr = configs.dataset.sample_rate
    ws = configs.dataset.transform.window_size
    hl = configs.dataset.transform.hop_length
    frame_count = librosa.time_to_frames(times=duration, hop_length=hl, n_fft=ws, sr=sr)
    print(f'frame count: {frame_count}')

    fs = sr / hl # frames per second = window size / hop length 
    
    n_keys = 88
    onset_matrix = np.zeros((n_keys, frame_count), dtype=bool)
    activation_matrix = np.zeros((n_keys, frame_count), dtype=bool)
    velocity_matrix = np.zeros((n_keys, frame_count), dtype=np.float32)

    for note in midi.instruments[0].notes:
        if note.pitch < 21 or note.pitch > 108:
            continue  # skip notes outside piano range

        pitch_index = note.pitch - 21
        onset_frame = int(np.round(note.start * fs))
        offset_frame = int(np.round(note.end * fs))

        if onset_frame < frame_count:
            onset_matrix[pitch_index, onset_frame] = True

        activation_matrix[pitch_index, onset_frame:offset_frame] = True
        velocity = note.velocity / 127.0  # normalize
        velocity_matrix[pitch_index, onset_frame:offset_frame] = velocity

    # convert to torch tensors 
    am = torch.from_numpy(activation_matrix)
    vm = torch.from_numpy(activation_matrix)
    om = torch.from_numpy(activation_matrix)

    ret = NoteLabels(activation_matrix=am, onset_matrix=om, velocity_matrix=vm)
    return ret


def split_track(configs, wav: np.ndarray, midi) -> list[tuple]:
    # get the midi vector
    midi_arr = split_pretty_midi(midi, segment_length=configs.dataset.segment_length)
    print(f'midi after being split, but before getting zipped up')
    print_midi(midi_arr[0])
    print(f'length: {len(midi_arr)}')

    # get the wav vector
    split_size_in_samples = configs.dataset.sample_rate * configs.dataset.segment_length
    wav_arr = []
    while len(wav) != 0:
        # pop the front of the wav file
        wav_arr.append(wav[:split_size_in_samples])
        wav = wav[split_size_in_samples:]

    # return a list of tuples 
    ret = list(zip(wav_arr, midi_arr))
    print(f'Split track into {len(ret)} segments')
    return ret

def add_segment_to_df(df: pd.DataFrame, name: str, split: str, i: int, duration: float, year: int, augs: AugmentationData):
    #     column_names = ['split', 'filename', 'duration', 'year', 'augmentations'] 
    # note: name must have gotten rid of the path which is normally stored in maestro dataset
    new_filename = os.path.join(split, name)
    new_row = {
            'split': split,
            'filename': new_filename,
            'duration': duration,
            'year': year,
            'augmentations': augs
            }
    df.loc[len(df)] = new_row

def calc_split(configs, row: pd.Series) -> str:
    if configs.dataset.use_default_splits:
        return str(row['split'])
    else:
        raise ValueError('Random split functionality not programmed yet')


