from omegaconf import DictConfig
from utils import download_wav, print_data, split_pretty_midi_into_segments, complex_to_log_spectrogram
from augment_data import augment_data, generate_augmentations
from datatypes import NoteLabels, ProcessedAudioSegment
import pandas as pd
import numpy as np
import os
import librosa
import pretty_midi
import shutil 
import torch

def build_corpus(configs: DictConfig):
    # get the dataframe with all the files
    csv_path = os.path.join(configs.dataset.data_root, configs.dataset.spreadsheet_name)
    df_unproc = pd.read_csv(csv_path)

    # clean up the dataframe (get rid of any null rows, unneeded cols, etc.)
    # pneumonic: dataframe, unprocessed
    df_unproc = clean_df(df_unproc, configs)
    print_data(df_unproc)

    # delete the old processed data (if it exists)
    if os.path.exists(configs.dataset.export_root):
        print(f"Deleting old processed data directory: {configs.dataset.export_root}")
        shutil.rmtree(configs.dataset.export_root)

        # add the folder back
        os.makedirs(configs.dataset.export_root, exist_ok=True)

    # preprocess data: convert it all from .wav to whatever spectrogram representation I want.
    # also adds noise to the data which could be from artifacts from an iphone, iff that is 
    # what configs say to do

    # make df for processed data
    column_names = ['split', 'filename', 'duration', 'year', 'augmentations']
    df_proc = pd.DataFrame(columns=column_names) # pneumonic: dataframe, processed

    # for every audio file 
    for index, row in df_unproc.itterrows():
        # download the .wav file 
        song_wav = download_wav(configs, filename=row['midi_filename'])

        # download the midi file while I am here
        path_to_midi = os.path.join(configs.dataset.data_root, row['midi_filename'])
        song_midi = pretty_midi.Pretty_MIDI(path_to_midi)
        # note that song_midi can either be a mono np.ndarray or stereo, depending on configs

        # call a split function to return a list of tuples, each with a .wav vector and 
        # the corresponding midi object
        segments = split_track(configs, wav=song_wav, midi=song_midi)

        # for each audio segment
        for index, segment in enumerate(segments):
            # separate the segment tuple
            wav, midi = segment

            # augment the segment audio (make it a lower quality)
            # TODO IN func below:
            # pad the end of the audio segment (abrupt end. remember not to train with abrupt start)
            # this is a little off, since piano audio is slightly muffled before ending. Is it good enough?
            augmentations = generate_augmentations(configs)
            augmented_wav = augment_data(configs, wav, augmentations)

            # compute the spectrogram or NMF 
            input_tensor = get_input_vector(configs, augmented_wav)

            # compute the ground truth tensors (NoteLabels object) 
            note_labels = get_truth_tensor(configs, midi)
            
            # calculate what the split should be (test, training, verification)
            split_type = calc_split(configs, row)

            # save the input tensor to wherever the split should be (use the seed for rng)
            processed_segment = ProcessedAudioSegment(model_input=input_tensor, ground_truth=note_labels)
            save_location = calc_save_location(configs, filename=row['audio_filename'], split_type=split_type, i=index)
            torch.save(processed_segment, os.path.join(save_location[0], save_location[1]))

            # add the data to a dataframe which has metadata and filename
            # remember to make file names have relative paths
            add_segment_to_df(df_proc, name=filename, split=split_type, i=index)

    # save the dataframe
    df_proc.to_csv(configs.dataset.export_root)

    # save the configs used to generate the data


    # also, save the split dataframes. Since they are split as csvs which point to certain options, 
    # it doesn't really matter if the preprocessed files are in the same folders or not
    # consider making them "not"

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

def get_input_vector(configs, wav: np.ndarray) -> torch.Tensor:
    # gather the configs 
    sr = configs.dataset.transform.sample_rate
    hl = configs.dataset.transform.hop_length
    nb = configs.dataset.transform.num_bins
    fs = configs.dataset.transform.filter_scale
    wt = configs.dataset.transform.window_type
    ws = configs.dataset.transform.window_size
    s = configs.dataset.transform.scale
    fmin = configs.dataset.transform.fmin
    fmax = configs.dataset.transform.fmax

    # take the correct transform:
    # log-mel spectrogram
    if configs.dataset.input_representation == 'mel':
        # compute complex power spectrogram 
        C = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=ws, hop_length=hl, window=wt, n_mels=nb)

        # convert the complex, power spectrogram to log-mel
        ret = complex_to_log_spectrogram(spectro=C) 
        

    # constant q transform 
    elif configs.dataset.input_representation == 'cqt':
        # compute the constant q transform
        C = librosa.cqt(y=wav, sr=sr, hop_length=hl, n_bins=nb, filter_scale=fs, window=wt, scale=s)

        # return var is 
        ret = complex_to_log_spectrogram(spectro=C)

    # non-negative matrix factorization
    elif configs.dataset.input_representation == 'nmf':
        raise ValueError('This functionality was not added yet')
    else:
        raise ValueError('Invalid input representation')


    # return the transform as a torch Tensor. The batch size will remain 1 here, will merge many tensors 
    # together before training


def get_truth_tensor(configs, midi) -> torch.Tensor

def add_segment_to_df(configs, )

def split_track(configs, wav: np.ndarray, midi) -> list[tuple]:
    # get the midi vector
    midi_arr = split_pretty_midi_into_segments(midi, segment_duration=configs.dataset.segment_length)

    # get the wav vector
    split_size_in_samples = configs.dataset.sample_rate * configs.dataset.segment_length
    wav_arr = []
    while wav:
        # pop the front of the wav file
        wav_arr.append(wav[:split_size_in_samples])
        wav = wav[split_size_in_samples:]

    # return a list of tuples 
    return list(zip(wav_arr, midi_arr))

