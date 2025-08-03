from omegaconf import DictConfig
from src.corpus.utils import print_data, split_pretty_midi_into_segments
import pandas as pd
import numpy as np
import os
import librosa
import pretty_midi
from src.corpus.datatypes import NoteLabels, ProcessedAudioSegment

def build_corpus(configs: DictConfig):
    # get the dataframe with all the files
    csv_path = os.path.join(configs.dataset.data_root, configs.dataset.spreadsheet_name)
    df_unproc = pd.read_csv(csv_path)

    # clean up the dataframe (get rid of any null rows, unneeded cols, etc.)
    # pneumonic: dataframe, unprocessed
    df_unproc = clean_df(df_unproc, configs)
    print_data(df_unproc)

    # delete the old processed data (if it exists)

    # preprocess data: convert it all from .wav to whatever spectrogram representation I want.
    # also adds noise to the data which could be from artifacts from an iphone, iff that is 
    # what configs say to do

    # make df for processed data
    # TODO: add columns for all of the metadata with how the audio was augmented
    column_names = ['split', 'data_filename', 'labels_filename', 'duration', 'year']
    df_proc = pd.DataFrame(columns=column_names) # pneumonic: dataframe, processed

    # for every audio file 
    for index, row in df_unproc.itterrows():
        # download the .wav file 
        path_to_wav = os.path.join(configs.dataset.data_root, row['audio_filename'])
        song_wav = librosa.load(path_to_wav, sr=configs.dataset.sample_rate)

        # download the midi file while I am here
        path_to_midi = os.path.join(configs.dataset.data_root, row['midi_filename'])
        song_midi = pretty_midi.Pretty_MIDI(path_to_midi)

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
            augmented_wav = augment_data(configs, wav)

            # compute the spectrogram or NMF 
            input_tensor = get_input_vector(configs, wav)

            # compute the ground truth tensors (NoteLabels object) 
            note_labels = get_truth_tensor(configs, midi)
            
            # calculate what the split should be (test, training, verification)
            split_type = calc_split(configs, row)

            # save the input tensor to wherever the split should be (use the seed for rng)
            # TODO: refactor this to handle other database setups
            processed_segment = ProcessedAudioSegment(model_input=input_tensor, ground_truth=note_labels)
            save_location = calc_save_location(configs, )
            torch.save(processed_segment, save_location)

            # add the data to a dataframe which has metadata and filename
            # remember to make file names have relative paths
            add_segment_to_df(df_proc, name=filename, split=split_type, i=index)

    # save the dataframe
    df_proc.to_csv(configs.dataset.export_root)

    # save the configs used to generate the data


    # also, save the split dataframes. Since they are split as csvs which point to certain options, 
    # it doesn't really matter if the preprocessed files are in the same folders or not
    # consider making them "not"

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['canonical_composer', 'canonical_title']) # consider dropping the recommended split
    df = df.dropna()
    return df

def calc_save_location(configs, row: pd.Series, split_type: str, i: int) -> str:
    path = os.path.join(configs.dataset.export_root, split_type)
    assert(row['audio_filename'].endswith('.wav'))
    filename = os.path.basename(row['audio_filename']).removesuffix('.wav') + f"_Section_{i:02d}.pt"
    return path.join(path, filename)


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

