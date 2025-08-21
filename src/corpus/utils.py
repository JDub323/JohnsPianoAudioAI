import omegaconf
import pandas as pd
import numpy as np
import pretty_midi
from librosa import power_to_db
from os import path, makedirs
import librosa
from tabulate import tabulate
import logging
import shutil 
import torch 
from .datatypes import NoteLabels
from pretty_midi.pretty_midi import PrettyMIDI

def print_data(df: pd.DataFrame) -> None:
    print("Data: ")
    print(df)
    # Get basic info (useful for checking data types, non-null counts)
    print("\nDataFrame Info:")
    df.info()

# function only called if configs.verbose
def print_midi(midi) -> None:
    print('MIDI:')

    note_rows = []
    headers = ['Instrument', 'Pitch (num)', 'Pitch (name)', 'Start (s rounded)', 'End (s rounded)', 'Velocity']
    for i, instrument in enumerate(midi.instruments):
        for note in instrument.notes:
            note_rows.append([
                instrument.name or f'Program {instrument.program}',
                note.pitch,
                pretty_midi.note_number_to_name(note.pitch),
                round(note.start, 3),
                round(note.end, 3),
                note.velocity,
            ])

    if not note_rows:
        print("No instruments or notes found in this MIDI.")
        return

    note_rows.sort(key=lambda x: x[3])
    print(tabulate(note_rows, headers=headers, tablefmt='fancy_grid'))

def add_processed_data_folders(configs):
    logging.info('Creating directory for processed data') # log
    makedirs(configs.dataset.export_root, exist_ok=True)
    subfolders = ['train', 'test', 'validation']
    for sf in subfolders:
        dir_ = path.join(configs.dataset.export_root, sf)
        makedirs(dir_, exist_ok=True)

def clean_old_proc_dir(configs):
    if path.exists(configs.dataset.export_root):
        logging.info(f"Deleting old processed data directory: {configs.dataset.export_root}") # log
        shutil.rmtree(configs.dataset.export_root)

# prints a raw audio augmentations dict to a nice format
def print_augs(augs) -> None:
    flat_dict = flatten_dict(augs)
    print(tabulate(flat_dict.items(), headers=['Key', 'Value'], tablefmt='fancy_grid'))

# returns a flattened version of an augmentations dictionary
def flatten_dict(augs) -> dict:
    return pd.json_normalize(augs, record_prefix='augmentations').to_dict(orient='records')[0]

def passed_row_limit(configs, i: int) -> bool:
    lim = configs.dataset.row_limit
    if lim == -1: return False # if the limit is set to -1, it is unbound: should process all rows

    return lim <= i

def get_raw_data_df(configs):
    logging.info('Downloading raw dataframe') # log
    csv_path = path.join(configs.dataset.data_root, configs.dataset.spreadsheet_name)
    ret = pd.read_csv(csv_path)
    return ret

def get_processed_data_df(configs):
    logging.info('Downloading processed dataframe') # log
    csv_path = path.join(configs.dataset.export_root, configs.dataset.spreadsheet_name)
    ret = pd.read_csv(csv_path)
    return ret

def clean_df(df: pd.DataFrame, configs) -> pd.DataFrame:
    logging.info('Cleaning dataframe') # log
    df = df.drop(columns=['canonical_composer', 'canonical_title']) # consider dropping the recommended split
    df = df.dropna()
    return df

def complex_to_log_spectrogram(spectro: np.ndarray) -> np.ndarray:
    return power_to_db(np.abs(spectro)).astype(np.float32)

def download_wav(configs, filename: str) -> np.ndarray:
    logging.info(f'Downloading {filename} audio') # log
    path_to_wav = path.join(configs.dataset.data_root, filename)
    use_mono = configs.dataset.num_channels == 1
    ret, _ = librosa.load(path_to_wav, sr=configs.dataset.sample_rate, mono=use_mono)
    return ret

def download_midi(configs, filename: str) -> PrettyMIDI | None:
    if configs.dataset.midis_exist:
        logging.info('Downloading midi') # log
        path_to_midi = path.join(configs.dataset.data_root, filename)
        ret = pretty_midi.PrettyMIDI(path_to_midi)
        if configs.verbose: print_midi(ret)
        # note that song_midi can either be a mono np.ndarray or stereo, depending on configs
    # make sure song midi is bound
    else: ret = None

    return ret

def calc_duration(configs, wav, midi) -> float:
    # midi time duration
    midi_duration = midi.get_end_time()
    wav_duration = len(wav) / configs.dataset.sample_rate

    # test to make sure the durations are the same length (allow for some floating point error)
    if configs.dataset.midis_exist:
        assert abs(midi_duration - wav_duration) < 1e-6, "midi and wav files differ in length"

    return wav_duration

def save_configs(configs) -> None:
    root = configs.dataset.export_root
    save_location = path.join(root, 'configs.yaml')
    omegaconf.OmegaConf.save(configs, save_location)
    logging.info('Configs saved!')

def get_input_vector(configs, wav: np.ndarray) -> torch.Tensor:
    logging.info('Computing input tensor') # log

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
    logging.info('Computing ground truth tensors') # log

    # if the pretty_midi file is empty, return immediately (loop will fail otherwise)
    if not configs.dataset.midis_exist:
        # make empty torch tensors to init a Notelabels object with
        blank = torch.empty(1,1)

        # exit the function safely
        return NoteLabels(activation_matrix=blank, onset_matrix=blank, velocity_matrix=blank)

    # otherwise, the midi object will work like normal

    # get data to count frames and make matrices
    duration = midi.get_end_time()
    
    sr = configs.dataset.sample_rate
    ws = configs.dataset.transform.window_size
    hl = configs.dataset.transform.hop_length
    frame_count = librosa.time_to_frames(times=duration, hop_length=hl, n_fft=ws, sr=sr)
    logging.info(f'frame count: {frame_count}') # log

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
    vm = torch.from_numpy(velocity_matrix)
    om = torch.from_numpy(onset_matrix)

    ret = NoteLabels(activation_matrix=am, onset_matrix=om, velocity_matrix=vm)
    return ret

def split_pretty_midi(pm: pretty_midi.PrettyMIDI, segment_length=30.0):
    segments = []
    total_time = pm.get_end_time()

    tempos, tempo_times = pm.get_tempo_changes()
    assert len(tempos) == 1, "this function only works when there aren't any tempo changes"

    # iterate over split points
    start_time = 0.0
    while start_time < total_time:
        end_time = min(start_time + segment_length, total_time)
        
        new_pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_times[0] if tempo_times.size > 0 else 120)

        for inst in pm.instruments:
            new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)

            for note in inst.notes:
                if note.end > start_time and note.start < end_time:
                    # cut 
                    new_start = max(note.start, start_time) - start_time 
                    new_end = min(note.end, end_time) - start_time
                    new_inst.notes.append(pretty_midi.Note(
                        velocity=note.velocity, 
                        pitch=note.pitch,
                        start=new_start,
                        end=new_end
                    ))

            if new_inst.notes:
                new_pm.instruments.append(new_inst)

        segments.append(new_pm)
        start_time += segment_length

    return segments
