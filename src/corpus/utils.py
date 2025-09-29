import omegaconf
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
import pretty_midi
from librosa import load, onset, power_to_db, time_to_frames, feature, cqt
from os import path, makedirs
from tabulate import tabulate
import logging
import shutil 
import torch 
from .datatypes import NoteLabels
from pretty_midi.pretty_midi import PrettyMIDI
import torch.nn.functional as F

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
        print("Processed data directory exists and is not empty.")
        print(f"Do you want to delete directory: {configs.dataset.export_root}?")
        consent_to_delete = get_user_confirmation()
        
        if consent_to_delete:
            logging.info(f"Deleting old processed data directory: {configs.dataset.export_root}") # log
            shutil.rmtree(configs.dataset.export_root)
        else:
            logging.info(f"Shutting down corpus build.")
            exit(0)

def get_user_confirmation(prompt="Confirm? (y/n): "):
    """Prompt the user for a yes/no confirmation and return True/False."""
    while True:
        response = input(prompt).strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        else:
            print("Please enter 'y' or 'n'.")

# prints a raw audio augmentations dict to a nice format
def print_augs_full(augs) -> None:
    flat_dict = flatten_dict(augs)
    print(tabulate(flat_dict.items(), headers=['Key', 'Value'], tablefmt='fancy_grid'))

# convert a NoteLabels object to a torch.Tensor batched, with channel 0 as frame, 1 as onset, 2 as velocity
def NoteLabel_to_tensor(labels: NoteLabels) -> torch.Tensor:
    apred = labels.activation_matrix
    opred = labels.onset_matrix
    vpred = labels.velocity_matrix

    return torch.cat([apred, opred, vpred], dim=1)

# prints a raw audio augmentations dict to a nice format
def print_augs_fancy(augs, augpath: str) -> None:
    print("----------------------------------------------------")
    print()
    print("AUGMENTATIONS:")
    # loop through all augmentations 
    # remember that augs is a nested dictionary
    for index, augdict in enumerate(augs.items()):
        key, aug = augdict
        if (aug['should_apply']): # only print if the augmentation was applied

            # the best way I could think of to decide what to print out, based on the aug. I don't want all the info.
            if (index == 0):
                print(f"Pitch shifted by {aug['num_semitones']}")

            elif (index == 1):
                print("Speech Applied:")
                print(f"  snr_db = {aug['snr_db']}")
                print(f"  rms_db = {aug['rms_db']}")

                relative_path = drop_beginning_path(full=aug['noise_file_path'], prefix=augpath)
                print(f"  noise from = {relative_path}")

            elif (index == 2):
                print("Environment Applied:")
                print(f"  snr_db = {aug['snr_db']}")
                print(f"  rms_db = {aug['rms_db']}")

                relative_path = drop_beginning_path(full=aug['noise_file_path'], prefix=augpath)
                print(f"  noise from = {relative_path}")

            elif (index == 3):
                print("Room Impluse Response Applied:")

                relative_path = drop_beginning_path(full=aug['ir_file_path'], prefix=augpath)
                print(f"  noise from = {relative_path}")

            elif (index == 4):
                print("Stationary was applied")

            elif (index == 5):
                print("Device Impulse Response Applied:")

                relative_path = drop_beginning_path(full=aug['ir_file_path'], prefix=augpath)
                print(f"  noise from = {relative_path}")

            elif (index == 6):
                print("Clipping applied")

            else: print("WHAT HAPPENDUH")


def drop_beginning_path(full: str, prefix: str) -> str:
    fullpath = Path(full)
    prefixpath = Path(prefix)
    return str(fullpath.relative_to(prefixpath))

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
    ret, _ = load(path_to_wav, sr=configs.dataset.sample_rate, mono=use_mono)
    return ret

def download_midi(configs, filename: str) -> PrettyMIDI | None:
    if configs.dataset.midis_exist:
        logging.info('Downloading midi') # log
        path_to_midi = path.join(configs.dataset.data_root, filename)
        ret = pretty_midi.PrettyMIDI(path_to_midi)
        # if configs.verbose: print_midi(ret)
        # note that song_midi can either be a mono np.ndarray or stereo, depending on configs
    # make sure song midi is bound
    else: ret = None

    return ret

def calc_duration(configs, wav, midi) -> float:
    # midi time duration
    midi_duration = midi.get_end_time()
    wav_duration = len(wav) / configs.dataset.sample_rate

    # WHY BELOW IS COMMENTED OUT: if there are no notes pressed when the midi is supposed to end: that is, if there 
    # happen to be no notes at exactly 30 seconds, then the pretty midi object will decide that 
    # the correct duration is really 29.97 (or whatever that time was) seconds long, rather than including the 
    # extra bit of silence at the end

    # test to make sure the durations are the same length (allow for some floating point error)
    # if configs.dataset.midis_exist:
        # length_matches = abs(midi_duration - wav_duration) < 1e-6 # roughly matches
        # if not length_matches:
        #     print_midi(midi)
        # assert length_matches, f"midi length: {midi_duration} not equal to wav length: {wav_duration}"

    return wav_duration

def save_configs(configs) -> None:
    root = configs.dataset.export_root
    save_location = path.join(root, 'configs.yaml')
    omegaconf.OmegaConf.save(configs, save_location)
    logging.info('Configs saved!')

def get_input_vector(configs, wav: np.ndarray) -> torch.Tensor:
    # logging.info('Computing input tensor') # log

    # gather the configs 
    sr = configs.dataset.sample_rate
    hl = configs.dataset.transform.hop_length
    nb = configs.dataset.transform.n_bins
    fs = configs.dataset.transform.filter_scale
    wt = configs.dataset.transform.window_type
    ws = configs.dataset.transform.window_size
    s = configs.dataset.transform.scale
    fmin = configs.dataset.transform.f_min
    fmax = configs.dataset.transform.f_max

    method = configs.dataset.transform.input_representation

    # take the correct transform:
    # log-mel spectrogram
    if method == 'mel':
        # compute complex power spectrogram 
        C = feature.melspectrogram(y=wav, sr=sr, n_fft=ws, hop_length=hl, window=wt, n_mels=nb, fmin=fmin, fmax=fmax)

        # convert the complex, power spectrogram to log-mel
        ret = complex_to_log_spectrogram(spectro=C) 

    # constant q transform 
    elif method == 'cqt':
        # compute the constant q transform
        C = cqt(y=wav, sr=sr, hop_length=hl, n_bins=nb, filter_scale=fs, window=wt, scale=s, fmin=fmin)

        #fmax is actually not used, since it uses spacing defined by the bins per octave, fmin, and n_bins

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

def get_truth_tensor(configs, midi: pretty_midi.PrettyMIDI | None, frame_count: int) -> NoteLabels:
    logging.info('Computing ground truth tensors') # log

    # if the pretty_midi file is empty, return immediately (loop will fail otherwise)
    if not configs.dataset.midis_exist:
        # make empty torch tensor to init a Notelabels object with
        blank = torch.empty(1,1)

        # exit the function safely
        return NoteLabels(activation_matrix=blank, onset_matrix=blank, velocity_matrix=blank)

    # otherwise, the midi object will work like normal

    # get data to count frames and make matrices
    sr = configs.dataset.sample_rate
    hl = configs.dataset.transform.hop_length
    fs = sr / hl # frames per second = sample rate / hop length 
    
    n_keys = 88
    onset_matrix = np.zeros((n_keys, frame_count), dtype=bool)
    activation_matrix = np.zeros((n_keys, frame_count), dtype=bool)
    velocity_matrix = np.zeros((n_keys, frame_count), dtype=np.float32)

    # all piano notes are instrument 0
    if midi != None and midi.instruments: # check to make sure that there are any notes in the midi file
        for note in midi.instruments[0].notes:
            if note.pitch < 21 or note.pitch > 108:
                continue  # skip notes outside piano range

            pitch_index = note.pitch - 21
            onset_frame = int(np.round(note.start * fs))
            offset_frame = int(np.round(note.end * fs))

            if onset_frame < frame_count:
                onset_matrix[pitch_index, onset_frame] = True
            else: print('ERROR: DROPPED NOTE')

            activation_matrix[pitch_index, onset_frame:offset_frame] = True
            velocity = note.velocity / 127.0  # normalize
            velocity_matrix[pitch_index, onset_frame] = velocity

    # convert to torch tensors 
    am = torch.from_numpy(activation_matrix)
    vm = torch.from_numpy(velocity_matrix)
    om = torch.from_numpy(onset_matrix)

    ret = NoteLabels(activation_matrix=am, onset_matrix=om, velocity_matrix=vm)
    return ret

# takes an entire song of note labels and returns the labels, split
# the last label will be padded to be of the same size as the rest of the labels (size segment_frames)
def split_truth_tensor(labels: NoteLabels, segment_frames: int):
    ret: List[NoteLabels] = [] # array of NoteLabels

    # make array to collect all split arrays of each note label
    truth_tensor_count = len(labels.__dict__)
    zipper = [[] for _ in range(truth_tensor_count)] # where there is a list for each truth tensor
    for i, entry in enumerate(labels.__dict__.items()):
        _, tensor = entry

        n_frames = tensor.size(1) # where 1 is the time axis. Can't use a const var here, since time axis changes sometimes

        # split the tensor into an array of segments of length segment_frames
        segments = [tensor[:, i:i+segment_frames] for i in range(0, n_frames, segment_frames)]

        # pad the last segment
        last_segment_len = segments[-1].shape[1]
        segments[-1] = F.pad(segments[-1], (0, segment_frames - last_segment_len, 0, 0))

        zipper[i].extend(segments)


    zipped = list(zip(*zipper)) # take the transpose of the 2d array, since things should be grouped by index
    
    # now construct the notelabels object array from the zipper 
    # yes this is hard coded, I don't plan on scaling the number of arrays any time soon
    for group in zipped:
        # the activation, onset, and velocity were added in insertion order from the dataclass, 
        # so things will break if I do things in a different order, or I change the order of the dataclass...
        ret.append(NoteLabels(activation_matrix=group[0],
                              onset_matrix=group[1], 
                              velocity_matrix=group[2]))

    return ret


def calc_midi_frame_count(configs, duration: float | None = None):
    # find duration, fallback to the configs duration if it is not passed as arg (should be default)
    if (duration == None): duration = float(configs.dataset.segment_length) # the cast is so my ide is not mad at me
    
    sr = configs.dataset.sample_rate
    hl = configs.dataset.transform.hop_length
    frame_count = time_to_frames(times=duration, hop_length=hl, sr=sr) + 1 # add 1 so size matches .wav spec size
    logging.info(f'frame count: {frame_count}') # log

    return frame_count

# deprecated function. Now, instead of splitting a pretty midi then converting all to note labels, 
# I convert to note labels then split, since note labels are easier to split than vice versa
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
