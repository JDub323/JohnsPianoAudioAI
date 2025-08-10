import omegaconf
import pandas as pd
import numpy as np
import pretty_midi
from librosa import power_to_db
from os import path, makedirs
import mido
from io import BytesIO
import librosa
# TODO install with pip
# from tabulate import tabulate

def print_data(df: pd.DataFrame) -> None:
    print("Data: ")
    print(df)
    # Get basic info (useful for checking data types, non-null counts)
    print("\nDataFrame Info:")
    df.info()

def print_midi(midi) -> None:
    print('MIDI:')
    print(midi)

    note_rows = []
    headers = ['Instrument', 'Pitch (num)', 'Pitch (name)', 'Start (s rounded)', 'End (s rounded)', 'Velocity']
    print(headers)
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
            print(note_rows[-1])

    note_rows.sort(key=lambda x: x[3])

    # print(tabulate(note_rows, headers=headers, tablefmt='fancy_grid'))

def add_processed_data_folders(configs):
    makedirs(configs.dataset.export_root, exist_ok=True)
    subfolders = ['train', 'test', 'validation']
    for sf in subfolders:
        dir_ = path.join(configs.dataset.export_root, sf)
        makedirs(dir_, exist_ok=True)


def complex_to_log_spectrogram(spectro: np.ndarray) -> np.ndarray:
    return power_to_db(np.abs(spectro)).astype(np.float32)

def download_wav(configs, filename: str) -> np.ndarray:
    print(f'Downloading {filename} audio')
    path_to_wav = path.join(configs.dataset.data_root, filename)
    use_mono = configs.dataset.num_channels == 1
    ret, _ = librosa.load(path_to_wav, sr=configs.dataset.sample_rate, mono=use_mono)
    return ret

def calc_duration(wav, midi) -> float:
    ret = midi.get_end_time()
    # TODO: make sure that the midi and wav files are the same length
    return ret

def save_configs(configs) -> None:
    root = configs.dataset.export_root
    save_location = path.join(root, 'configs.yaml')
    omegaconf.OmegaConf.save(configs, save_location)
    print('Configs saved!')

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
