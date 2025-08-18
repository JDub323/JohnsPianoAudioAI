import omegaconf
import pandas as pd
import numpy as np
import pretty_midi
from librosa import power_to_db
from os import path, makedirs
import librosa
from tabulate import tabulate
import logging

from src.corpus.datatypes import ProcessedAudioSegment

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
    makedirs(configs.dataset.export_root, exist_ok=True)
    subfolders = ['train', 'test', 'validation']
    for sf in subfolders:
        dir_ = path.join(configs.dataset.export_root, sf)
        makedirs(dir_, exist_ok=True)

def passed_row_limit(configs, i: int) -> bool:
    lim = configs.dataset.row_limit
    if lim == -1: return False # if the limit is set to -1, it is unbound: should process all rows

    return lim <= i

def complex_to_log_spectrogram(spectro: np.ndarray) -> np.ndarray:
    return power_to_db(np.abs(spectro)).astype(np.float32)

def download_wav(configs, filename: str) -> np.ndarray:
    logging.info(f'Downloading {filename} audio') # log
    path_to_wav = path.join(configs.dataset.data_root, filename)
    use_mono = configs.dataset.num_channels == 1
    ret, _ = librosa.load(path_to_wav, sr=configs.dataset.sample_rate, mono=use_mono)
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
