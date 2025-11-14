import logging
import random, time
import omegaconf
from ..corpus.datatypes import NoteLabels
from src.corpus.build_corpus import build_corpus
import os 
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from ..corpus.utils import calc_midi_frame_count, download_midi, download_wav, get_raw_data_df, get_truth_tensor, split_pretty_midi, get_input_vector
import sounddevice as sd
import torch

def prep_spectrogram_test(self, regenerate_data: bool):
    # make sure I'm not plotting nmf as a spectrogram
    self.assertFalse(self.cfg.dataset.transform.input_representation == 'nmf', "nmf isn't a spectrogram")

    # generate some data if it isn't already there and I should 
    path = self.cfg.dataset.export_root
    no_data_exists = not (os.path.exists(path) and len(os.listdir(path)) > 0)

    if regenerate_data or no_data_exists:
        logging.info("Generating new data")
        print("generating new data")
        build_corpus(self.cfg)
    else: 
        logging.info("Skipping data generation")
        print('skipping data generation')

def plot_spec(spec, sr: int, filename=""):
    # Create a figure and axes
    plt.figure(figsize=(10, 4))

    # Display the log-mel spectrogram
    librosa.display.specshow(spec, x_axis='time', y_axis='mel', sr=sr)

    # Add a color bar
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{filename} Spectrogram. Size: {spec.shape}')
    plt.tight_layout()
    plt.show() 

def generate_rand_row_indices(count: int, max_val: int) -> list[int]:
    ret = []
    # Make a separate RNG
    rng = random.Random()

    # Seed it with time (different every run)
    rng.seed(time.time_ns())   # nanosecond resolution

    for _ in range(count):
        ret.append(rng.randint(0, max_val))

    return ret

# returns an array of raw wavs and an array of the indices from which they were generated, as a list
def get_raw_wavs(configs, segment_length: int, row_indices: list[int] | None = None) -> list[np.ndarray]:
    # download dataframe which has paths to raw audio files
    df = get_raw_data_df(configs)

    # don't want to keep typing this-v
    row_count = configs.dataset.row_limit
    
    # figure out what I should be looping over
    # get row indices: just get the first row_count
    if row_indices == None:
        # if I am meant to do every row, I have to figure out how many rows I have to do in this way
        if row_count == -1: 
            row_count = len(df)

        looper = list(range(row_count))

    else:
        looper = row_indices

    assert len(looper) < 100, "Using too many rows"

    # get all wavs
    ret = []
    for i in looper:
        # get the wav file
        wav = download_wav(configs, filename=df.loc[i, 'audio_filename'])

        # only use the first n seconds of audio. Note that this won't work if this is stereo
        wav = wav[:(segment_length * configs.dataset.sample_rate)]
        ret.append(wav)

    return ret

def download_cut_labels(configs, segment_length: int, row_indices: list[int] | None = None) -> list[NoteLabels]:
    # download dataframe which has paths to raw audio files
    df = get_raw_data_df(configs)

    # don't want to keep typing this-v
    row_count = configs.dataset.row_limit
   
    # figure out what I should be looping over
    # get row indices: just get the first row_count
    if row_indices == None:
        # if I am meant to do every row, I have to figure out how many rows I have to do in this way
        if row_count == -1: 
            row_count = len(df)

        looper = list(range(row_count))

    else:
        looper = row_indices

    assert len(looper) < 100, "Using too many rows"
    # get all midis 
    ret = []
    for i in looper:
        # get the midi file
        song_midi = download_midi(configs, filename=df.loc[i, 'midi_filename'])
        
        # ensure midi is not None
        if song_midi == None:
            ret.append(song_midi)
            print('ERROR: MIDI FILE DOES NOT EXIST.')
            continue
        
        midi_arr = split_pretty_midi(pm=song_midi, segment_length=segment_length)

        # only use the first 30 second segment in the song to make labels
        labels = get_truth_tensor(configs=configs, midi=midi_arr[0], frame_count=calc_midi_frame_count(configs=configs, midi=midi_arr[0]))
        
        ret.append(labels)

    return ret 

# plays an audio file, and displays the spectrogram and labels associated with that file (if applicable)
# if the graph is closed, the rest of the audio is skipped 
def play_and_display(configs, wav: np.ndarray, labels: NoteLabels, audio_filename=""):
    # play the audio
    sd.play(wav, configs.dataset.sample_rate)

    # graph the spectrogram and notelabels 
    spectrogram = get_input_vector(configs, wav)

    # yeah sorry about this
    plot_spec_and_notes(spec=spectrogram, onset_mat=labels.onset_matrix, down_mat=labels.activation_matrix, 
                        vel_mat=labels.velocity_matrix, sr=configs.dataset.sample_rate,
                        hop_length=configs.dataset.transform.hop_length, filename=audio_filename)

# AI generated function, modified a bit
def plot_spec_and_notes(spec, onset_mat, down_mat, vel_mat, sr: int, hop_length=512, filename=""):
    """
    Plot log-mel spectrogram + onset/down/velocity matrices stacked
    
    spec: 2D np.ndarray or torch.Tensor [n_mels, n_frames]
    onset_mat, down_mat, vel_mat: torch.Tensors [n_notes, n_frames]
    sr: sample rate
    hop_length: STFT hop length (for frame-to-time mapping)
    """
    # Convert tensors to numpy if needed
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    onset_mat = onset_mat.cpu().numpy()
    down_mat = down_mat.cpu().numpy()
    vel_mat = vel_mat.cpu().numpy()

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False, gridspec_kw={'height_ratios':[2,1,1,1]})

    # --- 1. Spectrogram ---
    img1 = librosa.display.specshow(spec, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length, ax=axes[0])
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    axes[0].set_title(f'{filename} Spectrogram. Size: {spec.shape}')

    # --- 2. Onset matrix ---
    img2 = axes[1].imshow(onset_mat, aspect='auto', origin='lower', interpolation='nearest')
    axes[1].set_title(f"Note Onset Matrix. Size: {onset_mat.shape}")
    fig.colorbar(img2, ax=axes[1])

    # --- 3. Note down (sustain) matrix ---
    img3 = axes[2].imshow(down_mat, aspect='auto', origin='lower', interpolation='nearest')
    axes[2].set_title(f"Note Activation Matrix. Size: {down_mat.shape}")
    fig.colorbar(img3, ax=axes[2])

    # --- 4. Velocity matrix ---
    img4 = axes[3].imshow(vel_mat, aspect='auto', origin='lower', interpolation='nearest')
    axes[3].set_title(f"Note Velocity Matrix. Size: {vel_mat.shape}")
    fig.colorbar(img4, ax=axes[3])

    # Align everything
    plt.tight_layout()
    plt.show() # plot

# can also be used to plot model labels in piano roll representation. copied code from AI generated function above
def plot_outputs(outputs, name: str=""):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().numpy()

    activations, onsets, velocities = outputs

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    img3 = axes[0].imshow(activations, aspect='auto', origin='lower', interpolation='nearest')
    axes[0].set_title(f"Note Activation Matrix. Size: {activations.shape}")
    fig.colorbar(img3, ax=axes[0])

    img2 = axes[1].imshow(onsets, aspect='auto', origin='lower', interpolation='nearest')
    axes[1].set_title(f"Note Onset Matrix. Size: {onsets.shape}")
    fig.colorbar(img2, ax=axes[1])

    # --- 4. Velocity matrix ---
    img4 = axes[2].imshow(velocities, aspect='auto', origin='lower', interpolation='nearest')
    axes[2].set_title(f"Note Velocity Matrix. Size: {velocities.shape}")
    fig.colorbar(img4, ax=axes[2])

    # Align everything
    plt.tight_layout()
    plt.show() # plot
