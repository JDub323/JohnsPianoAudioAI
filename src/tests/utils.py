import logging
from src.corpus.build_corpus import build_corpus
import os 
import matplotlib.pyplot as plt
import librosa.display

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
    plt.title(f'{filename} Log-Mel Spectrogram')
    plt.tight_layout()
    plt.show() 
