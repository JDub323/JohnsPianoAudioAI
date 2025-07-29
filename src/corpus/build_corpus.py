import argparse

def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument()

if __name__ == 'main':
    # process the command line arguments and add them to the configs

    # split the data into training data, validation, and evaluation data 
    # on the csv, using the seed (same rng every time)
    split_dataset()

    # preprocess data: convert it all from .wav to whatever spectrogram representation I want.
    # also consider adding noise to the data which could be from artifacts from an iphone.
    preprocess_data()

    # If my U-net is to make low-quality audio high-quality, this is an important step
    # This can wait until later though.. maybe
    # adding noise to .wav files should be done right before the .wav is passed through the model. 
    # This is to save memory
    if some_bool:
        add_noise()

    save_data_split()
