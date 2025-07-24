import argparse

def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument()

if __name__ == 'main':
    print() 
    # split the data into training data, validation, and evaluation data 
    # on the csv, using the seed (same rng every time)
    split_dataset()

    # preprocess data: convert it all from .wav to whatever spectrogram representation I want.
    # also consider adding noise to the data which could be from artifacts from an iphone.

    # If my U-net is to make low-quality audio high-quality, this is an important step

    # This can wait until later though.. maybe
