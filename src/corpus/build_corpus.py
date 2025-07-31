from omegaconf import DictConfig
from src.corpus.utils import print_data
import pandas as pd
import os

def build_corpus(configs: DictConfig):
    # get the dataframe with all the files
    csv_path = os.path.join(configs.dataset.data_root, configs.dataset.relative_path_to_spreadsheet)
    df = pd.read_csv(csv_path)
    print_data(df)

    # clean up the dataframe (get rid of any null rows, unneeded cols, etc.)
    clean_df(df, configs)
    print_data(df)

    # preprocess data: convert it all from .wav to whatever spectrogram representation I want.
    # also adds noise to the data which could be from artifacts from an iphone, iff that is 
    # what configs say to do
    # If my U-net is to make low-quality audio high-quality, this is an important step
    # This can wait until later though.. maybe
    # this saves the processed data under the "processed" directory
    process_data(df, configs)

    # split the data into training data, validation, and evaluation data 
    # on the csv, using the seed (same rng every time)
    # also, save the split dataframes. Since they are split as csvs which point to certain options, 
    # it doesn't really matter if the preprocessed files are in the same folders or not
    split_and_save(df, configs)

def clean_df(df: pd.DataFrame, configs: DictConfig) -> None:
    df = df.dropna()
    df = df.drop(columns=['canonical_composer', 'canonical_title']) # consider dropping the recommended split

def process_data(df: pd.DataFrame, configs: DictConfig) -> None:
    
    print("make spectrogram image files here, potentially using logic in its own file")

def split_and_save(df: pd.DataFrame, configs: DictConfig) -> None:
    # maestro already gives recommended splits, so I will skip the step of splitting by piece
    # it is in the 'split' column of the df, with strings of 'train', 'validation', or 'test' 

    # split the pieces into small segments
    print("split and also save the splits too :)")
