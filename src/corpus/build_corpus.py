from omegaconf import DictConfig
from src.corpus.utils import print_data
import pandas as pd
import os

def build_corpus(configs: DictConfig):
    # get the dataframe with all the files
    csv_path = os.path.join(configs.dataset.data_root, configs.dataset.relative_path_to_spreadsheet)
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
        # call a split function to return a list of lists of floats for the .wav file 
        placeholder_float_list = [[0,0], [1.3, 7.1]]

        # for each audio segment
        for vector in placeholder_float_list:
            # augment the segment audio (make it a lower quality)

            # compute the spectrogram or NMF 

            # calculate what the split should be 

            # save the spectrogram or NMF to wherever the split should be (use the seed for rng)

            # add the data to a dataframe which has metadata and filename

    # save the dataframe


    # also, save the split dataframes. Since they are split as csvs which point to certain options, 
    # it doesn't really matter if the preprocessed files are in the same folders or not
    # consider making them "not"

def clean_df(df: pd.DataFrame, configs: DictConfig) -> pd.DataFrame:
    df = df.drop(columns=['canonical_composer', 'canonical_title']) # consider dropping the recommended split
    df = df.dropna()
    return df

