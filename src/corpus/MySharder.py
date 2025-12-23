# IMPORTANT STATS RELATED TO SHARDING:
# let sample mean one 5 second spectrogram segment
# songs = 50 (~3.92 % of the total database size)
# total songs in db = 1277
# total samples = 5536 
# total memory, processed = .757 GB
# estimated memory, unprocessed = 7 GB

# ~0.3 GB per shard at 2000 samples per shard (with 5 second samples, 88 freq bins, hop length = 512)
# average samples per song = 110 (meaning each song averages 550 seconds, or about 9 minutes)
# memory per song after processing = ~16 mb
# projected total processed data memory = 20 GB (from about 200GB)
# average time to process one song = appx 6 seconds (see if I can speed this up)
# processing ratio (seconds to process / seconds audio) = ~ 1/92
# memory ratio (gb data processed / gb data unprocessed) = ~ 1/10
# estimated total time to process = 2 hours 7 minutes # at 6 seconds per song, previously had slowdown after some processing


from torch import save
import pandas as pd
import logging
from os.path import join
from .datatypes import ProcessedAudioSegment
from .utils import flatten_dict

TRAIN = 0
VALID8 = 1
TEST = 2
SPLIT_DICT = {
        'test': TEST,
        'train': TRAIN, 
        'validation': VALID8,
        }

class MySharder():
    """
    Sharder to control I/O operations when torch vectors are made. Try not to be too inefficient by 
    waiting until the end to make the list of torch tensors one big torch tensor 

    """
# TODO: save as a single torch.tensor rather than nested dictionaries of torch.tensors 

    def __init__(self, configs):
        # make a stack and a max value before emptying the stack
        self.max_value = configs.dataset.sharding.samples_per_shard

        # save export dir 
        self.export_dir = configs.dataset.export_root

        # save spreadsheet name 
        self.spreadsheet_name = configs.dataset.processed_csv_name

        # make list for the tensors and for the dataframe data
        self.matrix = [[],[],[]]
        self.dict_list = []

        # other variables to keep track of 
        self.shard_counts = [0,0,0]
        self.shard_indices = [0,0,0] 

    # need to call this on an object which has been initialized with:
    # sharder = MySharder.__new__(MySharder)
    def manual_create(self, max_value, export_dir, spreadsheet_name):
        """Initialize object manually without Hydra configs."""
        self.max_value = max_value
        self.export_dir = export_dir
        self.spreadsheet_name = spreadsheet_name

        # Reset working data structures
        self.matrix = [[], [], []]
        self.dict_list = []

        self.shard_counts = [0, 0, 0]
        self.shard_indices = [0, 0, 0]
        

    def push(self, seg: ProcessedAudioSegment, name: str, split: str, year: int, augs):
        split_list = self.matrix[SPLIT_DICT[split]]

        # add the next element to the stack
        split_list.append(seg)
        # update the dataframe
        self._add_segment_to_df(name, split, year, augs)

        # update index inside shard
        self.shard_indices[SPLIT_DICT[split]] += 1

        # call upload if the element count is greater than or equal to the max value
        if len(split_list) >= self.max_value:
            self._upload_shard(split)
            split_list.clear() # remove all elements from split list after usage
            

    def _upload_shard(self, split: str):
        split_enum = SPLIT_DICT[split]

        # upload the shard
        file = join(self.export_dir, self._calc_filename(split))
        save(self.matrix[split_enum], f=file)
        
        # update the variables 
        self.shard_indices[split_enum] = 0 
        self.shard_counts[split_enum] += 1

    def done_adding_data(self):
        self._force_all_uploads()
        self._upload_df()

    # calculates the shard filename
    def _calc_filename(self, split: str):
        split_enum = SPLIT_DICT[split]
        shard_num = self.shard_counts[split_enum]
        filename = f"{split}_shard{shard_num:03d}.pt"

        # add in the path to the correct directory
        return join(split, filename)

    def _force_all_uploads(self):
        for key, _ in SPLIT_DICT.items():
            self._upload_shard(key)

    # uploads 3 dataframes in the test, train, and validation directories respectively
    def _upload_df(self):
        # hard-coded upload for spreadsheet name
        splits = ['train', 'test', 'validation']

        # save the dataframe
        logging.info('Saving dataframes') # log
        for split in splits:
            loc = join(self.export_dir, split, self.spreadsheet_name)
            df = pd.DataFrame(self.dict_list).reset_index(drop=True)
            df_proc = self.process_df(df, split)
            df_proc.to_csv(loc)

    def process_df(self, df: pd.DataFrame, split: str):
        # set max str length. Any string longer is not worth keeping, since it likely has tons of useless paths
        max_str_len = 20

        # remove all rows which are not in split 'split'
        df = pd.DataFrame(df[df["split"] == split].copy())

        # remove all cols with excessively long strings
        drop_cols = []
        for col in df.columns:
            if df[col].dtype == "object":  # likely strings
                max_len = df[col].astype(str).map(len).max()
                if max_len > max_str_len:
                    drop_cols.append(col)

        ret = df.drop(columns=drop_cols)

        # return the new df
        return ret 


    def _add_segment_to_df(self, name: str, split: str, year: int, augs):
        # to handle the augmentations, I need to flatten the dictionary, then merge the dictionaries
        flat_augs = flatten_dict(augs)
        
        new_row = {
                'split': split,
                'filename': name,
                'shard_number': self.shard_counts[SPLIT_DICT[split]],
                'og_shard': self.shard_counts[SPLIT_DICT[split]],
                'shard_index': self.shard_indices[SPLIT_DICT[split]],
                'year': year,
                **flat_augs # merges the dictionaries flat_augs and new_row
                }

        self.dict_list.append(new_row)

