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
    def __init__(self, configs):
        # make a stack and a max value before emptying the stack
        self.max_value = configs.dataset.sharding.samples_per_shard

        # save export dir 
        self.export_dir = configs.dataset.export_root

        # make list for the tensors and for the dataframe data
        self.matrix = [[],[],[]]
        self.dict_list = []

        # other variables to keep track of 
        self.shard_counts = [0,0,0]
        self.shard_indices = [0,0,0] 
        print("sharder initialized")


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
        filename = f"{split}_shard{shard_num: 03d}.pt"

        # add in the path to the correct directory
        return join(split, filename)

    def _force_all_uploads(self):
        print("Uploading all")
        for key, _ in SPLIT_DICT.items():
            self._upload_shard(key)

    def _upload_df(self):
        # hard-coded upload for spreadsheet name
        spreadsheet_name = "Processed_Maestro.csv"

        # save the dataframe
        logging.info('Saving dataframe') # log
        loc = join(self.export_dir, spreadsheet_name)
        df_proc = pd.DataFrame(self.dict_list).reset_index(drop=True)
        df_proc.to_csv(loc)

    def _add_segment_to_df(self, name: str, split: str, year: int, augs):
        # to handle the augmentations, I need to flatten the dictionary, then merge the dictionaries
        flat_augs = flatten_dict(augs)
        
        new_row = {
                'split': split,
                'filename': name,
                'shard_number': self.shard_counts[SPLIT_DICT[split]],
                'shard_index': self.shard_indices[SPLIT_DICT[split]],
                'year': year,
                **flat_augs # merges the dictionaries flat_augs and new_row
                }

        self.dict_list.append(new_row)

