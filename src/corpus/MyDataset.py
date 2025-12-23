# Dataset wrapper for pytorch to use to access data
from os import path
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from .utils import NoteLabel_to_tensor
from .datatypes import NoteLabels, ProcessedAudioSegment

class MyDataset(Dataset):
    def __init__(self, dir_str: str, csv_name: str):
        self.df = pd.read_csv(path.join(dir_str, csv_name))

        directory = Path(dir_str)
        Path_list = list(directory.glob("*.pt")) # only get .pt files
        self.files = sorted([str(f) for f in Path_list]) # convert Path to str
        
        # load the first shard
        self._load_shard(0)
        self._cached_shard_idx = 0 # this var is set in _load_shard but I thought i'd set it here explicitly

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            # download ProcessedAudioSegment object with torch
            row = self.df.iloc[idx]
            
            shard_idx, local_idx = int(row['shard_number']), int(row['shard_index'])

            shard = self._get_shard(shard_idx) # index of the shard different from index of the item trying to access
            item = shard[local_idx]

            # return a tuple rather than a ProcessedAudioSegment object
            # must convert the notelabel to a torch tensor object
            return item.model_input, NoteLabel_to_tensor(item.ground_truth)
        except Exception as e:
            print(f"[DEBUG] Error at shard {shard_idx}, local {local_idx}, global {idx}")
            print(e)
            raise

    def _get_shard(self, shard_idx) -> list[ProcessedAudioSegment]:
        if self._cached_shard_idx == shard_idx:
            return self._cached_shard

        self._load_shard(shard_idx)
        return self._cached_shard

    def _load_shard(self, shard) -> None:
        self._cached_shard = torch.load(
            self.files[shard],
            map_location="cpu",
            mmap=True,
        )
        self._cached_shard_idx = shard

