# Dataset wrapper for pytorch to use to access data
import torch
from torch.utils.data import Dataset
from pathlib import Path

from .datatypes import NoteLabels

class MyDataset(Dataset):
    def __init__(self, dir_str: str):
        directory = Path(dir_str)
        Path_list = list(directory.glob("*.pt")) # only get .pt files
        self.files = [str(f) for f in Path_list] # convert Path to str

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> tuple[torch.Tensor, NoteLabels]:
        # download ProcessedAudioSegment object with torch
        item = torch.load(self.files[idx])

        # return a tuple rather than a ProcessedAudioSegment object
        return item.model_input, item.ground_truth

