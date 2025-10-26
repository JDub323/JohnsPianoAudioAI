from ..evaluation.calc_metrics import tuple_pianoroll_to_notes
import os
import torch
from ..corpus.utils import NoteLabel_to_tensor
from ..corpus.datatypes import ProcessedAudioSegment, NoteLabels

def test_pianoroll_to_notes(data_dir: str = "./data/processed/test/"):
    data = None
    torch.serialization.add_safe_globals([ProcessedAudioSegment, NoteLabels])
    # search for some processed audio segment with a piano-roll torch.tensor dictionary 
    for filename in os.listdir(data_dir):
        if filename.endswith(".pt"):
            # download it 
            filepath = os.path.join(data_dir, filename)
            print("Opening:", filepath)
            data = torch.load(filepath)
            print("Loaded:", type(data))

    # get the piano roll 
    if not data:
        raise ValueError

    mytensor = NoteLabel_to_tensor(data[5].ground_truth) # data is a list, just take arbitrary element

    # send it to pianoroll_to_notes
    tuple_pianoroll_to_notes(mytensor, 44100./512)

if __name__ == '__main__':
    test_pianoroll_to_notes()
