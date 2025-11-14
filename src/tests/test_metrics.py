import torch
from ..corpus.datatypes import ProcessedAudioSegment, NoteLabels
from ..evaluation.eval_model import clean_pianoroll
from .utils import plot_outputs

def main():
    torch.serialization.add_safe_globals([ProcessedAudioSegment, NoteLabels])

    # outputs have dimension (20, 64, 6, 88, 157)
    outputs = torch.load("data/data_for_testing/example_outputs.pt")
    labels = outputs[:,:,:3] # labels are first 3
    outputs = outputs[:,:,3:] # outputs are second 3

    index = 30
    display_element(labels, index)
    display_element(outputs, index)


def display_element(outputs, index: int):
    # outputs should be torch tensor of shape (20, 64, 3, 88, T=157)
    disp = outputs[index//20, index%64]

    # disp should be torch tensor of shape (3, 88, T=157)
    # clean up for calculation of f1
    outputs_clean = clean_pianoroll(disp, clean_velo=True)
    plot_outputs(outputs_clean)

if __name__ == '__main__':
    main()
