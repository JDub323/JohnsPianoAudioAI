from pathlib import Path
from ..corpus.datatypes import ProcessedAudioSegment, NoteLabels
from ..training.train_utils import get_device
import torch
import hydra
from os.path import join
from torch.utils.data import DataLoader
from ..training.CheckpointManager import CheckpointManager
from ..model.AutomaticPianoTranscriptor import APT
from ..corpus.MyDataset import MyDataset

# download a random model and a random subset of validation data, and save the outputs. These example
# outputs can then be used in other unittests :)
this_dir = Path(__file__).parent
config_path = this_dir.parents[1] / "configs"

@hydra.main(config_path=str(config_path), config_name="config.yaml", version_base=None)
def main(configs):
    # allow the loading of my shards
    torch.serialization.add_safe_globals([ProcessedAudioSegment, NoteLabels])
    
    # search for a checkpoint
    print("making checkpoint")
    cm = CheckpointManager(configs) 
    cpt = cm.load_newest_checkpoint()

    # use model from newest checkpoint
    print("making model")
    model = APT()
    model.load_state_dict(cpt['model_state_dict'])

    # use dataloader to get a couple of data items 
    print("making dataloader")
    data_root = configs.dataset.export_root
    csv_name = configs.dataset.processed_csv_name
    training_dataset = MyDataset(join(data_root, "train"), csv_name)
    train_dl = DataLoader(training_dataset, batch_size=configs.training.batch_size, 
                          shuffle=False, # I will shuffle when I create the corpus, so I can use caches and not shuffle here
                          num_workers=4,
                          pin_memory=True,
                          persistent_workers=True) 
    train_iter = iter(train_dl)

    item_count = 20

    model.eval() # put model in eval mode
    device = get_device(configs)
    model.to(device) # put model on cuda

    # create the tensor which will be saved for this function :). it has that shape because it will be:
    # item_count items, then for each item concat by batch_size
    # for each item, there are 6 piano-roll representations saved: the first three representations are the standard 3 for 
    # the label (what is actually correct) and the second three are the three for the model's outputs
    ret = torch.empty(item_count, configs.training.batch_size, 6, 88, 157)

    print("starting training")
    with torch.no_grad():
        for i in range(item_count):
            print(f"cycle {i} begun")
            inputs, labels = next(train_iter) # get inputs and labels

            # add inputs and labels to gpu
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # forward pass
            outputs = model(inputs)

            # add data to the outputs tensor
            ret[i] = torch.cat((labels, outputs), dim=1).cpu()

    # save data to special dir
    print("saving data")
    save_dir = configs.dataset.testing_data_root
    torch.save(ret, join(save_dir, "example_outputs.pt"))
    print("done :)")


if __name__ == '__main__':
    main()


