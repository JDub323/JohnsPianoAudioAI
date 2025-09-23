# evaluation function: should be able to use the unseen test data to figure out error, confusion matrix, 
# and other important metrics, so I know how well my model is actually performing
from ..training.CheckpointManager import CheckpointManager
from ..model.AutomaticPianoTranscriptor import APT
from ..corpus.MyDataset import MyDataset
from torch.utils.data import DataLoader 
from os.path import join
import torch
from ..training.train_utils import get_basic_loss_fxn 

def evaluate(configs, data_split:str) -> None:
    # import the checkpoint 
    checkpointer = CheckpointManager(configs) 
    checkpoint = checkpointer.load_newest_checkpoint() # error will be thrown if no checkpoint exists

    # load the model from the checkpoint, making sure to set to eval mode
    model = APT() # custom "AutomaticPianoTranscriptor" object
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # load evaluation dataset
    data_root = configs.dataset.export_root
    csv_name = configs.dataset.processed_csv_name
    eval_dataset = MyDataset(join(data_root, data_split), csv_name)
    eval_loader = DataLoader(eval_dataset, batch_size=configs.training.batch_size, shuffle=False) # can validation be put in order?

    # get loss function
    criterion = get_basic_loss_fxn(configs.training.loss_function)

    # evaluate 
    scores = evaluate_and_log(model, eval_loader, criterion)

    # for all metrics and scores, print them out and save them 
    # TODO: LOGGING

def evaluate_and_log(model, eval_loader, criterion):
    model.eval() 
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
         for inputs, labels in eval_loader:
            # have the model make predictions
            outputs = model(inputs)

            # calculate the loss function: outputs vs labels
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_preds.append(outputs.argmax(dim=1))
            all_labels.append(labels)


    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = running_loss / len(eval_loader)

    return avg_loss
