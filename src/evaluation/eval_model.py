# evaluation function: should be able to use the unseen test data to figure out error, confusion matrix, 
# and other important metrics, so I know how well my model is actually performing
from ..training.CheckpointManager import CheckpointManager
from ..model.AutomaticPianoTranscriptor import APT
from ..corpus.MyDataset import MyDataset
from torch.utils.data import DataLoader 
from os.path import join
import torch
from ..training.train_utils import get_basic_loss_fxn, get_device 
from .calc_metrics import tuple_pianoroll_to_notes
import mir_eval

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

    # get device 
    device = get_device(configs)

    # evaluate 
    scores = dynamic_eval(model, eval_loader, criterion, device)

    # for all metrics and scores, print them out and save them 
    # TODO: LOGGING

# dynamic since it uses model, dataloader, criterion, and device which already exist as variables
# RETURN TYPE: tuple of a float which is the average loss output by the eval, and two lists of torch tensors,
# which have their tensors concatenated through the channel dimension in alphabetical order
# NEW RETURN TYPE: tuple of loss, avg_prec, avg_recall, avg_f1
def dynamic_eval(model, eval_loader, criterion, device, fs, tolerance):
    model.eval() 
    sum_loss = 0.0
    sum_prec =  0.0
    sum_recall = 0.0
    sum_f1 = 0.0

    # add model to device if it is not on the right device
    model = model.to(device, non_blocking=True)
    
    with torch.no_grad():
         for inputs, labels in eval_loader:
             # send the inputs and labels to the gpu 
             inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

             # have the model make predictions
             outputs = model(inputs)

             # calculate the loss function: outputs vs labels
             loss = criterion(outputs, labels)
            
             sum_loss += loss.item()
             prec, recall, f1 = eval_single_batch(outputs, labels, fs, tolerance)
             sum_prec += prec
             sum_recall += recall
             sum_f1 += f1

             del inputs, labels, outputs, loss

    avg_loss = sum_loss / len(eval_loader)

    return avg_loss, sum_prec, sum_recall, sum_f1

def eval_single_batch(outputs, labels, fs, tolerance):
    sum_prec = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    batch_size = outputs.size()[0]

    for i in range(batch_size):
        # Convert ground truth and predictions to events
        ref_intervals, ref_pitches = tuple_pianoroll_to_notes(labels[i], fs)
        est_intervals, est_pitches = tuple_pianoroll_to_notes(outputs[i], fs)

        # cannot pass intervals of size 0 to mir_eval. If either of these are 0, add 0 for all metrics, 
        # unless both are 0, when you add 1 since it is a perfect job guessing no notes
        if ref_pitches.size == 0 and est_pitches.size == 0:
            sum_prec += 1 
            sum_recall += 1 
            sum_f1 += 1 
            continue

        if ref_pitches.size == 0 or est_pitches.size == 0:
            continue

    # Compute note-level scores with 50ms onset tolerance and offset matching
        precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=tolerance,
        offset_ratio=None       # None = don’t require offset matching (ignore linter complaints)
        )

        sum_prec += precision
        sum_recall += recall
        sum_f1 += f1

    return sum_prec / batch_size, sum_recall / batch_size, sum_f1 / batch_size
