# objects which need to be made/will be used:
# CheckpointManager: used to log checkpoints. Saves model and metrics related to it in the models dir, 
    # saved as ckpt_00X.pt 
# Optimizer: applies gradients to weights and biases of model
# Scheduler: decides what the "learning rate" of the optimizer should be 
# Scaler: scales up gradients to avoid floating point rounding to 0 for small numbers, scaled back down before 
    # being applied. This is mostly useful if gradients are saved as float16 while w/b are float32, so I might not need this.
# ProgressBar: just used to show progress of training in terminal. Optional but nutty, can have chat make one
    # can use the TQDM python library for this 
# criterion: torch's nn loss function. used in a line like this: loss = criterion(logits, targets)
    # 'logits' are raw outputs from the model, before softmax/sigmoid. targets are the labels. 
    # loss is a scalar (dtype: torch.Tensor) which represents how bad my model did
# EMA: an "exponential moving average" of the model's weights. This means this thing makes another model, which lags 
    # behind the actual model, but in return is more stable and trains better. Not a priority, but should add eventually
# csv_logging object: a dataframe, but if the program crashes for whatever reason, not all is lost..? idk. It writes logs 
    # to a csv instead of normal logs. Probably more digestible in code to do this way, but not a priority addition
# logger object: THE MOST COMPLEX OF THESE, I have to keep track of: epoch, running loss, global step, and so much more.
# here is what chatGPT said about it:
# Training stats (per step or batch), 
# Loss
# Accuracy (or other task-specific metrics: F1, IoU, CER/WER for speech, etc.)
# Gradient norm (helps debug exploding gradients)
# Learning rate (from the scheduler)
# GPU memory usage (sometimes)
# Validation stats (per epoch or validation interval)
# Validation loss
# Validation accuracy/other metrics
# Best-so-far checkpoint metrics
# Training metadata
# Epoch number, global step
# Time per step/epoch (speed)
# Total training time so far
# Artifacts
# Model checkpoints (weights, optimizer state, scheduler state)
# Samples (e.g., spectrograms, predictions vs. targets, generated audio/images)
# Configuration used (hyperparameters, model definition, dataset info)
# Debug info (optional but helpful)
# Gradient histograms
# Weight histograms
# Learning curves
# Random seed, environment info (GPU type, library versions)
# chat also mentioned that there are some libraries which do a lot of this automatically, such as TensorBoard, 
# but I think I want a wrapper for that just so I don't have to worry about details on how to use the object within 
# this function, so it will still be a lot to do for a logger object

# TODO: make sure everything is on and stays on the same device (gpu?)

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .EarlyStopper import EarlyStopper
from . import train_utils

from .CheckpointManager import CheckpointManager
from ..model.AutomaticPianoTranscriptor import APT
import logging
from ..corpus.MyDataset import MyDataset
from os.path import join
from ..evaluation import eval_model

# TODO: make sure everything is on the gpu or something. IDK what is going on, all I know is that there are nasty bugs 
# everywhere here in this code, and I need to finish asap so I can find them all in time

def train(configs: DictConfig, checkpoint_path: str) -> None:
    # init some vars 
    num_epochs = configs.training.epochs

    # make data loaders for training and validation
    data_root = configs.dataset.export_root
    csv_name = configs.dataset.processed_csv_name
    training_dataset = MyDataset(join(data_root, "train"), csv_name)
    validation_dataset = MyDataset(join(data_root, "validation"), csv_name)
    train_dl = DataLoader(training_dataset, batch_size=configs.training.batch_size, shuffle=True)
    validation_dl = DataLoader(validation_dataset, batch_size=configs.training.batch_size, shuffle=False) # can validation be put in order?

    # load model from checkpoint if possible/necessary
    checkpointer = CheckpointManager(configs)

    checkpoint = None
    try:
        checkpoint = checkpointer.load_newest_checkpoint()
        checkpoint_found = True
    except:
        checkpoint_found = False

    model = APT() # custom "AutomaticPianoTranscriptor" object
    optimizer = train_utils.get_optimizer(configs.training.optimizer, model) # This order of things might cause bugs
    scheduler, epoch_based = train_utils.get_scheduler(configs.training.scheduler, optimizer, num_epochs)

    if checkpoint_found:
        assert checkpoint != None, "checkpoint found but not set"

        # load checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(state_dict=checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

    # set defaults if there was no checkpoint
    else:
        start_epoch = 0
        loss = 0 # TODO: see if this is a valid value for the loss

    # get loss function
    criterion = train_utils.get_basic_loss_fxn(configs.training.loss_function)

    # make an early-stopping object 
    early_stopper = EarlyStopper(configs)
    
    # for all epochs, for all iterations on the dataset
    for epoch in range(start_epoch, num_epochs): 
        # don't forget to put the model back into training mode before training
        model.train()

        for inputs, labels in train_dl:
            optimizer.zero_grad() # set gradient back to 0 (accumulates by default)
            outputs = model(inputs) # forward pass the input through the model
            loss = criterion(outputs, labels) # calculate loss
            loss.backward() # calculate negative gradient of error function
            optimizer.step() # apply the negative gradient (this is not exactly just addition, due to optimizer magic)
            # step the scheduler here if loss-based
            if not epoch_based:
                scheduler.step()

            # log metrics

        model.eval() # disable "learning" (this is also done within the function)
        # get the validation loss 
        val_loss = eval_model.evaluate_and_log(model, validation_dl, criterion) 

        # save a new checkpoint if improved
        checkpointer.save_checkpoint(epoch, model, optimizer, None, loss)

        # do early stopping
        if early_stopper(val_loss):
            logging.info(f"Early stopping at epoch {epoch+1}.")
            break

        # step scheduler here if epoch-based
        if epoch_based:
            scheduler.step()

    # get the testing dataset
    test_dataset = MyDataset(join(data_root, "test"), csv_name)
    test_dl = DataLoader(test_dataset, batch_size=configs.training.batch_size, shuffle=False)
    loss = eval_model.evaluate_and_log(model, test_dl, criterion)

    # compute and log test metrics and sample predictions
    # save final model. Final model can now be used to make predictions in real time

    return
