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

# TESTING: I started the training program at 11:00, and fixed all bugs before then. takes about 6-7 minutes to start training.
# the longest portion of the startup is the imports, but making the optimizer takes a second. Obviously, training takes much
# longer than everything else. I think that my biggest bottleneck is actually disc I/O, since on task manager, it says I am using
# 1% of my GPU. I did 10 cycles in about 10 minutes, so it will take about 517 hours to run this entire program, or about 5 hours
# for one epoch with 100 rows processed
# at 11:32, I am at cycle 18
# at 12:20, I am at cycle 57
# at 12:29, I am at cycle 58 (closed laptop, program halted but didn't quit)
# at 2:52 am the next day, I am at cycle 136. I do not know how this happened.
# quit at cycle 141 at 3:00 am

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .MyLogger import TrainingLogger

from .EarlyStopper import EarlyStopper
from . import train_utils

from .CheckpointManager import CheckpointManager
from ..model.AutomaticPianoTranscriptor import APT
import logging
from ..corpus.MyDataset import MyDataset
from os.path import join
from ..evaluation import eval_model
from .train_utils import get_device, get_grad_norm
import torch 
from src.corpus.datatypes import ProcessedAudioSegment, NoteLabels
from .loss_wrapper import LossWrapper
import pdb

def train(configs: DictConfig) -> None:
    # get device
    print("getting device")
    device = get_device(configs)

    # init some vars 
    num_epochs = configs.training.epochs
    tolerance = configs.evaluation.tolerance
    fs = 1 / configs.dataset.transform.hop_length 

    # allow the loading of my shards
    torch.serialization.add_safe_globals([ProcessedAudioSegment, NoteLabels])

    # make data loaders for training and validation
    print("Dataloaders made")
    data_root = configs.dataset.export_root
    csv_name = configs.dataset.processed_csv_name
    training_dataset = MyDataset(join(data_root, "train"), csv_name)
    validation_dataset = MyDataset(join(data_root, "validation"), csv_name)
    train_dl = DataLoader(training_dataset, batch_size=configs.training.batch_size, shuffle=True)
    validation_dl = DataLoader(validation_dataset, batch_size=configs.training.batch_size, shuffle=False)

    # load model from checkpoint if possible/necessary
    checkpointer = CheckpointManager(configs)
    print("made checkpointer")

    checkpoint = None
    try:
        checkpoint = checkpointer.load_newest_checkpoint()
        checkpoint_found = True
        print("loaded checkpoint")
    except:
        checkpoint_found = False
        print("No checkpoint found. Starting from scratch")

    model = APT() # custom "AutomaticPianoTranscriptor" object
    print("made model")
    optimizer = train_utils.get_optimizer(configs.training.optimizer, model) # This order of things might cause bugs
    print("made optmizer")
    scheduler, epoch_based = train_utils.get_scheduler(configs.training.scheduler, optimizer, num_epochs)
    print("made scheduler")

    if checkpoint_found:
        assert checkpoint != None, "checkpoint found but not set"

        # load checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(state_dict=checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        global_step = checkpoint['global_step']
        best_f1 = checkpoint['best_f1']

    # set defaults if there was no checkpoint
    else:
        start_epoch = 0
        
        best_loss = float('inf')
        global_step = 0

    # get loss function
    criterion = LossWrapper(configs)
    print("got loss function")

    # make an early-stopping object 
    early_stopper = EarlyStopper(configs, best_score=best_loss)
    print("made early stopper")

    # make a logging object
    logger = TrainingLogger(configs, start_epoch, global_step, best_loss)
    print("made a logger")

    # add model to device
    model = model.to(device)
    print(f"put model on device: {device}")
    
    # for all epochs, for all iterations on the dataset
    for epoch in range(start_epoch, num_epochs): 
        print(f"starting epoch {epoch} out of {num_epochs}")
        # don't forget to put the model back into training mode before training
        model.train()

        for inputs, labels in train_dl:
            # add inputs and labels to gpu
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # set gradient back to 0 (accumulates by default)

            outputs = model(inputs) # forward pass the input through the model
            loss = criterion(outputs, labels) # calculate loss

            loss.backward() # calculate negative gradient of error function
            grad_norm = get_grad_norm(model) # calculation for logging

            optimizer.step() # apply the negative gradient (this is not exactly just addition, due to optimizer magic)
            global_step += 1 # increment the global step for each time the optimizer is called
            print(f"completed training cycle {global_step}")

            # step the scheduler here if loss-based
            if not epoch_based:
                scheduler.step()

            # log metrics
            logger.log_time()
            lr = scheduler.get_last_lr()[0]
            logger.step(loss, lr, grad_norm)
            logger.log_time()
            if global_step >= 5:
                break

        model.eval() # disable "learning" (this is also done within the function)
        # get the validation loss, y_pred, and y_true
        val_loss, y_true, y_pred = eval_model.dynamic_eval(model, validation_dl, criterion, device) 

        logger.log_epoch_metrics(val_loss, y_true, y_pred, tolerance, fs)

        # save a new checkpoint if improved
        # TODO: update loss to make it average, innstead of the most recent 
        checkpointer.save_checkpoint(epoch, model, optimizer, None, loss, global_step)

        # do early stopping
        if early_stopper(val_loss):
            logging.info(f"Early stopping at epoch {epoch+1}.")
            break

        # step scheduler here if epoch-based
        if epoch_based:
            scheduler.step()

    return 

    # test the model on test dataset
    test_dataset = MyDataset(join(data_root, "test"), csv_name)
    test_dl = DataLoader(test_dataset, batch_size=configs.training.batch_size, shuffle=False)
    test_loss, y_true, y_pred = eval_model.dynamic_eval(model, test_dl, criterion, device)

    # compute and log test metrics and sample predictions
    # Final model can now be used to make predictions in real time. you can get it from the checkpoint probably
    logger.log_epoch_metrics(test_loss, y_true, y_pred, tolerance, fs)
    logger.log_histograms(model)
    logger.close()

    return
