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

# MORE TESTING: I am starting the training program at 2:03. I am on the wsl file system, and I am not shuffling 
# (so all time spent accessing disc is not used, since the shard is cached)
# complete 100 training cycles every 20 SECONDS!!!!!!!!!!
# 300 cycles per minute!!!
# done with first epoch at 2:07 (including evaluation and logging). Logging takes longer than expected, f1 calculation
# seems very slow now.
# done with epoch 2 at 2:10
# at this pace, I will be done in about 5 hours. This is only around 2% of the dataset I am looping over
# done with epoch 5 at 2:22. slight slowdown it seems like, taking around 4 minutes instead of 3
# ended this test at 2:27 on epoch 7. Things slowed down further and disc usage was spiking, so i believe 
# there to be an error in how I am logging training cycle-level metrics. I will look into that at a further time

# it turns out I was logging things fine, but I had a ton of memory leaks, not taking things off of gpu (i didn't know...)
# NEW TEST: I changed some things so hopefully I have no more memory leaks. I will be tracking that, but I will also be timing
# whether chatGPT's optimization recommendations work
# I am now using a batch size of 64 btw. Now it takes around 2 minutes for 100 rows, 
# pre-optimal, post memory-safe test: 1 epoch, time = 4 min 30 sec
# post-GPT optimizations: 1 epoch, time = 1 min 10 sec, with full GPU usage (before, I would see spikes and waiting)
# both these tests include commented-out batch-level logging

# long test: start epoch 3 at 2:21. Fixed memory leak issues, so I am going to let my model run for a while.
# Going to go work out then come back. By 3:07, I FINISHED???. I only got 34 epochs in, since early stopping caused
# me to stop training, since I wasn't getting any better, which is embarrasing, since I think I am nowhere near being 
# accurate from what I can tell.

# new test in prob incoming: only got 22 cycles in before early stopping. Doesn't that mean it was better?


from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .MyLogger import TrainingLogger

from .EarlyStopper import EarlyStopper
from . import train_utils
from .ModelDebugger import ModelDebugger
from .CheckpointManager import CheckpointManager
from ..model.APT1 import APT1
import logging
from ..corpus.MyDataset import MyDataset
from os.path import join
from ..evaluation import eval_model
from .train_utils import get_device, get_grad_norm
import torch 
from src.corpus.datatypes import ProcessedAudioSegment, NoteLabels
from .loss_wrapper import CorrectedLossWrapper, LossWrapper, get_loss_wrapper

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
    print("making dataloaders")
    data_root = configs.dataset.export_root
    csv_name = configs.dataset.processed_csv_name
    training_dataset = MyDataset(join(data_root, "train"), csv_name)
    validation_dataset = MyDataset(join(data_root, "validation"), csv_name)
    train_dl = DataLoader(training_dataset, batch_size=configs.training.batch_size, 
                          shuffle=False, # I will shuffle when I create the corpus, so I can use caches and not shuffle here
                          num_workers=1, # TODO: don't hard code this
                          pin_memory=True,
                          # persistent_workers=True,
                          )
    validation_dl = DataLoader(validation_dataset, batch_size=configs.training.batch_size, 
                               shuffle=False,
                               num_workers=4,
                               pin_memory=True,
                               persistent_workers=True)

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
        print("No checkpoint to load. Starting from scratch.")

    # model = APT0(configs) # custom "AutomaticPianoTranscriptor" object
    # model = CorrectedAPT() # AI generated version
    model = APT1()

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
        best_f1 = 0
        global_step = 0

    # get loss function
    criterion = get_loss_wrapper(configs)
    print("got loss function")

    # make debugger which, depending on configs, saves/prints model memory usage, or does nothing
    mdb = ModelDebugger(configs) # pneumonic: model debugger
    print("made a model debugger")

    # make an early-stopping object 
    early_stopper = EarlyStopper(configs, best_score=best_loss)
    print("made early stopper")

    # make a logging object
    logger = TrainingLogger(configs, start_epoch, global_step, best_loss, best_f1)
    print("made a logger")

    # add model to device
    model = model.to(device, non_blocking=True)
    print(f"put model on device: {device}")
    
    # for all epochs, for all iterations on the dataset
    breakpoint()
    for epoch in range(start_epoch, num_epochs): 
        print(f"starting epoch {epoch} out of {num_epochs}")

        # don't forget to put the model back into training mode before training
        model.train()

        # start memory debug here 
        mdb.start_record_memory_history()

        for inputs, labels in train_dl:
            # add inputs and labels to gpu
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # set gradient back to 0 (accumulates by default)

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
            lr = scheduler.get_last_lr()[0]
            logger.step(loss, lr, grad_norm) # not a memory leak, my logger takes the torch tensor and calls loss.item()

            # check to see if I should stop recording memory leaks for the memory debugger
            mdb.check_cycle_limit()

        print("evaluating model")
        model.eval() # disable "learning" (this is also done within the function)
        # get the validation loss, y_pred, and y_true
        val_loss, prec, recall, f1 = eval_model.dynamic_eval(model, validation_dl, criterion, device, fs, tolerance) 
        best_f1 = max(f1, best_f1)

        print("logging epoch metrics")
        logger.log_epoch_metrics(val_loss, prec, recall, f1)

        print("saving checkpoint")
        # save a new checkpoint if improved
        checkpointer.save_checkpoint(epoch, model, optimizer, None, val_loss, best_f1, global_step)

        # do early stopping
        if early_stopper(val_loss):
            print(f"Early stopping at epoch {epoch}.")
            break

        # step scheduler here if epoch-based
        print("stepping scheduler")
        if epoch_based:
            scheduler.step()

        if configs.training.shuffle_during_training:
            print("shuffling training dataset in between epochs")
            pass

    # test the model on test dataset
    print("testing model")
    test_dataset = MyDataset(join(data_root, "test"), csv_name)
    test_dl = DataLoader(test_dataset, batch_size=configs.training.batch_size,
                               shuffle=False,
                               num_workers=4,
                               pin_memory=True,
                               persistent_workers=True)

    test_loss, test_prec, test_rec, test_f1 = eval_model.dynamic_eval(model, test_dl, criterion, device, fs, tolerance)

    # compute and log test metrics and sample predictions
    # Final model can now be used to make predictions in real time. you can get it from the checkpoint probably
    print("doing final logging")
    logger.log_test_metrics(test_loss, test_prec, test_rec, test_f1)
    logger.log_histograms(model)
    logger.log_model(model)
    logger.close()

    mdb.export_memory_snapshot()
    mdb.stop_record_memory_history()

    return
