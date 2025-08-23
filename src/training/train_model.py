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

from omegaconf import DictConfig

def train(configs: DictConfig, checkpoint_path: str) -> None:
    # make data loaders for training and validation

    # load model from checkpoint if possible/necessary
    
    # for epoch in epochs: 
        # for inputs, labels in data loader
            # optimizer.zero_grad()
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
            # log metrics

        # disable "learning"
        # for each batch in validation data
            # forward pass and record validation metrics
        # log the sum of validation metrics 
        # save a new checkpoint if improved

        # this can be wrapped in a function
        # print(f"Epoch {epoch+1} done. Train loss: {running_loss/len(train_loader):.4f}, Train acc: {running_acc/len(train_loader):.4f} | Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        # do early stopping
        # if early stopping enabled:
            # if the model performed better than the prev record + some minimum improvement value:
                # patience = 0 
            # else patience++
            # if patience is greater than max patience:
                # print(f"Early stopping at epoch {epoch+1}.")
                # break

    # the following two lines should be their own method, called "evaluate":
    # for each batch in test set 
        # forward pass and record test metrics

    # compute and log test metrics and sample predictions
    # save final model. Final model can now be used to make predictions in real time

    return
