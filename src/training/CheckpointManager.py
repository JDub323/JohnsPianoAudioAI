# file houses an object which must:
    # clean up checkpoint directory on init
    # be able to load a model from a checkpoint
    # save a model at each checkpoint
    # only save best k models?
    # keep track of checkpoint number which I am currently on 
import torch
import logging 
import os
import glob

class CheckpointManager():
    def __init__(self, configs):
        self.checkpoint_dir = configs.training.checkpoint_dir 

    def load_newest_checkpoint(self):
        # find newest checkpoint. if no checkpoints, throw error
        # newest is defined as the most recently modified
        pattern = os.path.join(self.checkpoint_dir, "*.pth")
        checkpoint_paths = glob.glob(pattern)

        if not checkpoint_paths:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

        # get newest by modification time
        newest_checkpoint_path = max(checkpoint_paths, key=os.path.getmtime)

        return self.load_checkpoint(newest_checkpoint_path)

    def load_checkpoint(self, directory: str) -> dict:
        checkpoint = torch.load(directory)
        logging.info(f"Loading checkpoint: {directory}")
        return checkpoint

    def save_checkpoint(self, epoch: int, model, optimizer, scheduler, loss):
        checkpoint = {
          "epoch": epoch,
          "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
          "scheduler_state_dict": scheduler.state_dict(),  # if you use one
          "loss": loss,
          # (optional) anything else you want to track
        }
        torch.save(checkpoint, self.checkpoint_dir)
        


