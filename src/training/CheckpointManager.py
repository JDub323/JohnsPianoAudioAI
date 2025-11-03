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
import torch
import os
from ..corpus.utils import get_user_confirmation

class CheckpointManager():
    def __init__(self, configs):
        self.checkpoint_dir = configs.training.checkpoint_dir 
        self.run_name = configs.experiment.name

    def load_newest_checkpoint(self):
        # find newest checkpoint. if no checkpoints, throw error
        # newest is defined as the most recently modified
        pattern = os.path.join(self.checkpoint_dir, "*.pt")
        checkpoint_paths = glob.glob(pattern)

        if not checkpoint_paths:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

        # get newest by modification time
        newest_checkpoint_path = max(checkpoint_paths, key=os.path.getmtime)

        permission = get_user_confirmation(prompt=f"Load checkpoint {newest_checkpoint_path}?")

        if permission:
            print(f"loading checkpoint: {newest_checkpoint_path}")
            return self.load_checkpoint(newest_checkpoint_path)
        else:
            raise PermissionError("No permission to load checkpoint.")

    def load_checkpoint(self, directory: str) -> dict:
        checkpoint = torch.load(directory)
        logging.info(f"Loading checkpoint: {directory}")
        return checkpoint

    def save_checkpoint(self, epoch: int, model, optimizer, scheduler, loss, best_f1, global_step):
        if scheduler != None:
            schedule_dict = scheduler.state_dict()
        else:
            schedule_dict = None

        checkpoint = {
          "epoch": epoch,
          "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
          "scheduler_state_dict": schedule_dict,
          "loss": loss,
          "global_step": global_step,
          "best_f1": best_f1
        }

        chkpt = os.path.join(self.checkpoint_dir, self.run_name + f"_E{epoch:03d}")

        # handle repeated checkpoint names
        if os.path.exists(chkpt + ".pt"):
            i = 0
            while True:
                chkpt_new = chkpt + f"_({i})" + ".pt"
                if not os.path.exists(chkpt_new): break
                i += 1

            chkpt = chkpt_new
        else: 
            chkpt += ".pt"

        torch.save(checkpoint, chkpt)


