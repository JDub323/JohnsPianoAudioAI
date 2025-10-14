from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from ..evaluation import calc_metrics
import time
import psutil
import GPUtil
import os 

# class written by chatgpt, edited by me
class TrainingLogger:
    # need all of these vars on init because I need to be able to run the logger from any checkpoint
    # TODO: make sure checkpoints save all this data. Consider making a struct
    def __init__(self, configs, epoch, global_step, best_val_loss):
        # init the var "self.base_dir" and make directories for the logs, plots, and predicitons for the base dir
        self.set_tensorboard_base_dir(configs)

        self.writer = SummaryWriter(log_dir=f"{self.base_dir}/logs")
        self.global_step = global_step # total number of optimizer steps since the start of training
        self.epoch = epoch
        self.best_val_loss = best_val_loss
        self.num_labels = 88 # there are 88 piano keys. This will never change in the project

        self.start_time = time.time()

    def set_tensorboard_base_dir(self, configs, run_name: str | None = None):
        if (run_name == None): 
            now = datetime.now()
            formatted = now.strftime("%Y-%m-%d_%H-%M-%S")
            run_name = "run_" + formatted

        self.base_dir = os.path.join(str(configs.training.logging_dir), run_name)
        os.makedirs(f"{self.base_dir}", exist_ok=False)
        os.makedirs(f"{self.base_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.base_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.base_dir}/predictions", exist_ok=True)

    def step(self, loss: torch.Tensor, lr: float, grad_norm: float):
        """Log per training step."""
        self.global_step += 1
        self.writer.add_scalar("Train/Loss", loss.item(), self.global_step)
        self.writer.add_scalar("Train/LR", lr, self.global_step)
        self.writer.add_scalar("Train/GradNorm", grad_norm, self.global_step)

        # GPU usage 
        gpus = GPUtil.getGPUs()
        if gpus:
            self.writer.add_scalar("Resources/GPU_Mem_MB", gpus[0].memoryUsed, self.global_step)

    def log_epoch_metrics(self, val_loss, y_true, y_pred, tolerance, fs):
        """Log per epoch metrics (validation, accuracy, F1, etc.)."""
        self.epoch += 1
        self.writer.add_scalar("Val/Loss", val_loss.item(), self.epoch)

        # Compute note-wise/multilabel F1
        prec, recall, f1 = calc_metrics.get_prec_recall_f1(y_true, y_pred, tolerance, fs)
        self.writer.add_scalar("Val/F1", f1, self.epoch)
        self.writer.add_scalar("Val/recall", recall, self.epoch)
        self.writer.add_scalar("Val/precision", prec, self.epoch)

        # Track best model
        improved = False
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            improved = True
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            improved = True

    def log_time(self):
        elapsed = time.time() - self.start_time
        self.writer.add_scalar("Time/TotalSeconds", elapsed, self.global_step)

    def log_histograms(self, model: torch.nn.Module):
        """Optional: log parameter & gradient histograms."""
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"Weights/{name}", param.data.cpu().numpy(), self.global_step)
            if param.grad is not None:
                self.writer.add_histogram(f"Grads/{name}", param.grad.cpu().numpy(), self.global_step)

    def close(self):
        self.writer.close()
