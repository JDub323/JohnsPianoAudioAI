from typing import Tuple
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch.optim as optim
from torch.nn import MSELoss, Module, BCELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging 
import torch.cuda

def get_optimizer(optimizer: str, model: Module):
    # TODO: add changes in optimizer params through updating the configs file
    if (optimizer == "Adam"):
        return optim.Adam(model.parameters())
    elif (optimizer == "SGD"):
        return optim.SGD(model.parameters())
    else:
        raise ValueError("Invalid optimizer type")

def get_basic_loss_fxn(loss_func: str):
    # TODO: same as above
    if (loss_func == "BCE"):
        return BCELoss()
    elif (loss_func == "BCElogits"):
        return BCEWithLogitsLoss()
    elif (loss_func == "MSE"):
        return MSELoss()
    else:
        raise ValueError("Invalid loss function")

# returns a bool too, true if the optimizer is step (epoch) based
def get_scheduler(scheduler: str, optimizer, num_epochs):
    # TODO: same as above 
    if (scheduler == "cosine"):
        return CosineAnnealingLR(optimizer, T_max=num_epochs), True
    else:
        raise ValueError("Invalid scheduler")

def get_device(configs) -> str:
    if torch.cuda.is_available and configs.training.use_gpu:
        return "cuda"
    else:
        return "cpu"


def log_basic_info(epoch, running_loss, train_loader, running_acc, val_loss, val_acc):
    logging.info(f"Epoch {epoch+1} done. Train loss: {running_loss/len(train_loader):.4f}, Train acc: {running_acc/len(train_loader):.4f} | Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

