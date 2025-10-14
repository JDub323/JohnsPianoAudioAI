
class EarlyStopper:
    def __init__(self, configs, best_score):
        self.patience = configs.training.early_stop.patience
        self.delta = configs.training.early_stop.delta
        self.best_score = None
        self.counter = 0
        self.early_stop = configs.training.early_stopping

    def __call__(self, val_loss) -> bool:
        # if I don't want to enable early stopping, always return false
        if not self.early_stop:
            return False

        # otherwise, if there is an increase greater than the min change 
        if self.best_score is None or val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
            return False

        # otherwise, increase counter and check if patience has run out
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            # otherwise, if patience has not run out, return false (don't stop training)
            else: return False

