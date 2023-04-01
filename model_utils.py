import torch


def r2_loss(output, target):
    """
        R2 loss function.

        output : torch.Tensor
    """
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, tolerance=5, min_delta=0):
        """
        Args:
            tolerance (int): How long to wait after last time validation loss improved.
                            Default: 5
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
        """
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, validation_loss):
        if (self.best_loss is None) or (validation_loss < self.best_loss - self.min_delta):
            self.best_loss = validation_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
                return True
