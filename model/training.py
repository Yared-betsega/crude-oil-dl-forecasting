import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

EPOCHS     = 50
LR         = 1e-3
ALPHA      = 10

LOSS_CHOICES = ("mse", "mae")


def get_criterion(loss: str = "mse") -> nn.Module:
    """Return the loss criterion for *loss* ('mse' or 'mae')."""
    loss = loss.lower()
    if loss == "mae":
        return nn.L1Loss()
    if loss == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss '{loss}'. Choose from {LOSS_CHOICES}.")


class DirectionalMSELoss(nn.Module):
    def __init__(self, alpha: float = ALPHA):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target, prev):
        mse      = (pred - target) ** 2
        pred_dir = torch.sign(pred   - prev)
        true_dir = torch.sign(target - prev)
        wrong    = (pred_dir != true_dir).float()
        return (mse * (1.0 + self.alpha * wrong)).mean()


def train_step(model: nn.Module, loader: DataLoader,
               epochs: int = EPOCHS, lr: float = LR,
               loss: str = "mse", verbose: bool = True) -> list:
    criterion = get_criterion(loss)
    log_key   = f"train_{loss.lower()}"
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history   = []
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            batch_loss = criterion(model(xb), yb)
            batch_loss.backward()
            optimizer.step()
            running += batch_loss.item() * len(xb)
        epoch_loss = running / len(loader.dataset)
        history.append({"epoch": epoch, log_key: round(epoch_loss, 6)})
        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  train {loss.upper()}: {epoch_loss:.4f}")
    return history


def train_step_directional(model: nn.Module, loader: DataLoader,
                            alpha: float = ALPHA, epochs: int = EPOCHS,
                            lr: float = LR, verbose: bool = True) -> list:
    criterion = DirectionalMSELoss(alpha=alpha)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history   = []
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for xb, yb, prevb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb, prevb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        epoch_loss = running / len(loader.dataset)
        history.append({"epoch": epoch, "dir_loss": round(epoch_loss, 6)})
        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  dir-loss: {epoch_loss:.4f}")
    return history
