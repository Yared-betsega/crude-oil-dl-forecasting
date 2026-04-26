import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

EPOCHS     = 50
LR         = 1e-3
ALPHA      = 10


class DirectionalMAELoss(nn.Module):
    def __init__(self, alpha: float = ALPHA):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target, prev):
        mae      = torch.abs(pred - target)
        pred_dir = torch.sign(pred   - prev)
        true_dir = torch.sign(target - prev)
        wrong    = (pred_dir != true_dir).float()
        return (mae * (1.0 + self.alpha * wrong)).mean()


def train_step(model: nn.Module, loader: DataLoader,
               epochs: int = EPOCHS, lr: float = LR,
               verbose: bool = True) -> list:
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history   = []
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        epoch_mae = running / len(loader.dataset)
        history.append({"epoch": epoch, "train_mae": round(epoch_mae, 6)})
        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  train MAE: {epoch_mae:.4f}")
    return history


def train_step_directional(model: nn.Module, loader: DataLoader,
                            alpha: float = ALPHA, epochs: int = EPOCHS,
                            lr: float = LR, verbose: bool = True) -> list:
    criterion = DirectionalMAELoss(alpha=alpha)
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
