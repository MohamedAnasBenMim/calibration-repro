"""Training loop (SGD + momentum, step learning-rate schedule as in the paper)."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from .data import Splits
from .metrics import evaluate_logits
from .models import build_model, count_params


@dataclass
class TrainConfig:
    name: str
    model: str = "resnet"
    model_kwargs: dict = field(default_factory=dict)
    epochs: int = 15
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0
    # Learning rate is multiplied by 0.1 at these fractions of training
    # (the paper's CIFAR schedule divides it by 10 at 50% and 75%).
    lr_milestones: tuple[float, ...] = (0.5, 0.75)
    seed: int = 0
    train_subset: int | None = None
    eval_test_every_epoch: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        d = {k: v for k, v in d.items() if k != "label"}  # "label" is display-only
        if "lr_milestones" in d:
            d["lr_milestones"] = tuple(d["lr_milestones"])
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 1000) -> np.ndarray:
    model.eval()
    out = [model(x[i : i + batch_size]) for i in range(0, len(x), batch_size)]
    return torch.cat(out).double().numpy()


@dataclass
class TrainResult:
    config: dict
    history: list[dict]
    val_logits: np.ndarray
    test_logits: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray
    n_params: int
    train_seconds: float


def train(cfg: TrainConfig, splits: Splits, log=print) -> TrainResult:
    set_seed(cfg.seed)
    model = build_model(cfg.model, **cfg.model_kwargs)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    milestones = sorted({max(1, round(m * cfg.epochs)) for m in cfg.lr_milestones})
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)

    x, y = splits.x_train, splits.y_train
    if cfg.train_subset is not None:
        x, y = x[: cfg.train_subset], y[: cfg.train_subset]
    gen = torch.Generator().manual_seed(cfg.seed)
    y_test = splits.y_test.numpy()
    history: list[dict] = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(x), generator=gen)
        tot_loss, tot_correct, n = 0.0, 0, 0
        for i in range(0, len(x), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb, yb = x[idx], y[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += (logits.argmax(1) == yb).sum().item()
            n += len(idx)
        sched.step()
        row = {
            "epoch": epoch,
            "lr": opt.param_groups[0]["lr"],
            "train_loss": tot_loss / n,
            "train_error": 100.0 * (1 - tot_correct / n),
        }
        if cfg.eval_test_every_epoch or epoch == cfg.epochs:
            m = evaluate_logits(predict_logits(model, splits.x_test), y_test)
            row.update({f"test_{k}": v for k, v in m.items()})
        history.append(row)
        log(
            f"[{cfg.name}] epoch {epoch}/{cfg.epochs} "
            + " ".join(f"{k}={v:.4g}" for k, v in row.items() if k != "epoch")
            + f" ({time.time() - t0:.0f}s)"
        )
    train_seconds = time.time() - t0
    return TrainResult(
        config=cfg.to_dict(),
        history=history,
        val_logits=predict_logits(model, splits.x_val),
        test_logits=predict_logits(model, splits.x_test),
        val_labels=splits.y_val.numpy(),
        test_labels=y_test,
        n_params=count_params(model),
        train_seconds=train_seconds,
    )
