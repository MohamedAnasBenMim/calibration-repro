"""Model zoo: an "old-style" LeNet-5 and "modern" CIFAR-style ResNets / ConvNets.

All models take 1x28x28 inputs (Fashion-MNIST) and return logits.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LeNet5(nn.Module):
    """LeNet-5 (LeCun et al., 1998), ReLU + max-pooling variant, no normalization."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 28 -> 28 -> pool 14
        self.conv2 = nn.Conv2d(6, 16, 5)  # 14 -> 10 -> pool 5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def _norm(c: int, bn: bool) -> nn.Module:
    return nn.BatchNorm2d(c) if bn else nn.Identity()


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, bn: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=not bn)
        self.bn1 = _norm(cout, bn)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=not bn)
        self.bn2 = _norm(cout, bn)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=not bn), _norm(cout, bn))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    """CIFAR-style ResNet (He et al., 2016) with depth 6n+2 and three stages.

    ``width`` is the number of filters of the first stage (16 in the paper's
    CIFAR ResNets); the stages use width, 2*width, 4*width filters.
    """

    def __init__(self, depth: int = 8, width: int = 16, bn: bool = True, num_classes: int = 10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2"
        n = (depth - 2) // 6
        self.conv = nn.Conv2d(1, width, 3, 1, 1, bias=not bn)
        self.bn = _norm(width, bn)
        layers = []
        cin = width
        for stage, mult in enumerate((1, 2, 4)):
            cout = width * mult
            for i in range(n):
                stride = 2 if (stage > 0 and i == 0) else 1
                layers.append(BasicBlock(cin, cout, stride, bn))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class ConvNet(nn.Module):
    """Plain 6-layer ConvNet (as in the paper's batch-norm experiment, scaled down).

    Five 3x3 conv layers + one linear layer, optional BatchNorm after each conv.
    """

    def __init__(self, width: int = 32, bn: bool = True, num_classes: int = 10):
        super().__init__()
        chans = [1, width, width, 2 * width, 2 * width, 4 * width]
        strides = [1, 1, 2, 1, 2]
        layers: list[nn.Module] = []
        for cin, cout, s in zip(chans[:-1], chans[1:], strides, strict=True):
            layers += [nn.Conv2d(cin, cout, 3, s, 1, bias=not bn), _norm(cout, bn), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(chans[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "lenet5":
        return LeNet5()
    if name == "resnet":
        return ResNet(**kwargs)
    if name == "convnet":
        return ConvNet(**kwargs)
    raise ValueError(f"unknown model {name!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
