import pytest
import torch

from calrepro.models import build_model


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("lenet5", {}),
        ("resnet", {"depth": 8, "width": 4}),
        ("resnet", {"depth": 14, "width": 4, "bn": False}),
        ("convnet", {"width": 4, "bn": True}),
        ("convnet", {"width": 4, "bn": False}),
    ],
)
def test_models_output_logits(name, kwargs):
    model = build_model(name, **kwargs)
    out = model(torch.randn(3, 1, 28, 28))
    assert out.shape == (3, 10)


def test_no_batchnorm_when_disabled():
    model = build_model("resnet", depth=8, width=4, bn=False)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
