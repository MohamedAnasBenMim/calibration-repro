"""Post-hoc calibration methods from Guo et al. (2017), Section 4.

All calibrators share the interface ``fit(val_logits, val_labels)`` and
``predict_proba(logits) -> probs``. Binary methods (histogram binning,
isotonic regression) are extended to K classes one-vs-rest, and the K
calibrated scores are renormalized to sum to one, as described in the paper
(Section 4.2). The scaling methods (temperature, vector, matrix) are fitted by
minimizing the validation NLL with L-BFGS.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from .metrics import softmax


class Calibrator:
    name = "base"

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> Calibrator:  # pragma: no cover
        raise NotImplementedError

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class Uncalibrated(Calibrator):
    name = "Uncalibrated"

    def fit(self, logits, labels):
        return self

    def predict_proba(self, logits):
        return softmax(logits)


def _renormalize(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    scores = np.clip(scores, 0.0, None)
    s = scores.sum(1, keepdims=True)
    k = scores.shape[1]
    # If every class gets a zero score, fall back to the uniform distribution.
    return np.where(s > eps, scores / np.maximum(s, eps), 1.0 / k)


class HistogramBinning(Calibrator):
    """Zadrozny & Elkan (2001) histogram binning, one-vs-rest.

    For each class k, the uncalibrated probability p_k is put into one of
    ``n_bins`` equal-width bins and replaced by the fraction of validation
    samples of class k in that bin (the NLL/MSE-optimal constant). Empty bins
    keep their bin midpoint.
    """

    name = "Hist. Binning"

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def _bin(self, p: np.ndarray) -> np.ndarray:
        return np.clip(np.ceil(p * self.n_bins).astype(int) - 1, 0, self.n_bins - 1)

    def fit(self, logits, labels):
        probs = softmax(logits)
        k = probs.shape[1]
        mids = (np.arange(self.n_bins) + 0.5) / self.n_bins
        self.theta = np.tile(mids, (k, 1))
        for c in range(k):
            idx = self._bin(probs[:, c])
            y = (labels == c).astype(np.float64)
            cnt = np.bincount(idx, minlength=self.n_bins)
            pos = np.bincount(idx, weights=y, minlength=self.n_bins)
            nz = cnt > 0
            self.theta[c, nz] = pos[nz] / cnt[nz]
        return self

    def predict_proba(self, logits):
        probs = softmax(logits)
        out = np.empty_like(probs)
        for c in range(probs.shape[1]):
            out[:, c] = self.theta[c, self._bin(probs[:, c])]
        return _renormalize(out)


class IsotonicCalibration(Calibrator):
    """Zadrozny & Elkan (2002) isotonic regression, one-vs-rest (sklearn PAV)."""

    name = "Isotonic"

    def fit(self, logits, labels):
        probs = softmax(logits)
        self.models = []
        for c in range(probs.shape[1]):
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            ir.fit(probs[:, c], (labels == c).astype(np.float64))
            self.models.append(ir)
        return self

    def predict_proba(self, logits):
        probs = softmax(logits)
        out = np.stack([m.predict(probs[:, c]) for c, m in enumerate(self.models)], 1)
        return _renormalize(out)


def _fit_lbfgs(params: list[torch.Tensor], loss_fn, max_iter: int = 500) -> None:
    opt = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)


class TemperatureScaling(Calibrator):
    """softmax(z / T) with a single scalar T > 0 fitted on validation NLL.

    T is parameterized as exp(log_t) so it stays positive. Dividing all logits
    by the same positive constant cannot change their argmax, so accuracy is
    unchanged by construction.
    """

    name = "Temp. Scaling"

    def fit(self, logits, labels):
        z = torch.as_tensor(logits, dtype=torch.float64)
        y = torch.as_tensor(labels, dtype=torch.long)
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        _fit_lbfgs([log_t], lambda: torch.nn.functional.cross_entropy(z / log_t.exp(), y))
        self.temperature = float(log_t.exp().item())
        return self

    def predict_proba(self, logits):
        return softmax(np.asarray(logits, dtype=np.float64) / self.temperature)


class VectorScaling(Calibrator):
    """softmax(w * z + b) with per-class w, b (diagonal matrix scaling)."""

    name = "Vector Scaling"

    def fit(self, logits, labels):
        z = torch.as_tensor(logits, dtype=torch.float64)
        y = torch.as_tensor(labels, dtype=torch.long)
        k = z.shape[1]
        w = torch.ones(k, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(k, dtype=torch.float64, requires_grad=True)
        _fit_lbfgs([w, b], lambda: torch.nn.functional.cross_entropy(z * w + b, y))
        self.w, self.b = w.detach().numpy(), b.detach().numpy()
        return self

    def predict_proba(self, logits):
        return softmax(np.asarray(logits, dtype=np.float64) * self.w + self.b)


class MatrixScaling(Calibrator):
    """softmax(W z + b) with a full K x K matrix W (initialized to identity)."""

    name = "Matrix Scaling"

    def fit(self, logits, labels):
        z = torch.as_tensor(logits, dtype=torch.float64)
        y = torch.as_tensor(labels, dtype=torch.long)
        k = z.shape[1]
        w = torch.eye(k, dtype=torch.float64).requires_grad_(True)
        b = torch.zeros(k, dtype=torch.float64, requires_grad=True)
        _fit_lbfgs([w, b], lambda: torch.nn.functional.cross_entropy(z @ w.T + b, y))
        self.w, self.b = w.detach().numpy(), b.detach().numpy()
        return self

    def predict_proba(self, logits):
        return softmax(np.asarray(logits, dtype=np.float64) @ self.w.T + self.b)


def all_calibrators() -> list[Calibrator]:
    return [
        Uncalibrated(),
        HistogramBinning(),
        IsotonicCalibration(),
        TemperatureScaling(),
        VectorScaling(),
        MatrixScaling(),
    ]
