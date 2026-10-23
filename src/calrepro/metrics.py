"""Calibration metrics: ECE, MCE, NLL, error, and reliability-diagram statistics.

Definitions follow Guo et al. (2017), Section 2: predictions are grouped into
M equal-width confidence bins B_m = ((m-1)/M, m/M], and

    ECE = sum_m |B_m|/n * |acc(B_m) - conf(B_m)|
    MCE = max_m |acc(B_m) - conf(B_m)|   (over non-empty bins)

where confidence is the maximum predicted class probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


@dataclass
class BinStats:
    edges: np.ndarray  # (M+1,)
    counts: np.ndarray  # (M,)
    accuracy: np.ndarray  # (M,), NaN for empty bins
    confidence: np.ndarray  # (M,), NaN for empty bins


def bin_stats(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> BinStats:
    """Group predictions into equal-width bins ((m-1)/M, m/M].

    A confidence of exactly 0 is placed in the first bin.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins: index m such that edges[m] < c <= edges[m+1].
    idx = np.clip(np.ceil(confidences * n_bins).astype(int) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_acc = np.bincount(idx, weights=correct, minlength=n_bins)
    sum_conf = np.bincount(idx, weights=confidences, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(counts > 0, sum_acc / counts, np.nan)
        conf = np.where(counts > 0, sum_conf / counts, np.nan)
    return BinStats(edges, counts, acc, conf)


def ece_mce(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    s = bin_stats(confidences, correct, n_bins)
    mask = s.counts > 0
    gaps = np.abs(s.accuracy[mask] - s.confidence[mask])
    ece = float(np.sum(s.counts[mask] / s.counts.sum() * gaps))
    mce = float(gaps.max()) if gaps.size else 0.0
    return ece, mce


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    return ece_mce(conf, correct, n_bins)[0]


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    p = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(p, eps, 1.0))))


def evaluate_probs(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """Error (%), NLL, ECE (%), MCE (%), mean confidence (%) of a probability matrix."""
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = pred == labels
    ece, mce = ece_mce(conf, correct, n_bins)
    return {
        "error": 100.0 * float(1.0 - correct.mean()),
        "nll": nll(probs, labels),
        "ece": 100.0 * ece,
        "mce": 100.0 * mce,
        "avg_confidence": 100.0 * float(conf.mean()),
    }


def evaluate_logits(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    return evaluate_probs(softmax(logits), labels, n_bins)
