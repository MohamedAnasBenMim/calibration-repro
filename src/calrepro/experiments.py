"""Experiment drivers for the three claims of Guo et al. (2017).

Each driver reads a JSON config, writes metrics (JSON + CSV) and PNG figures
into an output directory, and never reads the test set for any fitting
decision (calibrators are fitted on the held-out validation split only).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .calibration import all_calibrators
from .data import load_splits
from .metrics import bootstrap_ece_ci, evaluate_logits, softmax
from .plots import factors_figure, reliability_figure, training_curves_figure
from .train import TrainConfig, TrainResult, train


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _round(d: dict, nd: int = 4) -> dict:
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in d.items()}


def _setup(cfg: dict) -> None:
    torch.set_num_threads(int(cfg.get("num_threads", 2)))


def _splits(cfg: dict):
    d = cfg.get("data", {})
    return load_splits(
        n_val=d.get("n_val", 5000),
        seed=d.get("split_seed", 0),
        test_subset=d.get("test_subset"),
    )


def _save_logits(path: Path, r: TrainResult) -> None:
    np.savez_compressed(
        path,
        val_logits=r.val_logits.astype(np.float32),
        val_labels=r.val_labels.astype(np.int64),
        test_logits=r.test_logits.astype(np.float32),
        test_labels=r.test_labels.astype(np.int64),
    )


# --------------------------------------------------------------------------- claim 1
def run_claim1(cfg: dict, out: Path) -> dict:
    """Train an old-style and a modern network; report calibration (Fig. 1, Fig. 3)."""
    _setup(cfg)
    out.mkdir(parents=True, exist_ok=True)
    splits = _splits(cfg)
    summary = {}
    for run in cfg["runs"]:
        tc = TrainConfig.from_dict(run)
        r = train(tc, splits)
        test_m = evaluate_logits(r.test_logits, r.test_labels)
        summary[tc.name] = {
            "label": run.get("label", tc.name),
            "n_params": r.n_params,
            "train_seconds": round(r.train_seconds, 1),
            "final_train_loss": round(r.history[-1]["train_loss"], 5),
            "final_train_error": round(r.history[-1]["train_error"], 3),
            **_round(test_m),
            "config": r.config,
        }
        _write_csv(out / f"history_{tc.name}.csv", [_round(h, 5) for h in r.history])
        _save_logits(out / f"logits_{tc.name}.npz", r)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    _write_csv(
        out / "metrics.csv",
        [{"model": k, **{kk: vv for kk, vv in v.items() if kk != "config"}} for k, v in summary.items()],
    )
    plot_claim1(cfg, out)
    return summary


def plot_claim1(cfg: dict, out: Path) -> None:
    """Redraw Fig. 1 and Fig. 3 from the saved logits / histories (no training)."""
    panels, histories = [], {}
    for run in cfg["runs"]:
        name, label = run["name"], run.get("label", run["name"])
        d = np.load(out / f"logits_{name}.npz")
        panels.append((label, softmax(d["test_logits"].astype(np.float64)), d["test_labels"]))
        with open(out / f"history_{name}.csv") as f:
            histories[label] = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    reliability_figure(panels, out / "fig1_reliability.png")
    training_curves_figure(histories, out / "fig3_nll_overfitting.png")


# --------------------------------------------------------------------------- claim 2
def run_claim2(cfg: dict, out: Path) -> dict:
    """Vary depth / width / batch norm / weight decay (Fig. 2, scaled down)."""
    _setup(cfg)
    out.mkdir(parents=True, exist_ok=True)
    splits = _splits(cfg)
    cache: dict[str, dict] = {}
    rows, groups = [], {}
    for g in cfg["groups"]:
        factor = g["factor"]
        groups[factor] = []
        for value in g["values"]:
            run = json.loads(json.dumps(cfg["base"]))
            run.update(g.get("base_overrides", {}))
            run.setdefault("model_kwargs", {})
            # A factor targets either a top-level training key or a model kwarg.
            if g["key"] in ("weight_decay", "epochs", "lr"):
                run[g["key"]] = value
            else:
                run["model_kwargs"] = {**run["model_kwargs"], g["key"]: value}
            run["name"] = f"{factor}={value}"
            key = json.dumps({k: v for k, v in run.items() if k != "name"}, sort_keys=True)
            if key not in cache:
                r = train(TrainConfig.from_dict(run), splits)
                cache[key] = {
                    **_round(evaluate_logits(r.test_logits, r.test_labels)),
                    "final_train_loss": round(r.history[-1]["train_loss"], 5),
                    "final_train_error": round(r.history[-1]["train_error"], 3),
                    "n_params": r.n_params,
                    "train_seconds": round(r.train_seconds, 1),
                    "model": run["model"],
                    "model_kwargs": json.dumps(run["model_kwargs"], sort_keys=True),
                    "weight_decay": run.get("weight_decay", 0.0),
                }
            row = {"factor": factor, "value": value, **cache[key]}
            rows.append(row)
            groups[factor].append(row)
    _write_csv(out / "factors.csv", rows)
    (out / "factors.json").write_text(json.dumps({"config": cfg, "rows": rows}, indent=2))
    factors_figure(groups, out / "fig2_factors.png")
    return {"rows": rows}


# --------------------------------------------------------------------------- claim 3
def calibrate_logits(val_logits, val_labels, test_logits, test_labels) -> tuple[list[dict], dict]:
    rows, probs = [], {}
    for cal in all_calibrators():
        cal.fit(val_logits, val_labels)
        p = cal.predict_proba(test_logits)
        m = evaluate_probs_rounded(p, test_labels)
        lo, hi = bootstrap_ece_ci(p, test_labels)
        m["ece_ci95_low"], m["ece_ci95_high"] = round(lo, 4), round(hi, 4)
        if hasattr(cal, "temperature"):
            m["temperature"] = round(cal.temperature, 4)
        rows.append({"method": cal.name, **m})
        probs[cal.name] = p
    return rows, probs


def evaluate_probs_rounded(p, y) -> dict:
    from .metrics import evaluate_probs

    return _round(evaluate_probs(p, y))


def run_claim3(cfg: dict, out: Path) -> dict:
    """Post-hoc calibration on the validation split; evaluate on test (Table 1)."""
    # One thread makes the L-BFGS fits bit-reproducible (multi-threaded float
    # reductions can move matrix scaling's optimum slightly between machines).
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    logits_dir = Path(cfg["logits_dir"])
    all_rows, result = [], {}
    for name in cfg["models"]:
        path = logits_dir / f"logits_{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run the claim1 experiment first.")
        d = np.load(path)
        rows, probs = calibrate_logits(
            d["val_logits"].astype(np.float64),
            d["val_labels"],
            d["test_logits"].astype(np.float64),
            d["test_labels"],
        )
        for r in rows:
            all_rows.append({"model": name, **r})
        result[name] = rows
        panels = [(f"{name}: {k}", v, d["test_labels"]) for k, v in probs.items()]
        reliability_figure(panels, out / f"fig4_calibrated_{name}.png", hist=False)
    _write_csv(out / "table1.csv", all_rows)
    (out / "table1.json").write_text(json.dumps(result, indent=2))
    (out / "table1.md").write_text(_markdown_table1(result))
    return result


def _markdown_table1(result: dict[str, list[dict]]) -> str:
    models = list(result)
    methods = [r["method"] for r in result[models[0]]]
    lines = []
    for metric, title in (("ece", "ECE (%)"), ("error", "Error (%)"), ("nll", "NLL")):
        note = " (95% bootstrap CI over test samples)" if metric == "ece" else ""
        lines.append(f"**Test {title}**{note}\n")
        lines.append("| Model | " + " | ".join(methods) + " |")
        lines.append("|---" * (len(methods) + 1) + "|")
        for mname in models:
            cells = []
            for r in result[mname]:
                if metric == "ece":
                    cells.append(f"{r['ece']:.2f} [{r['ece_ci95_low']:.2f}, {r['ece_ci95_high']:.2f}]")
                else:
                    cells.append(("{:.3f}" if metric == "nll" else "{:.2f}").format(r[metric]))
            lines.append(f"| {mname} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
