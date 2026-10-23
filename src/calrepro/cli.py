"""Command-line entry point: ``python -m calrepro {claim1,claim2,claim3,smoke} ...``."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .experiments import plot_claim1, run_claim1, run_claim2, run_claim3


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="calrepro", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, default_cfg, default_out in (
        ("claim1", "configs/claim1.json", "results/claim1"),
        ("claim2", "configs/claim2.json", "results/claim2"),
        ("claim3", "configs/claim3.json", "results/claim3"),
    ):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=default_cfg)
        sp.add_argument("--out", default=default_out)
    sp = sub.add_parser("plot-claim1", help="redraw claim-1 figures from saved logits")
    sp.add_argument("--config", default="configs/claim1.json")
    sp.add_argument("--out", default="results/claim1")
    sp = sub.add_parser("smoke", help="tiny end-to-end run of all three claims")
    sp.add_argument("--config", default="configs/smoke.json")
    sp.add_argument("--out", default=None)
    args = p.parse_args(argv)

    if args.cmd == "claim1":
        run_claim1(_load(args.config), Path(args.out))
    elif args.cmd == "claim2":
        run_claim2(_load(args.config), Path(args.out))
    elif args.cmd == "claim3":
        run_claim3(_load(args.config), Path(args.out))
    elif args.cmd == "plot-claim1":
        plot_claim1(_load(args.config), Path(args.out))
    elif args.cmd == "smoke":
        cfg = _load(args.config)
        out = Path(args.out or tempfile.mkdtemp(prefix="calrepro-smoke-"))
        run_claim1(cfg["claim1"], out / "claim1")
        run_claim2(cfg["claim2"], out / "claim2")
        run_claim3({**cfg["claim3"], "logits_dir": str(out / "claim1")}, out / "claim3")
        print(f"smoke run OK -> {out}")


if __name__ == "__main__":
    main()
