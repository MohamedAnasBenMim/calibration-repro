#!/usr/bin/env bash
# Regenerate every table and figure in results/ from scratch (~35 min on 2 CPU cores).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m calrepro claim1   # Fig. 1 + Fig. 3, trains LeNet-5 and ResNet-8, saves logits
python -m calrepro claim3   # Table 1, uses the logits saved by claim1
python -m calrepro claim2   # Fig. 2 (depth / width / BN / weight decay)
