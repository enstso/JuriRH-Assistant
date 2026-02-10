#!/usr/bin/env bash
set -euo pipefail
cp -n config.example.yaml config.yaml || true
python -m src.ingest --input_dir data/samples/corpus --out_dir data/index
