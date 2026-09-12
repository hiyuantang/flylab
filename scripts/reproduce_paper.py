#!/usr/bin/env python3
"""Download pinned paper tables if requested, then run the full-graph protocol."""
import argparse
import json
from pathlib import Path
import urllib.request
from flylab.paper_experiment import COMMIT, SOURCES, feeding_experiment

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=Path('data/paper-reference'))
parser.add_argument('--download', action='store_true')
parser.add_argument('--duration-ms', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
if args.download:
    args.data.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        path = args.data / name
        if not path.exists():
            urllib.request.urlretrieve(f'https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{COMMIT}/{name}', path)
import torch
torch.set_num_threads(1)
result = feeding_experiment(args.data, args.duration_ms, args.seed, progress=lambda p: print(json.dumps(p), flush=True))
destination = args.data / f'feeding-{args.seed}-{args.duration_ms}ms.json'
destination.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
