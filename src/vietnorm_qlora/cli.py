"""CLI that runs on CPU with only the standard library."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .evaluation import identity_baseline, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="VietNorm-QLoRA research starter")
    subs = parser.add_subparsers(dest="command", required=True)
    p_base = subs.add_parser("baseline", help="Identity baseline; does NOT correct text")
    p_base.add_argument("text", help="Text to leave unchanged")
    p_eval = subs.add_parser("evaluate", help="Evaluate original/normalized/prediction JSONL")
    p_eval.add_argument("--file", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "baseline":
        print(identity_baseline(args.text))
    elif args.command == "evaluate":
        with args.file.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
