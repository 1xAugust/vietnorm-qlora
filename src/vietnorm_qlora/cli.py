"""CLI that runs on CPU with only the standard library."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import (
    DEFAULT_NEAR_THRESHOLD,
    execute_audit,
    validate_threshold,
)
from .evaluation import identity_baseline, summarize


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VietNorm-QLoRA research starter")
    subs = parser.add_subparsers(dest="command", required=True)

    p_base = subs.add_parser("baseline", help="Identity baseline; does NOT correct text")
    p_base.add_argument("text", help="Text to leave unchanged")

    p_eval = subs.add_parser("evaluate", help="Evaluate original/normalized/prediction JSONL")
    p_eval.add_argument("--file", required=True, type=Path)

    p_audit = subs.add_parser("audit", help="Audit local dataset splits for leakage and duplicates")
    p_audit.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p_audit.add_argument("--train", type=Path, help="Override path to train.jsonl")
    p_audit.add_argument("--dev", type=Path, help="Override path to dev.jsonl")
    p_audit.add_argument("--test", type=Path, help="Override path to test.jsonl")
    p_audit.add_argument("--manifest", type=Path, help="Path to manifest.json")
    p_audit.add_argument(
        "--near-threshold",
        type=float,
        default=DEFAULT_NEAR_THRESHOLD,
        help=f"Similarity threshold for near-duplicates (default: {DEFAULT_NEAR_THRESHOLD})",
    )
    p_audit.add_argument("--output-report", type=Path, help="Path to save JSON audit report")
    p_audit.add_argument(
        "--update-manifest",
        action="store_true",
        help="Update manifest.json with data leakage audit findings",
    )
    p_audit.add_argument(
        "--allow-leakage",
        action="store_true",
        help="Do not exit with error even if leakage is detected",
    )
    p_audit.add_argument("--quiet", action="store_true", help="Suppress console summary")

    args = parser.parse_args(argv)

    if args.command == "baseline":
        print(identity_baseline(args.text))
    elif args.command == "evaluate":
        with args.file.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
    elif args.command == "audit":
        try:
            validate_threshold(args.near_threshold)
        except ValueError as e:
            p_audit.error(str(e))

        exit_code = execute_audit(
            data_dir=args.data_dir,
            train_path=args.train,
            dev_path=args.dev,
            test_path=args.test,
            manifest_path=args.manifest,
            near_threshold=args.near_threshold,
            output_report_path=args.output_report,
            update_manifest=args.update_manifest,
            allow_leakage=args.allow_leakage,
            quiet=args.quiet,
            format_as_json=True,
        )
        if exit_code != 0:
            sys.exit(exit_code)
