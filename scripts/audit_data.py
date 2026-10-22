"""CLI script to audit dataset splits for duplicate and cross-split data leakage."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from vietnorm_qlora.audit import (
    DEFAULT_NEAR_THRESHOLD,
    execute_audit,
    validate_threshold,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit local dataset splits for duplicate and cross-split data leakage."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing train.jsonl, dev.jsonl, test.jsonl (default: data/processed)",
    )
    parser.add_argument("--train", type=Path, help="Override path to train.jsonl")
    parser.add_argument("--dev", type=Path, help="Override path to dev.jsonl")
    parser.add_argument("--test", type=Path, help="Override path to test.jsonl")
    parser.add_argument("--manifest", type=Path, help="Path to manifest.json")
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=DEFAULT_NEAR_THRESHOLD,
        help=f"Similarity threshold for near-duplicates (default: {DEFAULT_NEAR_THRESHOLD})",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        help="Path to write JSON audit report (default: <data-dir>/audit_report.json)",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Update manifest.json with data leakage audit metadata and config",
    )
    parser.add_argument(
        "--allow-leakage",
        action="store_true",
        help="Exit with 0 even if leakage is detected (default exits with 1 on leakage)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary output",
    )
    args = parser.parse_args(argv)

    try:
        validate_threshold(args.near_threshold)
    except ValueError as e:
        parser.error(str(e))

    return execute_audit(
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
        format_as_json=False,
    )


if __name__ == "__main__":
    sys.exit(main())
