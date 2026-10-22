"""Prepare ViLexNorm SFT splits. Requires `pip install -e '.[train]'`."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Sequence
from pathlib import Path

INSTRUCTION = (
    "Chuẩn hóa văn bản tiếng Việt trên mạng xã hội: sửa viết tắt, teencode và lỗi chính tả "
    "nhưng giữ nguyên ý nghĩa, tên riêng và phần đã đúng. "
    "Chỉ trả về văn bản đã chuẩn hóa."
)
REQUIRED_COLUMNS = frozenset({"original", "normalized", "type"})
SPLIT_NAMES = ("train", "dev", "test")


def format_example(row: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": row["original"]},
        ],
        "completion": [{"role": "assistant", "content": row["normalized"]}],
    }


def _resolved_revision(dataset: object, requested_revision: str | None) -> str | None:
    for source in (dataset, getattr(dataset, "info", None)):
        if source is None:
            continue
        for attribute in ("resolved_revision", "dataset_revision", "revision", "_revision"):
            value = getattr(source, attribute, None)
            if isinstance(value, str) and value:
                return value
    if requested_revision and re.fullmatch(r"[0-9a-fA-F]{40}", requested_revision):
        return requested_revision
    return None


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="visolex/ViLexNorm")
    p.add_argument("--revision", help="Dataset revision or commit to load")
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--max-train", type=int, default=3000, help="0 means all training rows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--audit",
        action="store_true",
        help="Run data leakage and collision audit after preparing splits",
    )
    p.add_argument(
        "--near-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for Vietnamese near-duplicates (default: 0.85)",
    )
    args = p.parse_args(argv)

    if args.max_train < 0:
        p.error("--max-train must be nonnegative")

    from vietnorm_qlora.audit import NearDuplicateConfig, audit_splits, validate_threshold

    try:
        validate_threshold(args.near_threshold)
    except ValueError as e:
        p.error(str(e))

    from datasets import load_dataset

    all_rows = load_dataset(args.dataset, split="train", revision=args.revision)
    missing_columns = REQUIRED_COLUMNS.difference(all_rows.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    unique_type_values = set(all_rows.unique("type"))
    permitted_type_values = set(SPLIT_NAMES)
    missing_types = permitted_type_values - unique_type_values
    unexpected_types = unique_type_values - permitted_type_values
    if missing_types or unexpected_types:
        details = []
        if missing_types:
            details.append(f"missing expected values: {', '.join(sorted(missing_types))}")
        if unexpected_types:
            values = ", ".join(sorted(repr(value) for value in unexpected_types))
            details.append(f"unexpected values: {values}")
        raise ValueError(
            "Dataset 'type' values must be exactly train, dev, test; " + "; ".join(details)
        )

    splits = {}
    split_counts = {}
    for value in SPLIT_NAMES:
        ds = all_rows.filter(lambda row: row["type"] == value)
        before_subset = len(ds)
        rows = [dict(row) for row in ds]
        if value == "train":
            random.Random(args.seed).shuffle(rows)
            if args.max_train:
                rows = rows[:args.max_train]
        splits[value] = rows
        split_counts[value] = {
            "before_subset": before_subset,
            "after_subset": len(rows),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(format_example(row), ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} -> {path}")

    manifest = {
        "dataset_identifier": args.dataset,
        "requested_revision": args.revision,
        "resolved_revision": _resolved_revision(all_rows, args.revision),
        "fingerprint": getattr(all_rows, "_fingerprint", None),
        "seed": args.seed,
        "max_train": args.max_train,
        "input_schema": list(all_rows.column_names),
        "unique_type_values": sorted(unique_type_values),
        "split_counts": split_counts,
    }
    if args.audit:
        split_records = {
            name: [{"original": r["original"], "normalized": r["normalized"]} for r in rows]
            for name, rows in splits.items()
        }
        config = NearDuplicateConfig(threshold=args.near_threshold)
        audit_res = audit_splits(
            split_records,
            config=config,
            manifest_data=manifest,
            manifest_path=args.output_dir / "manifest.json",
        )
        manifest["data_leakage_audit"] = {
            "status": audit_res["status"],
            "has_leakage": audit_res["has_leakage"],
            "near_duplicate_config": audit_res["near_duplicate_config"],
            "cross_split_collisions": audit_res["cross_split_audit"],
            "internal_collisions": audit_res["internal_audit"],
            "collision_totals": audit_res["collision_totals"],
        }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"manifest: {manifest_path}")
    print("Reminder: output split membership follows original ViLexNorm `type`. Do not tune on test.")


if __name__ == "__main__":
    main()
