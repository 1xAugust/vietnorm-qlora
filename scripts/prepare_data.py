"""Prepare ViLexNorm SFT splits. Requires `pip install -e '.[train]'`."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

INSTRUCTION = (
    "Chuẩn hóa văn bản tiếng Việt trên mạng xã hội: sửa viết tắt, teencode và lỗi chính tả "
    "nhưng giữ nguyên ý nghĩa, tên riêng và phần đã đúng. "
    "Chỉ trả về văn bản đã chuẩn hóa."
)


def format_example(row: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": row["original"]},
        ],
        "completion": [{"role": "assistant", "content": row["normalized"]}],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="visolex/ViLexNorm")
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--max-train", type=int, default=3000, help="0 means all training rows")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    from datasets import load_dataset

    if args.max_train < 0:
        p.error("--max-train must be nonnegative")
    all_rows = load_dataset(args.dataset, split="train")
    if not {"original", "normalized", "type"}.issubset(all_rows.column_names):
        raise ValueError("Expected columns original, normalized, type")
    splits = {}
    for value, filename in [("train", "train"), ("dev", "dev"), ("test", "test")]:
        ds = all_rows.filter(lambda row: row["type"] == value)
        if not len(ds):
            raise ValueError(f"No rows for expected split: {value}")
        rows = [dict(row) for row in ds]
        if value == "train":
            random.Random(args.seed).shuffle(rows)
            if args.max_train:
                rows = rows[:args.max_train]
        splits[filename] = rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(format_example(row), ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} -> {path}")
    print("Reminder: output split membership follows original ViLexNorm `type`. Do not tune on test.")


if __name__ == "__main__":
    main()
