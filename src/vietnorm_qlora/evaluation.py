"""Minimal evaluation utilities. Not a reproduction of ViLexNorm ERR."""
from __future__ import annotations


def identity_baseline(text: str) -> str:
    """Leave-As-Is (LAI): returns exactly the original text."""
    return text


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance on Unicode codepoints."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def summarize(records: list[dict[str, str]]) -> dict[str, float | int]:
    """Compute exact-match, pooled CER and input-change fraction."""
    if not records:
        raise ValueError("Evaluation dataset must not be empty")
    for idx, row in enumerate(records):
        for field in ("original", "normalized", "prediction"):
            if field not in row or not isinstance(row[field], str):
                raise ValueError(f"Row {idx} is missing string field {field!r}")
    n = len(records)
    exact = sum(row["prediction"] == row["normalized"] for row in records)
    changed = sum(row["prediction"] != row["original"] for row in records)
    edits = sum(edit_distance(row["prediction"], row["normalized"]) for row in records)
    gold_chars = sum(len(row["normalized"]) for row in records)
    return {
        "count": n,
        "exact_match": round(exact / n, 6),
        "cer": round(edits / max(1, gold_chars), 6),
        "change_rate": round(changed / n, 6),
    }
