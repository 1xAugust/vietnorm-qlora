"""Deterministic data leakage and duplicate audit for Vietnamese lexical normalization."""
from __future__ import annotations

import json
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .evaluation import edit_distance

DEFAULT_NEAR_THRESHOLD = 0.85
SPLIT_PAIRS = (("train", "dev"), ("train", "test"), ("dev", "test"))


def validate_threshold(threshold: float) -> float:
    """Validate that near-duplicate similarity threshold is in (0, 1]."""
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(
            f"Threshold must be a float, got {type(threshold).__name__}: {threshold!r}"
        )
    val = float(threshold)
    if not (0.0 < val <= 1.0):
        raise ValueError(
            f"Near-duplicate threshold must satisfy 0 < threshold <= 1.0, got {threshold}"
        )
    return val


@dataclass(frozen=True)
class NearDuplicateConfig:
    """Configuration for Vietnamese Unicode near-duplicate detection."""

    method: str = "normalized_levenshtein_similarity"
    threshold: float = DEFAULT_NEAR_THRESHOLD
    unicode_normalization: str = "NFC"
    case_fold: bool = True
    collapse_whitespace: bool = True

    def __post_init__(self) -> None:
        validate_threshold(self.threshold)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonicalize(text: str, config: NearDuplicateConfig | None = None) -> str:
    """Canonicalize Vietnamese text with Unicode normalization, casefold and whitespace collapse."""
    if config is None:
        config = NearDuplicateConfig()
    normalized = unicodedata.normalize(config.unicode_normalization, text)
    if config.case_fold:
        normalized = normalized.casefold()
    if config.collapse_whitespace:
        normalized = " ".join(normalized.split())
    return normalized


def bounded_edit_distance(a: str, b: str, max_dist: int) -> int | None:
    """Levenshtein edit distance with early exit when exceeding max_dist."""
    if abs(len(a) - len(b)) > max_dist:
        return None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        min_val = cur[0]
        for j, cb in enumerate(b, 1):
            val = min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb))
            cur.append(val)
            if val < min_val:
                min_val = val
        if min_val > max_dist:
            return None
        prev = cur
    return prev[-1] if prev[-1] <= max_dist else None


def calculate_similarity(
    a: str, b: str, config: NearDuplicateConfig | None = None
) -> float:
    """Compute normalized Levenshtein similarity [0.0, 1.0] on canonicalized text."""
    if config is None:
        config = NearDuplicateConfig()
    ca = canonicalize(a, config)
    cb = canonicalize(b, config)
    if ca == cb:
        return 1.0
    max_len = max(len(ca), len(cb))
    if max_len == 0:
        return 1.0
    max_dist = int((1.0 - config.threshold) * max_len)
    dist = bounded_edit_distance(ca, cb, max_dist)
    if dist is None:
        dist = edit_distance(ca, cb)
    return round(1.0 - (dist / max_len), 6)


def parse_jsonl_row(data: dict[str, Any], row_idx: int = 0) -> tuple[str, str]:
    """Extract (original, normalized) from SFT or raw JSONL schema."""
    if (
        "original" in data
        and "normalized" in data
        and isinstance(data["original"], str)
        and isinstance(data["normalized"], str)
    ):
        return data["original"], data["normalized"]

    if "prompt" in data and "completion" in data:
        try:
            prompt_msgs = data["prompt"]
            completion_msgs = data["completion"]
            user_msg = next(
                m["content"]
                for m in prompt_msgs
                if isinstance(m, dict) and m.get("role") == "user"
            )
            asst_msg = next(
                m["content"]
                for m in completion_msgs
                if isinstance(m, dict) and m.get("role") == "assistant"
            )
            if isinstance(user_msg, str) and isinstance(asst_msg, str):
                return user_msg, asst_msg
        except (StopIteration, TypeError, KeyError):
            pass

    raise ValueError(
        f"Row {row_idx} is missing required string fields: either "
        "('original' and 'normalized') or SFT ('prompt' and 'completion')."
    )


def load_split_rows(path: Path | str) -> list[dict[str, str]]:
    """Load JSONL split into list of {original, normalized} dicts."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {file_path}")
    rows: list[dict[str, str]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            original, normalized = parse_jsonl_row(data, row_idx=idx)
            rows.append({"original": original, "normalized": normalized})
    return rows


def _find_near_duplicate_pairs(
    unique_a: list[str],
    unique_b: list[str] | None,
    config: NearDuplicateConfig,
) -> int:
    """Count distinct pairs with similarity >= threshold and distinct raw strings.

    Uses length partitioning and Pigeonhole substring indexing for deterministic,
    exact candidate filtering with zero false negatives.
    """
    is_internal = unique_b is None
    strings_b = unique_a if is_internal else unique_b
    assert strings_b is not None

    by_len_a: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for raw_a in unique_a:
        can_a = canonicalize(raw_a, config)
        by_len_a[len(can_a)].append((raw_a, can_a))

    by_len_b: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for raw_b in strings_b:
        can_b = canonicalize(raw_b, config)
        by_len_b[len(can_b)].append((raw_b, can_b))

    th = config.threshold
    count = 0
    lengths_a = sorted(by_len_a.keys())

    for L_a in lengths_a:
        items_a = by_len_a[L_a]
        min_L = int(th * L_a)
        max_L = int(L_a / th) if th > 0 else 10**6

        if is_internal:
            target_lengths = [L for L in lengths_a if min_L <= L <= max_L and L >= L_a]
        else:
            target_lengths = [L for L in by_len_b.keys() if min_L <= L <= max_L]

        for L_b in target_lengths:
            items_b = by_len_b[L_b]
            max_len = max(L_a, L_b)
            if max_len == 0:
                continue

            max_dist = int((1.0 - th) * max_len)

            if max_dist == 0:
                # Similarity requires distance 0 (exact canonical match with distinct raw text)
                b_map: dict[str, list[str]] = defaultdict(list)
                for raw_b, can_b in items_b:
                    b_map[can_b].append(raw_b)

                for raw_a, can_a in items_a:
                    for raw_b in b_map.get(can_a, []):
                        if is_internal:
                            if raw_a < raw_b:
                                count += 1
                        else:
                            if raw_a != raw_b:
                                count += 1
                continue

            k_blocks = max_dist + 1
            block_len = L_a // k_blocks

            if block_len < 2:
                # Direct comparison for small blocks
                for pos_a, (raw_a, can_a) in enumerate(items_a):
                    start_b = (pos_a + 1) if (is_internal and L_a == L_b) else 0
                    for raw_b, can_b in items_b[start_b:]:
                        if raw_a == raw_b:
                            continue
                        d = bounded_edit_distance(can_a, can_b, max_dist)
                        if d is not None:
                            count += 1
            else:
                # Pigeonhole Principle: At least one of k_blocks non-overlapping
                # blocks of can_a must occur verbatim as a substring in can_b.
                b_sub_index: dict[str, list[int]] = defaultdict(list)
                for pos_b, (_, can_b) in enumerate(items_b):
                    for st in range(len(can_b) - block_len + 1):
                        b_sub_index[can_b[st : st + block_len]].append(pos_b)

                for pos_a, (raw_a, can_a) in enumerate(items_a):
                    candidate_pos_b: set[int] = set()
                    for b_idx in range(k_blocks):
                        block = can_a[b_idx * block_len : (b_idx + 1) * block_len]
                        for pb in b_sub_index.get(block, []):
                            if is_internal and L_a == L_b and pb <= pos_a:
                                continue
                            candidate_pos_b.add(pb)

                    for pb in candidate_pos_b:
                        raw_b, can_b = items_b[pb]
                        if raw_a == raw_b:
                            continue
                        d = bounded_edit_distance(can_a, can_b, max_dist)
                        if d is not None:
                            count += 1

    return count


def audit_split_internal(
    rows: Sequence[dict[str, str]], config: NearDuplicateConfig | None = None
) -> dict[str, int]:
    """Audit a single split for internal duplicates and inconsistencies."""
    if config is None:
        config = NearDuplicateConfig()

    n = len(rows)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    orig_to_norms: dict[str, set[str]] = defaultdict(set)
    norm_counts: dict[str, int] = defaultdict(int)

    for r in rows:
        pair_counts[(r["original"], r["normalized"])] += 1
        orig_to_norms[r["original"]].add(r["normalized"])
        norm_counts[r["normalized"]] += 1

    unique_pairs = len(pair_counts)
    exact_pair_duplicates = n - unique_pairs

    unique_originals = sorted(orig_to_norms.keys())
    same_original_duplicates = n - len(unique_originals)
    same_normalized_duplicates = n - len(norm_counts)

    inconsistent_target = sum(1 for norms in orig_to_norms.values() if len(norms) > 1)
    near_duplicate_original = _find_near_duplicate_pairs(
        unique_originals, None, config
    )

    return {
        "total_rows": n,
        "exact_pair_duplicates": exact_pair_duplicates,
        "inconsistent_target": inconsistent_target,
        "same_original_duplicates": same_original_duplicates,
        "same_normalized_duplicates": same_normalized_duplicates,
        "near_duplicate_original": near_duplicate_original,
    }


def audit_split_pair(
    rows_a: Sequence[dict[str, str]],
    rows_b: Sequence[dict[str, str]],
    config: NearDuplicateConfig | None = None,
) -> dict[str, int]:
    """Audit cross-split collisions between two splits."""
    if config is None:
        config = NearDuplicateConfig()

    pairs_a = {(r["original"], r["normalized"]) for r in rows_a}
    pairs_b = {(r["original"], r["normalized"]) for r in rows_b}
    exact_full_pair = len(pairs_a.intersection(pairs_b))

    orig_to_norms_a: dict[str, set[str]] = defaultdict(set)
    for r in rows_a:
        orig_to_norms_a[r["original"]].add(r["normalized"])

    orig_to_norms_b: dict[str, set[str]] = defaultdict(set)
    for r in rows_b:
        orig_to_norms_b[r["original"]].add(r["normalized"])

    common_originals = sorted(set(orig_to_norms_a.keys()).intersection(orig_to_norms_b.keys()))
    exact_original = len(common_originals)

    inconsistent_target = 0
    for orig in common_originals:
        if orig_to_norms_a[orig] != orig_to_norms_b[orig]:
            inconsistent_target += 1

    norms_a = {r["normalized"] for r in rows_a}
    norms_b = {r["normalized"] for r in rows_b}
    exact_normalized = len(norms_a.intersection(norms_b))

    unique_orig_a = sorted(orig_to_norms_a.keys())
    unique_orig_b = sorted(orig_to_norms_b.keys())
    near_duplicate_original = _find_near_duplicate_pairs(
        unique_orig_a, unique_orig_b, config
    )

    return {
        "exact_full_pair": exact_full_pair,
        "exact_original": exact_original,
        "inconsistent_target": inconsistent_target,
        "exact_normalized": exact_normalized,
        "near_duplicate_original": near_duplicate_original,
    }


def audit_splits(
    splits: dict[str, Sequence[dict[str, str]]],
    config: NearDuplicateConfig | None = None,
    manifest_data: dict[str, Any] | None = None,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete deterministic audit on train/dev/test splits."""
    if config is None:
        config = NearDuplicateConfig()

    split_sizes = {name: len(rows) for name, rows in sorted(splits.items())}

    internal_audit: dict[str, dict[str, int]] = {}
    for name in sorted(splits.keys()):
        internal_audit[name] = audit_split_internal(splits[name], config)

    cross_split_audit: dict[str, dict[str, int]] = {}
    for a, b in SPLIT_PAIRS:
        pair_key = f"{a}-{b}"
        if a in splits and b in splits:
            cross_split_audit[pair_key] = audit_split_pair(splits[a], splits[b], config)
        else:
            cross_split_audit[pair_key] = {
                "exact_full_pair": 0,
                "exact_original": 0,
                "inconsistent_target": 0,
                "exact_normalized": 0,
                "near_duplicate_original": 0,
            }

    cross_split_collisions = sum(
        res["exact_original"] + res["near_duplicate_original"]
        for res in cross_split_audit.values()
    )
    internal_collisions = sum(
        res["exact_pair_duplicates"] + res["inconsistent_target"]
        for res in internal_audit.values()
    )

    has_cross_split_leakage = cross_split_collisions > 0
    has_internal_collision = internal_collisions > 0
    has_leakage = has_cross_split_leakage or has_internal_collision
    status = "FAIL" if has_leakage else "PASS"

    collision_totals = {
        "cross_split_exact_full_pair": sum(
            c["exact_full_pair"] for c in cross_split_audit.values()
        ),
        "cross_split_exact_original": sum(
            c["exact_original"] for c in cross_split_audit.values()
        ),
        "cross_split_inconsistent_target": sum(
            c["inconsistent_target"] for c in cross_split_audit.values()
        ),
        "cross_split_exact_normalized": sum(
            c["exact_normalized"] for c in cross_split_audit.values()
        ),
        "cross_split_near_duplicate_original": sum(
            c["near_duplicate_original"] for c in cross_split_audit.values()
        ),
        "internal_exact_pair_duplicates": sum(
            i["exact_pair_duplicates"] for i in internal_audit.values()
        ),
        "internal_inconsistent_target": sum(
            i["inconsistent_target"] for i in internal_audit.values()
        ),
        "internal_same_original_duplicates": sum(
            i["same_original_duplicates"] for i in internal_audit.values()
        ),
        "internal_same_normalized_duplicates": sum(
            i["same_normalized_duplicates"] for i in internal_audit.values()
        ),
        "internal_near_duplicate_original": sum(
            i["near_duplicate_original"] for i in internal_audit.values()
        ),
    }

    manifest_ref: dict[str, Any] | None = None
    if manifest_data or manifest_path:
        manifest_ref = {
            "manifest_path": str(manifest_path) if manifest_path else None,
            "dataset_identifier": (
                manifest_data.get("dataset_identifier") if manifest_data else None
            ),
            "requested_revision": (
                manifest_data.get("requested_revision") if manifest_data else None
            ),
            "resolved_revision": (
                manifest_data.get("resolved_revision") if manifest_data else None
            ),
            "fingerprint": (
                manifest_data.get("fingerprint") if manifest_data else None
            ),
        }

    return {
        "status": status,
        "has_leakage": has_leakage,
        "has_cross_split_leakage": has_cross_split_leakage,
        "has_internal_collision": has_internal_collision,
        "near_duplicate_config": config.to_dict(),
        "manifest_reference": manifest_ref,
        "split_sizes": split_sizes,
        "internal_audit": internal_audit,
        "cross_split_audit": cross_split_audit,
        "collision_totals": collision_totals,
    }


def execute_audit(
    *,
    data_dir: Path | None = None,
    train_path: Path | None = None,
    dev_path: Path | None = None,
    test_path: Path | None = None,
    manifest_path: Path | None = None,
    near_threshold: float = DEFAULT_NEAR_THRESHOLD,
    output_report_path: Path | None = None,
    update_manifest: bool = False,
    allow_leakage: bool = False,
    quiet: bool = False,
    format_as_json: bool = False,
) -> int:
    """Unified audit runner used across scripts and CLI."""
    try:
        validated_threshold = validate_threshold(near_threshold)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    resolved_data_dir = data_dir or Path("data/processed")
    t_path = train_path or (resolved_data_dir / "train.jsonl")
    d_path = dev_path or (resolved_data_dir / "dev.jsonl")
    s_path = test_path or (resolved_data_dir / "test.jsonl")

    for name, p in (("train", t_path), ("dev", d_path), ("test", s_path)):
        if not p.is_file():
            print(f"Error: Split file '{name}' not found: {p}", file=sys.stderr)
            return 2

    m_path = manifest_path or (resolved_data_dir / "manifest.json")
    manifest_data: dict[str, Any] | None = None

    if update_manifest:
        if not m_path.is_file():
            print(
                f"Error: --update-manifest was requested but manifest file was not found: {m_path}",
                file=sys.stderr,
            )
            return 2
        try:
            with m_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("Manifest content must be a JSON object")
            manifest_data = loaded
        except Exception as e:
            print(f"Error: Failed to parse manifest at {m_path}: {e}", file=sys.stderr)
            return 2
    elif manifest_path is not None:
        if not m_path.is_file():
            print(f"Error: Explicitly provided manifest file not found: {m_path}", file=sys.stderr)
            return 2
        try:
            with m_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("Manifest content must be a JSON object")
            manifest_data = loaded
        except Exception as e:
            print(f"Error: Failed to parse manifest at {m_path}: {e}", file=sys.stderr)
            return 2
    elif m_path.is_file():
        try:
            with m_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                manifest_data = loaded
        except Exception:
            manifest_data = None

    splits = {
        "train": load_split_rows(t_path),
        "dev": load_split_rows(d_path),
        "test": load_split_rows(s_path),
    }

    config = NearDuplicateConfig(threshold=validated_threshold)
    report = audit_splits(
        splits,
        config=config,
        manifest_data=manifest_data,
        manifest_path=m_path if manifest_data is not None else None,
    )

    if not quiet:
        if format_as_json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print("=" * 60)
            print(f"VietNorm-QLoRA Data Leakage Audit - Status: {report['status']}")
            print("=" * 60)
            print(
                f"Near-duplicate method: {config.method} (threshold={config.threshold}, "
                f"normalization={config.unicode_normalization})"
            )
            print(
                "Split sizes: "
                + ", ".join(f"{name}={count}" for name, count in report["split_sizes"].items())
            )
            print("\nCross-split collisions:")
            for pair, counts in report["cross_split_audit"].items():
                print(
                    f"  {pair:10s} -> exact_original={counts['exact_original']}, "
                    f"exact_full_pair={counts['exact_full_pair']}, "
                    f"inconsistent_target={counts['inconsistent_target']}, "
                    f"exact_normalized={counts['exact_normalized']}, "
                    f"near_dup_original={counts['near_duplicate_original']}"
                )
            print("\nInternal collisions per split:")
            for split_name, counts in report["internal_audit"].items():
                print(
                    f"  {split_name:10s} -> exact_pair_duplicates={counts['exact_pair_duplicates']}, "
                    f"inconsistent_target={counts['inconsistent_target']}, "
                    f"same_orig={counts['same_original_duplicates']}, "
                    f"same_norm={counts['same_normalized_duplicates']}, "
                    f"near_dup_original={counts['near_duplicate_original']}"
                )
            print("=" * 60)

    out_report = output_report_path
    if out_report is None and resolved_data_dir.is_dir():
        out_report = resolved_data_dir / "audit_report.json"

    if out_report is not None:
        out_report.parent.mkdir(parents=True, exist_ok=True)
        with out_report.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        if not quiet and not format_as_json:
            print(f"Audit report written to: {out_report}")

    if update_manifest and manifest_data is not None:
        manifest_data["data_leakage_audit"] = {
            "status": report["status"],
            "has_leakage": report["has_leakage"],
            "near_duplicate_config": report["near_duplicate_config"],
            "cross_split_collisions": report["cross_split_audit"],
            "internal_collisions": report["internal_audit"],
            "collision_totals": report["collision_totals"],
        }
        with m_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        if not quiet and not format_as_json:
            print(f"Updated manifest at: {m_path}")

    if report["has_leakage"] and not allow_leakage:
        if not quiet and not format_as_json:
            print("Audit failed: data leakage or collision detected!", file=sys.stderr)
        return 1

    return 0
