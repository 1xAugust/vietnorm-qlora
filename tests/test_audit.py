"""Unit tests for dataset data leakage and duplicate audit."""
from __future__ import annotations

import json
import random
import unicodedata

import pytest

from scripts import audit_data, prepare_data
from tests.test_prepare_data import FakeDataset, install_fake_datasets
from vietnorm_qlora import cli
from vietnorm_qlora.audit import (
    NearDuplicateConfig,
    _find_near_duplicate_pairs,
    audit_split_internal,
    audit_split_pair,
    audit_splits,
    bounded_edit_distance,
    calculate_similarity,
    canonicalize,
    load_split_rows,
    parse_jsonl_row,
    validate_threshold,
)
from vietnorm_qlora.evaluation import edit_distance


def _brute_force_oracle_internal(strings: list[str], threshold: float) -> int:
    """Exact brute-force ground-truth oracle for internal near-duplicate search."""
    cfg = NearDuplicateConfig(threshold=threshold)
    canonical = [(s, canonicalize(s, cfg)) for s in strings]
    count = 0
    n = len(canonical)
    for i in range(n):
        raw_a, can_a = canonical[i]
        len_a = len(can_a)
        for j in range(i + 1, n):
            raw_b, can_b = canonical[j]
            if raw_a == raw_b:
                continue
            len_b = len(can_b)
            max_len = max(len_a, len_b)
            if max_len == 0:
                count += 1
                continue
            max_dist = int((1.0 - threshold) * max_len)
            dist = edit_distance(can_a, can_b)
            if dist <= max_dist:
                count += 1
    return count


def _brute_force_oracle_cross(
    strings_a: list[str], strings_b: list[str], threshold: float
) -> int:
    """Exact brute-force ground-truth oracle for cross-split near-duplicate search."""
    cfg = NearDuplicateConfig(threshold=threshold)
    canonical_a = [(s, canonicalize(s, cfg)) for s in strings_a]
    canonical_b = [(s, canonicalize(s, cfg)) for s in strings_b]
    count = 0
    for raw_a, can_a in canonical_a:
        len_a = len(can_a)
        for raw_b, can_b in canonical_b:
            if raw_a == raw_b:
                continue
            len_b = len(can_b)
            max_len = max(len_a, len_b)
            if max_len == 0:
                count += 1
                continue
            max_dist = int((1.0 - threshold) * max_len)
            dist = edit_distance(can_a, can_b)
            if dist <= max_dist:
                count += 1
    return count


def test_threshold_validation_api():
    assert validate_threshold(0.85) == 0.85
    assert validate_threshold(1.0) == 1.0
    assert validate_threshold(0.001) == 0.001

    for invalid in [0.0, -0.1, 1.01, 2.0, "0.5", None, True, False]:
        with pytest.raises(ValueError):
            validate_threshold(invalid)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            NearDuplicateConfig(threshold=invalid)  # type: ignore[arg-type]


def test_canonicalize_vietnamese_and_casefold():
    # NFD decomposed 'o' + combining acute accent
    decomposed = "ho" + "\u0301" + "a"
    precomposed = "hóa"
    assert decomposed != precomposed

    config = NearDuplicateConfig(unicode_normalization="NFC", case_fold=True, collapse_whitespace=True)
    c1 = canonicalize(decomposed, config)
    c2 = canonicalize(precomposed, config)
    assert c1 == c2 == "hóa"
    assert unicodedata.is_normalized("NFC", c1)

    # Whitespace and casefold test (casefold handles full Unicode case mapping)
    raw = "  ĐÀ NẴNG   ko   BT  "
    assert canonicalize(raw, config) == "đà nẵng ko bt"


def test_bounded_edit_distance():
    assert bounded_edit_distance("abc", "abc", 2) == 0
    assert bounded_edit_distance("abc", "abd", 2) == 1
    assert bounded_edit_distance("abc", "axyz", 1) is None
    assert bounded_edit_distance("short", "much too long string", 2) is None


def test_calculate_similarity_and_threshold():
    cfg = NearDuplicateConfig(threshold=0.85)
    assert calculate_similarity("mik ko bt lm sao", "mik ko bt lm sao", cfg) == 1.0

    # 1 char difference on length 18 -> sim = 17 / 18 = 0.944444 >= 0.85
    sim_near = calculate_similarity("mik ko bt lm sao", "mik ko bt lam sao", cfg)
    assert sim_near >= 0.85

    # Completely different text -> sim < 0.5
    sim_diff = calculate_similarity("mik ko bt lm sao", "hom nay troi dep lam ban oi", cfg)
    assert sim_diff < 0.5


def test_optimized_candidate_search_equivalence_with_oracle():
    """Verify optimized candidate search matches brute-force oracle with zero discrepancy."""
    random.seed(42)
    base_phrases = [
        "nghiên cứu khoa học",
        "chuẩn hóa tiếng việt",
        "dữ liệu văn bản",
        "mô hình ngôn ngữ",
        "tinh chỉnh mô hình",
        "đăng ký môn học",
        "không biết làm sao",
    ]
    synthetic_set: list[str] = []
    for _ in range(40):
        phrase = " ".join(random.choices(base_phrases, k=2))
        synthetic_set.append(phrase)
        # Create subtle mutation (1 char insert, delete, or replace)
        chars = list(phrase)
        if chars:
            idx = random.randint(0, len(chars) - 1)
            op = random.choice(["replace", "insert", "delete"])
            if op == "replace":
                chars[idx] = "x"
            elif op == "delete":
                chars.pop(idx)
            else:
                chars.insert(idx, "a")
            synthetic_set.append("".join(chars))

    # Add Unicode decomposed NFD vs NFC strings
    decomposed_nfd = unicodedata.normalize("NFD", "chuẩn hóa dữ liệu")
    precomposed_nfc = unicodedata.normalize("NFC", "chuẩn hóa dữ liệu")
    synthetic_set.extend([decomposed_nfd, precomposed_nfc])

    unique_strings = sorted(dict.fromkeys(synthetic_set))

    # Test internal and cross search across boundary thresholds (including 1.0)
    for threshold in [0.70, 0.80, 0.85, 0.90, 0.95, 1.0]:
        cfg = NearDuplicateConfig(threshold=threshold)
        oracle_int = _brute_force_oracle_internal(unique_strings, threshold)
        fast_int = _find_near_duplicate_pairs(unique_strings, None, cfg)
        assert fast_int == oracle_int, f"Internal mismatch at threshold {threshold}"

        # Cross split test
        split_a = unique_strings[: len(unique_strings) // 2]
        split_b = unique_strings[len(unique_strings) // 2 :]
        oracle_cr = _brute_force_oracle_cross(split_a, split_b, threshold)
        fast_cr = _find_near_duplicate_pairs(split_a, split_b, cfg)
        assert fast_cr == oracle_cr, f"Cross-split mismatch at threshold {threshold}"


def test_parse_jsonl_row_formats(tmp_path):
    # Simple schema
    orig, norm = parse_jsonl_row({"original": "ko sao", "normalized": "Không sao."})
    assert orig == "ko sao" and norm == "Không sao."

    # SFT schema
    sft_row = {
        "prompt": [
            {"role": "system", "content": "Instruction"},
            {"role": "user", "content": "mik bt"},
        ],
        "completion": [{"role": "assistant", "content": "Mình biết."}],
    }
    orig_sft, norm_sft = parse_jsonl_row(sft_row)
    assert orig_sft == "mik bt" and norm_sft == "Mình biết."

    # Invalid schemas
    with pytest.raises(ValueError):
        parse_jsonl_row({"bad_key": "val"})
    with pytest.raises(ValueError):
        parse_jsonl_row({"prompt": []})

    # Test load_split_rows
    p = tmp_path / "sample.jsonl"
    p.write_text(json.dumps(sft_row) + "\n\n", encoding="utf-8")
    loaded = load_split_rows(p)
    assert len(loaded) == 1
    assert loaded[0] == {"original": "mik bt", "normalized": "Mình biết."}

    with pytest.raises(FileNotFoundError):
        load_split_rows(tmp_path / "nonexistent.jsonl")


def test_internal_audit_clean_and_colliding():
    clean_rows = [
        {"original": "input một", "normalized": "chuẩn hóa một"},
        {"original": "input hai", "normalized": "chuẩn hóa hai"},
        {"original": "input ba", "normalized": "chuẩn hóa ba"},
    ]
    res_clean = audit_split_internal(clean_rows)
    assert res_clean["exact_pair_duplicates"] == 0
    assert res_clean["inconsistent_target"] == 0
    assert res_clean["same_original_duplicates"] == 0
    assert res_clean["same_normalized_duplicates"] == 0
    assert res_clean["near_duplicate_original"] == 0

    colliding_rows = [
        {"original": "input một", "normalized": "chuẩn hóa một"},
        {"original": "input một", "normalized": "chuẩn hóa một"},
        {"original": "input ba", "normalized": "chuẩn hóa ba"},
        {"original": "input ba", "normalized": "chuẩn hóa khác"},
        {"original": "câu dài số một a", "normalized": "chuẩn hóa dài"},
        {"original": "câu dài số một b", "normalized": "chuẩn hóa dài khác"},
    ]
    res_coll = audit_split_internal(colliding_rows)
    assert res_coll["exact_pair_duplicates"] == 1
    assert res_coll["inconsistent_target"] == 1
    assert res_coll["near_duplicate_original"] >= 1


def test_cross_split_audit_combinations():
    train_rows = [
        {"original": "câu chia sẻ chung", "normalized": "chuẩn hóa chung"},
        {"original": "câu xung đột nhãn", "normalized": "nhãn train"},
        {"original": "câu nguồn một", "normalized": "đích chung duy nhất"},
        {"original": "câu gần giống nhau một", "normalized": "đích một"},
    ]
    dev_rows = [
        {"original": "câu chia sẻ chung", "normalized": "chuẩn hóa chung"},
        {"original": "câu xung đột nhãn", "normalized": "nhãn dev khác"},
        {"original": "câu nguồn hai", "normalized": "đích chung duy nhất"},
        {"original": "câu gần giống nhau hai", "normalized": "đích hai"},
    ]
    cfg = NearDuplicateConfig(threshold=0.80)
    res = audit_split_pair(train_rows, dev_rows, config=cfg)

    assert res["exact_full_pair"] == 1
    assert res["exact_original"] == 2
    assert res["inconsistent_target"] == 1
    assert res["exact_normalized"] == 2
    assert res["near_duplicate_original"] == 1


def test_audit_report_does_not_leak_raw_text():
    secret_canary = "BiMatDocQuyenKhongDuocPhepXuatHienTrongReport12345"
    splits = {
        "train": [{"original": secret_canary, "normalized": "chuẩn hóa một"}],
        "dev": [{"original": secret_canary, "normalized": "chuẩn hóa hai"}],
        "test": [{"original": "an toàn", "normalized": "an toàn"}],
    }
    report = audit_splits(splits)
    serialized = json.dumps(report, ensure_ascii=False)

    assert secret_canary not in serialized
    assert "chuẩn hóa một" not in serialized
    assert report["status"] == "FAIL"
    assert report["has_leakage"] is True
    assert report["cross_split_audit"]["train-dev"]["exact_original"] == 1
    assert report["cross_split_audit"]["train-dev"]["inconsistent_target"] == 1


def test_audit_deterministic_output():
    splits = {
        "train": [
            {"original": "mik ko bt", "normalized": "Mình không biết."},
            {"original": "nay di dau", "normalized": "Hôm nay đi đâu?"},
        ],
        "dev": [
            {"original": "hom nay di dau", "normalized": "Hôm nay đi đâu?"},
        ],
        "test": [
            {"original": "test clean", "normalized": "Test sạch."},
        ],
    }
    r1 = audit_splits(splits)
    r2 = audit_splits(splits)

    b1 = json.dumps(r1, ensure_ascii=False, indent=2, sort_keys=True)
    b2 = json.dumps(r2, ensure_ascii=False, indent=2, sort_keys=True)
    assert b1 == b2


def test_scripts_audit_data_clean_and_leaky_cli(tmp_path):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    for name in ("train", "dev", "test"):
        (clean_dir / f"{name}.jsonl").write_text(
            json.dumps({"original": f"{name} orig", "normalized": f"{name} norm"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (clean_dir / "manifest.json").write_text(
        json.dumps({"dataset_identifier": "test/clean", "fingerprint": "abc-123"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 1. Clean run -> exit 0
    exit_code = audit_data.main([
        "--data-dir", str(clean_dir),
        "--update-manifest",
        "--quiet",
    ])
    assert exit_code == 0

    report_path = clean_dir / "audit_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["has_leakage"] is False
    assert report["collision_totals"]["cross_split_exact_original"] == 0

    # Verify manifest updated with ALL published metrics
    manifest = json.loads((clean_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "data_leakage_audit" in manifest
    audit_meta = manifest["data_leakage_audit"]
    assert audit_meta["status"] == "PASS"
    assert audit_meta["near_duplicate_config"]["threshold"] == 0.85
    assert "exact_normalized" in audit_meta["cross_split_collisions"]["train-dev"]
    assert "same_original_duplicates" in audit_meta["internal_collisions"]["train"]
    assert "collision_totals" in audit_meta

    # 2. Leaky run -> exit 1
    leaky_dir = tmp_path / "leaky"
    leaky_dir.mkdir()
    (leaky_dir / "train.jsonl").write_text(
        json.dumps({"original": "leaked input", "normalized": "norm train"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (leaky_dir / "dev.jsonl").write_text(
        json.dumps({"original": "dev safe", "normalized": "norm dev"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (leaky_dir / "test.jsonl").write_text(
        json.dumps({"original": "leaked input", "normalized": "norm test"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    exit_leaky = audit_data.main([
        "--data-dir", str(leaky_dir),
        "--quiet",
    ])
    assert exit_leaky == 1

    # With --allow-leakage -> exit 0
    exit_allowed = audit_data.main([
        "--data-dir", str(leaky_dir),
        "--allow-leakage",
        "--quiet",
    ])
    assert exit_allowed == 0


def test_update_manifest_missing_or_invalid_fails(tmp_path):
    data_dir = tmp_path / "no_manifest"
    data_dir.mkdir()
    for name in ("train", "dev", "test"):
        (data_dir / f"{name}.jsonl").write_text(
            json.dumps({"original": f"{name} a", "normalized": f"{name} b"}) + "\n",
            encoding="utf-8",
        )

    # Missing manifest with --update-manifest must return nonzero (2)
    exit_code = audit_data.main([
        "--data-dir", str(data_dir),
        "--update-manifest",
        "--quiet",
    ])
    assert exit_code == 2

    # Invalid JSON manifest must return nonzero (2)
    (data_dir / "manifest.json").write_text("invalid json content {{{", encoding="utf-8")
    exit_invalid = audit_data.main([
        "--data-dir", str(data_dir),
        "--update-manifest",
        "--quiet",
    ])
    assert exit_invalid == 2


def test_threshold_validation_in_cli_and_scripts(tmp_path):
    data_dir = tmp_path / "val_dir"
    data_dir.mkdir()
    for name in ("train", "dev", "test"):
        (data_dir / f"{name}.jsonl").write_text(
            json.dumps({"original": f"{name} x", "normalized": f"{name} y"}) + "\n",
            encoding="utf-8",
        )

    # audit_data script rejects invalid thresholds
    with pytest.raises(SystemExit):
        audit_data.main(["--data-dir", str(data_dir), "--near-threshold", "0.0"])

    with pytest.raises(SystemExit):
        audit_data.main(["--data-dir", str(data_dir), "--near-threshold", "1.5"])

    # CLI rejects invalid thresholds
    with pytest.raises(SystemExit):
        cli.main(["audit", "--data-dir", str(data_dir), "--near-threshold", "-0.1"])

    with pytest.raises(SystemExit):
        cli.main(["audit", "--data-dir", str(data_dir), "--near-threshold", "1.01"])

    # prepare_data rejects invalid threshold
    with pytest.raises(SystemExit):
        prepare_data.main(["--near-threshold", "0.0"])


def test_prepare_data_with_audit_flag(tmp_path, monkeypatch):
    rows = [
        {"original": "train one", "normalized": "train norm one", "type": "train"},
        {"original": "train two", "normalized": "train norm two", "type": "train"},
        {"original": "dev input", "normalized": "dev output", "type": "dev"},
        {"original": "test input", "normalized": "test output", "type": "test"},
    ]
    install_fake_datasets(monkeypatch, FakeDataset(rows))

    out_dir = tmp_path / "prep_audit"
    prepare_data.main([
        "--output-dir", str(out_dir),
        "--audit",
        "--near-threshold", "0.85",
    ])

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "data_leakage_audit" in manifest
    assert manifest["data_leakage_audit"]["status"] == "PASS"
    assert manifest["data_leakage_audit"]["near_duplicate_config"]["threshold"] == 0.85
    assert manifest["data_leakage_audit"]["cross_split_collisions"]["train-test"]["exact_original"] == 0


def test_vietnorm_cli_audit_subcommand(tmp_path, capsys):
    data_dir = tmp_path / "cli_data"
    data_dir.mkdir()
    for name in ("train", "dev", "test"):
        (data_dir / f"{name}.jsonl").write_text(
            json.dumps({"original": f"{name} text", "normalized": f"{name} target"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # Should run and print json without exiting error
    cli.main(["audit", "--data-dir", str(data_dir)])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["status"] == "PASS"
    assert report["has_leakage"] is False
