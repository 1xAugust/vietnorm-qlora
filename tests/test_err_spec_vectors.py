"""Tests verifying the integrity, arithmetic correctness, and dynamic event extraction of ERR test vectors."""
from __future__ import annotations

import json
from pathlib import Path
import unicodedata

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "err_test_vectors.json"


def tokenize_canonical(text: str) -> list[str]:
    """Proposed canonical tokenization: NFKC Unicode normalization, whitespace strip and split."""
    return unicodedata.normalize("NFKC", text).strip().split()


def token_levenshtein_distance(a: list[str], b: list[str]) -> int:
    """Standard dynamic programming Levenshtein distance on token lists with unit costs."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,  # match / substitution
                dp[i - 1][j] + 1,         # deletion from a
                dp[i][j - 1] + 1,         # insertion into a
            )
    return dp[len(a)][len(b)]


def token_accuracy(seq: list[str], gold: list[str]) -> float:
    """Proposed deterministic token accuracy formula with insertion penalty."""
    if not gold:
        return 1.0 if not seq else 0.0
    dist = token_levenshtein_distance(seq, gold)
    return max(0.0, 1.0 - dist / len(gold))


def align_wagner_fischer(a: list[str], b: list[str]) -> list[tuple[str, int, int, int, int]]:
    """Wagner-Fischer alignment with fixed tie-break priority: match, substitution, deletion, insertion."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    i, j = m, n
    ops: list[tuple[str, int, int, int, int]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", i - 1, i, j - 1, j))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, i, j - 1, j))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, i, j, j))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", i, i, j - 1, j))
            j -= 1
        else:
            raise RuntimeError("Traceback error in Wagner-Fischer")
    ops.reverse()
    return ops


def extract_edit_events(a: list[str], b: list[str]) -> list[dict[str, list[int] | list[str]]]:
    """Group maximal contiguous non-match operations into canonical edit events."""
    ops = align_wagner_fischer(a, b)
    events: list[dict[str, list[int] | list[str]]] = []
    curr_a_start = None
    curr_a_end = None
    curr_b_start = None
    curr_b_end = None

    for op, a_s, a_e, b_s, b_e in ops:
        if op == "match":
            if curr_a_start is not None and curr_a_end is not None and curr_b_start is not None and curr_b_end is not None:
                events.append({
                    "source_span": [curr_a_start, curr_a_end],
                    "source_tokens": a[curr_a_start:curr_a_end],
                    "replacement_tokens": b[curr_b_start:curr_b_end],
                })
                curr_a_start = None
        else:
            if curr_a_start is None:
                curr_a_start = a_s
                curr_b_start = b_s
            curr_a_end = a_e
            curr_b_end = b_e

    if curr_a_start is not None and curr_a_end is not None and curr_b_start is not None and curr_b_end is not None:
        events.append({
            "source_span": [curr_a_start, curr_a_end],
            "source_tokens": a[curr_a_start:curr_a_end],
            "replacement_tokens": b[curr_b_start:curr_b_end],
        })
    return events


def test_fixtures_json_structure() -> None:
    assert FIXTURES_PATH.is_file(), f"Missing fixture file: {FIXTURES_PATH}"
    with FIXTURES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    assert "vectors" in data
    vectors = data["vectors"]
    assert len(vectors) == 9

    vector_ids = [v["vector_id"] for v in vectors]
    assert len(vector_ids) == len(set(vector_ids)), "Vector IDs must be unique"


def test_canonical_tokenization() -> None:
    """Verify that canonical tokenization generates exactly the declared tokens."""
    with FIXTURES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    for vec in data["vectors"]:
        vid = vec["vector_id"]
        assert tokenize_canonical(vec["source_text"]) == vec["source_tokens"], f"Source mismatch in {vid}"
        assert tokenize_canonical(vec["gold_text"]) == vec["gold_tokens"], f"Gold mismatch in {vid}"
        assert tokenize_canonical(vec["prediction_text"]) == vec["prediction_tokens"], f"Pred mismatch in {vid}"


def test_dynamic_event_extraction_and_matching() -> None:
    """Verify that Wagner-Fischer event extraction matches declared target_events, predicted_events, and TP/FP/FN."""
    with FIXTURES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    for vec in data["vectors"]:
        vid = vec["vector_id"]
        s = vec["source_tokens"]
        g = vec["gold_tokens"]
        p = vec["prediction_tokens"]

        calc_t_events = extract_edit_events(s, g)
        calc_p_events = extract_edit_events(s, p)

        assert calc_t_events == vec["target_events"], f"Target events mismatch in {vid}"
        assert calc_p_events == vec["predicted_events"], f"Predicted events mismatch in {vid}"

        # Dynamic TP, FP, FN calculation
        matched_target_indices: set[int] = set()
        tp = 0
        fp = 0
        for pe in calc_p_events:
            matched = False
            for idx, te in enumerate(calc_t_events):
                if idx not in matched_target_indices:
                    if pe["source_span"] == te["source_span"] and pe["replacement_tokens"] == te["replacement_tokens"]:
                        matched = True
                        matched_target_indices.add(idx)
                        break
            if matched:
                tp += 1
            else:
                fp += 1
        fn = len(calc_t_events) - tp

        events = vec["correction_events"]
        assert tp == events["tp"], f"TP mismatch in {vid}: {tp} != {events['tp']}"
        assert fp == events["fp"], f"FP mismatch in {vid}: {fp} != {events['fp']}"
        assert fn == events["fn"], f"FN mismatch in {vid}: {fn} != {events['fn']}"


def test_dynamic_token_levenshtein_and_accuracy() -> None:
    """Directly execute token Levenshtein and token accuracy on token lists, verifying JSON declarations."""
    with FIXTURES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    for vec in data["vectors"]:
        v_id = vec["vector_id"]
        src = vec["source_tokens"]
        gold = vec["gold_tokens"]
        pred = vec["prediction_tokens"]

        calc_base_dist = token_levenshtein_distance(src, gold)
        calc_sys_dist = token_levenshtein_distance(pred, gold)
        calc_base_acc = token_accuracy(src, gold)
        calc_sys_acc = token_accuracy(pred, gold)

        assert calc_base_dist == vec["token_distances"]["baseline_distance"], f"Base dist mismatch in {v_id}"
        assert calc_sys_dist == vec["token_distances"]["system_distance"], f"Sys dist mismatch in {v_id}"
        assert len(gold) == vec["token_distances"]["gold_length"], f"Gold length mismatch in {v_id}"
        assert round(calc_base_acc, 6) == round(vec["baseline_accuracy"], 6), f"Base acc mismatch in {v_id}"
        assert round(calc_sys_acc, 6) == round(vec["system_accuracy"], 6), f"Sys acc mismatch in {v_id}"

        # ERR calculation
        if calc_base_acc < 1.0:
            assert vec["err_status"] == "exact"
            denom = 1.0 - calc_base_acc
            calc_err = round((calc_sys_acc - calc_base_acc) / denom, 6)
            assert calc_err == vec["expected_err"], f"ERR mismatch in {v_id}: {calc_err} != {vec['expected_err']}"
        else:
            assert vec["err_status"] in ("zero_denominator_clean", "zero_denominator_degraded")
            if vec["err_status"] == "zero_denominator_clean":
                assert vec["expected_err"] == 0.0
            else:
                assert vec["expected_err"] is None


def test_precision_recall_consistency() -> None:
    """Verify consistency of TP/FP/FN and undefined handling for P/R."""
    with FIXTURES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    for vec in data["vectors"]:
        v_id = vec["vector_id"]
        events = vec["correction_events"]
        tp = events["tp"]
        fp = events["fp"]
        fn = events["fn"]

        # Precision verification
        pred_denominator = tp + fp
        if pred_denominator > 0:
            calc_prec = round(tp / pred_denominator, 6)
            assert vec["precision"] is not None, f"Precision should not be None in {v_id}"
            assert round(vec["precision"], 6) == calc_prec
            assert vec["precision_status"] == "exact"
        else:
            assert vec["precision"] is None, f"Precision must be None when TP+FP=0 in {v_id}"
            assert vec["precision_status"] == "undefined_no_predictions"

        # Recall verification
        target_denominator = tp + fn
        if target_denominator > 0:
            calc_rec = round(tp / target_denominator, 6)
            assert vec["recall"] is not None, f"Recall should not be None in {v_id}"
            assert round(vec["recall"], 6) == calc_rec
            assert vec["recall_status"] == "exact"
        else:
            assert vec["recall"] is None, f"Recall must be None when TP+FN=0 in {v_id}"
            assert vec["recall_status"] == "undefined_no_targets"
