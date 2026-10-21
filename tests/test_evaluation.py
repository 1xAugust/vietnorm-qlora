import pytest
from vietnorm_qlora.evaluation import identity_baseline, edit_distance, summarize


def test_identity_keeps_vietnamese_untouched():
    assert identity_baseline("mik ko bt") == "mik ko bt"


def test_unicode_distance():
    assert edit_distance("mình", "mình") == 0
    assert edit_distance("a", "á") == 1
    assert edit_distance("", "abc") == 3


def test_summary_metrics():
    rows = [
        {"original": "sai", "normalized": "đúng", "prediction": "đúng"},
        {"original": "tốt", "normalized": "tốt", "prediction": "tốt"},
    ]
    result = summarize(rows)
    assert result == {"count": 2, "exact_match": 1.0, "cer": 0.0, "change_rate": 0.5}


def test_summary_rejects_bad_rows():
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        summarize([{"original": "a", "normalized": "b"}])
