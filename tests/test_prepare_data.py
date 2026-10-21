import json
import types

import pytest

from scripts import prepare_data


class FakeDataset:
    def __init__(
        self,
        rows,
        *,
        column_names=("original", "normalized", "type"),
        fingerprint="fake-fingerprint",
        resolved_revision="resolved-commit",
    ):
        self._rows = list(rows)
        self.column_names = list(column_names)
        self._fingerprint = fingerprint
        self.info = types.SimpleNamespace(dataset_revision=resolved_revision)

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def filter(self, predicate):
        return FakeDataset(
            [row for row in self._rows if predicate(row)],
            column_names=self.column_names,
            fingerprint=self._fingerprint,
            resolved_revision=self.info.dataset_revision,
        )

    def unique(self, column):
        return list(dict.fromkeys(row[column] for row in self._rows))


def install_fake_datasets(monkeypatch, dataset):
    calls = []
    module = types.ModuleType("datasets")

    def load_dataset(identifier, **kwargs):
        calls.append((identifier, kwargs))
        return dataset

    module.load_dataset = load_dataset
    monkeypatch.setitem(__import__("sys").modules, "datasets", module)
    return calls


def test_manifest_is_stable_and_records_provenance_without_row_text(tmp_path, monkeypatch):
    rows = [
        {"original": "train input one", "normalized": "train output one", "type": "train"},
        {"original": "train input two", "normalized": "train output two", "type": "train"},
        {
            "original": "train input three",
            "normalized": "train output three",
            "type": "train",
        },
        {"original": "dev input", "normalized": "dev output", "type": "dev"},
        {"original": "test input", "normalized": "test output", "type": "test"},
    ]
    dataset = FakeDataset(rows)
    calls = install_fake_datasets(monkeypatch, dataset)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    common_args = [
        "--dataset",
        "visolex/ViLexNorm",
        "--revision",
        "requested-commit",
        "--max-train",
        "2",
        "--seed",
        "7",
    ]
    prepare_data.main([*common_args, "--output-dir", str(first_output)])
    prepare_data.main([*common_args, "--output-dir", str(second_output)])

    assert calls == [
        ("visolex/ViLexNorm", {"split": "train", "revision": "requested-commit"}),
        ("visolex/ViLexNorm", {"split": "train", "revision": "requested-commit"}),
    ]
    first_manifest_bytes = (first_output / "manifest.json").read_bytes()
    assert first_manifest_bytes == (second_output / "manifest.json").read_bytes()
    manifest = json.loads(first_manifest_bytes)
    assert manifest == {
        "dataset_identifier": "visolex/ViLexNorm",
        "fingerprint": "fake-fingerprint",
        "input_schema": ["original", "normalized", "type"],
        "max_train": 2,
        "requested_revision": "requested-commit",
        "resolved_revision": "resolved-commit",
        "seed": 7,
        "split_counts": {
            "dev": {"after_subset": 1, "before_subset": 1},
            "test": {"after_subset": 1, "before_subset": 1},
            "train": {"after_subset": 2, "before_subset": 3},
        },
        "unique_type_values": ["dev", "test", "train"],
    }
    manifest_text = first_manifest_bytes.decode("utf-8")
    assert all(row["original"] not in manifest_text for row in rows)
    assert all(row["normalized"] not in manifest_text for row in rows)

    train_rows = [
        json.loads(line)
        for line in (first_output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(train_rows) == 2
    assert all(set(row) == {"prompt", "completion"} for row in train_rows)
    assert all(row["prompt"][0]["content"] == prepare_data.INSTRUCTION for row in train_rows)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {"original": "train input", "normalized": "train output", "type": "train"},
                {"original": "dev input", "normalized": "dev output", "type": "dev"},
            ],
            "missing expected values: test",
        ),
        (
            [
                {"original": "train input", "normalized": "train output", "type": "train"},
                {"original": "dev input", "normalized": "dev output", "type": "dev"},
                {"original": "test input", "normalized": "test output", "type": "test"},
                {
                    "original": "unexpected input",
                    "normalized": "unexpected output",
                    "type": "validation",
                },
            ],
            "unexpected values: 'validation'",
        ),
    ],
)
def test_rejects_type_values_that_are_not_exactly_the_permitted_splits(
    tmp_path, monkeypatch, rows, message
):
    install_fake_datasets(monkeypatch, FakeDataset(rows))

    with pytest.raises(ValueError, match=message):
        prepare_data.main(["--output-dir", str(tmp_path)])


def test_rejects_missing_required_columns(tmp_path, monkeypatch):
    dataset = FakeDataset([], column_names=("original", "type"))
    install_fake_datasets(monkeypatch, dataset)

    with pytest.raises(ValueError, match="missing required columns: normalized"):
        prepare_data.main(["--output-dir", str(tmp_path)])
