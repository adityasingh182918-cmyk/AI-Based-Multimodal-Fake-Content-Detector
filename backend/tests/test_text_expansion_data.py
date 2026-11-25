from __future__ import annotations

from scripts.download_text_expansion import sample_split
from scripts.prepare_expanded_text_dataset import normalize, summary


def test_mage_sampler_is_seeded_source_stratified_and_applies_label_mapping():
    rows = [
        {"text": f"A generated sample number {i} with enough words to pass the length checks.", "label": 0, "src": "news_machine_gpt4"}
        for i in range(8)
    ] + [
        {"text": f"A human source sample number {i} with enough words to pass the length checks.", "label": 1, "src": "news_human"}
        for i in range(6)
    ]
    first, first_report = sample_split(rows, "train", seed=42, ai_cap=3, human_cap=2)
    second, second_report = sample_split(rows, "train", seed=42, ai_cap=3, human_cap=2)

    assert [(r["label"], r["text"]) for r in first] == [(r["label"], r["text"]) for r in second]
    assert first_report == second_report
    assert first_report["selected_class_counts"] == {"AI_GENERATED": 3, "HUMAN_WRITTEN": 2}
    assert {r["source_group"] for r in first} == {"news_machine_gpt4", "news_human"}
    assert {r["source_label"] for r in first} == {0, 1}


def test_mage_sampler_rejects_invalid_and_too_short_rows():
    rows = [
        {"text": "tiny", "label": 0, "src": "short"},
        {"text": "A sufficiently long sample has enough separate words for training.", "label": 9, "src": "bad-label"},
        {"text": "A sufficiently long human sample has enough words for training.", "label": 1, "src": "human"},
    ]
    selected, report = sample_split(rows, "train", seed=1, ai_cap=4, human_cap=4)
    assert len(selected) == 1
    assert selected[0]["label"] == "HUMAN_WRITTEN"
    assert report["skipped_short"] == 1
    assert report["skipped_invalid"] == 1


def test_expanded_dataset_summary_and_exact_normalization():
    rows = [
        {"text": "few words", "label": "HUMAN_WRITTEN", "source": "hc3"},
        {"text": "This example has exactly enough words to classify as a short paragraph.", "label": "AI_GENERATED", "source": "mage:gpt"},
    ]
    result = summary(rows)
    assert result["class_counts"] == {"AI_GENERATED": 1, "HUMAN_WRITTEN": 1}
    assert result["word_length_buckets"] == {"1-19": 2}
    assert normalize(" A\tB\nC ") == "a b c"
