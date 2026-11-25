import json

import pytest

from app.services.text_detector import (InvalidTextError, TextDetector, TextModelNotReadyError,
                                        split_text)
from scripts.validate_text_dataset import validate


def test_missing_text_checkpoint_returns_readable_error(tmp_path):
    detector = TextDetector(tmp_path / "absent.joblib")
    with pytest.raises(TextModelNotReadyError, match="missing"):
        detector.predict("This is long enough text that is valid but has no trained checkpoint.")


def test_empty_short_and_too_long_text_are_rejected(tmp_path):
    detector = TextDetector(tmp_path / "absent.joblib")
    for value in ("", "tiny text", "a" * 50_001):
        with pytest.raises(InvalidTextError):
            detector.predict(value)


def test_unicode_text_and_long_text_are_split_without_content_loss():
    content = ("Café résumé naïve, 東京。\n" * 300)
    chunks = split_text(content, 200)
    assert len(chunks) > 1
    assert "Café" in "".join(chunks)
    assert "東京" in "".join(chunks)
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_text_prediction_loads_once_and_uses_validation_thresholds(tmp_path):
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(CountVectorizer(), LogisticRegression())
    model.fit(["human words plain prose", "authored writing by person", "generated synthetic text", "machine generated answer"],
              ["HUMAN_WRITTEN", "HUMAN_WRITTEN", "AI_GENERATED", "AI_GENERATED"])
    checkpoint = tmp_path / "fixture.joblib"
    joblib.dump(model, checkpoint)
    checkpoint.with_suffix(".json").write_text(json.dumps({"model_name": "Expanded TF-IDF SVM",
        "model_version": "svm_expanded_test", "decision_thresholds": {
        "real_max_ai_score": .4, "ai_min_ai_score": .6}}), encoding="utf-8")
    detector = TextDetector(checkpoint)
    result = detector.predict("This is a sufficiently long human authored piece with several ordinary words.")
    assert result["label"] in {"HUMAN_WRITTEN", "AI_GENERATED", "UNCERTAIN"}
    assert result["input_word_count"] > 5
    assert result["chunk_count"] == 1
    assert result["model_name"] == "Expanded TF-IDF SVM"
    assert result["model_version"] == "svm_expanded_test"
    assert result["confidence_type"] == "calibrated_tfidf_probability"
    assert result["decision_status"] in {"thresholded", "uncertain"}
    assert detector.is_loaded


def test_prepared_dataset_validation_detects_group_leakage(tmp_path):
    for split in ("train", "validation", "test"):
        row = {"text": f"This is a valid example for the class in the {split} split.",
               "label": "HUMAN_WRITTEN" if split != "validation" else "AI_GENERATED",
               "group_id": "same-question" if split != "test" else "different-question"}
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = validate(tmp_path)
    assert any("group leakage" in error for error in report["errors"])
