"""Shared data/model helpers for the CPU-friendly HC3 TF-IDF baseline."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

LABELS = ("HUMAN_WRITTEN", "AI_GENERATED")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_model(kind: str):
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=150_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000, sublinear_tf=True)),
    ], transformer_weights={"word": 1.0, "char": 0.8})
    if kind == "logistic":
        classifier = LogisticRegression(max_iter=1200, class_weight="balanced", C=2.0, random_state=42)
    elif kind == "linear_svm":
        classifier = CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=1.0, random_state=42), cv=3)
    else:
        raise ValueError(kind)
    return Pipeline([("features", features), ("classifier", classifier)])


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_model(path: Path):
    return joblib.load(path)
