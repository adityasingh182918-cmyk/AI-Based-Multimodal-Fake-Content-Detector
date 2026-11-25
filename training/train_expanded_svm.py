"""Select and train a separate TF-IDF model on expanded train/validation data."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

try:
    from .text_baseline import read_jsonl
except ImportError:
    from text_baseline import read_jsonl


def thresholds_for(scores: np.ndarray, labels: list[str], max_class_error_rate: float = .05) -> dict:
    human = sorted(float(s) for s, label in zip(scores, labels) if label == "HUMAN_WRITTEN")
    generated = sorted(float(s) for s, label in zip(scores, labels) if label == "AI_GENERATED")
    low = human[max(0, int(len(human) * (1 - max_class_error_rate)) - 1)]
    high = generated[min(len(generated) - 1, int(len(generated) * max_class_error_rate))]
    if low >= high:
        low, high = 0.0, 1.0
    return {"method": "validation_quantile_abstention", "max_class_error_rate": max_class_error_rate,
            "real_max_ai_score": low, "ai_min_ai_score": high, "validation_rows": len(labels)}


def model_candidates(seed: int):
    return [
        ("word_logistic", "word", "logistic", 2.0),
        ("char_logistic", "char", "logistic", 2.0),
        ("word_char_logistic", "word_char", "logistic", 2.0),
        ("word_char_svm_c05", "word_char", "linear_svm", .5),
        ("word_char_svm_c1", "word_char", "linear_svm", 1.0),
        ("word_char_svm_c2", "word_char", "linear_svm", 2.0),
    ]


def vectorizer(kind: str, seed: int):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                           max_features=150_000, sublinear_tf=True, strip_accents=None)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                           max_features=200_000, sublinear_tf=True)
    if kind == "word":
        return word
    if kind == "char":
        return char
    return FeatureUnion([("word", word), ("char", char)], transformer_weights={"word": 1.0, "char": .8})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/text/processed/expanded_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/text_detector/svm_expanded_v1.joblib"))
    parser.add_argument("--report", type=Path, default=Path("training/reports/text_svm_expanded_validation.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    sidecar = args.checkpoint.with_suffix(".json")
    if (args.checkpoint.exists() or sidecar.exists()) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite expanded SVM output: {args.checkpoint}; choose a new version or pass --overwrite")
    train = read_jsonl(args.dataset_dir / "train.jsonl")
    validation = read_jsonl(args.dataset_dir / "validation.jsonl")
    if len({row["label"] for row in train}) != 2 or len({row["label"] for row in validation}) != 2:
        raise SystemExit("Both classes are required in train and validation")
    X_train, y_train = [row["text"] for row in train], [row["label"] for row in train]
    X_val, y_val = [row["text"] for row in validation], [row["label"] for row in validation]
    reports, best_by_kind = [], {}
    cache: dict[str, tuple[object, object, object]] = {}
    started = time.perf_counter()
    for name, feature_kind, classifier_kind, c_value in model_candidates(args.seed):
        if feature_kind not in cache:
            features = vectorizer(feature_kind, args.seed)
            X_fit = features.fit_transform(X_train)
            X_valid = features.transform(X_val)
            cache[feature_kind] = (features, X_fit, X_valid)
        features, X_fit, X_valid = cache[feature_kind]
        if classifier_kind == "logistic":
            classifier = LogisticRegression(C=c_value, max_iter=1500, class_weight="balanced", random_state=args.seed)
        else:
            classifier = CalibratedClassifierCV(LinearSVC(C=c_value, class_weight="balanced", random_state=args.seed), cv=3)
        fit_started = time.perf_counter()
        classifier.fit(X_fit, y_train)
        pred = classifier.predict(X_valid)
        ai_index = list(classifier.classes_).index("AI_GENERATED")
        scores = classifier.predict_proba(X_valid)[:, ai_index]
        metrics = {"accuracy": float(accuracy_score(y_val, pred)),
                   "macro_f1": float(f1_score(y_val, pred, average="macro")),
                   "validation_rows": len(y_val), "fit_seconds": time.perf_counter() - fit_started}
        candidate = {"name": name, "feature_kind": feature_kind, "classifier_kind": classifier_kind,
                     "C": c_value, "model": Pipeline([("features", features), ("classifier", classifier)]),
                     "scores": scores, "metrics": metrics}
        reports.append({key: value for key, value in candidate.items() if key not in {"model", "scores"}})
        print(f"{name}: validation macro-F1={metrics['macro_f1']:.4f} accuracy={metrics['accuracy']:.4f}", flush=True)
        current = best_by_kind.get(classifier_kind)
        if current is None or metrics["macro_f1"] > current["metrics"]["macro_f1"]:
            best_by_kind[classifier_kind] = candidate

    best = best_by_kind["linear_svm"]
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], args.checkpoint, compress=3)
    thresholds = thresholds_for(best["scores"], y_val)
    metadata = {"model_name": "Expanded word/character TF-IDF " + best["classifier_kind"],
                "model_version": "svm_expanded_v1", "model_type": best["classifier_kind"],
                "checkpoint": str(args.checkpoint), "dataset_dir": str(args.dataset_dir), "seed": args.seed,
                "label2id": {"HUMAN_WRITTEN": 0, "AI_GENERATED": 1},
                "feature_kind": best["feature_kind"], "classifier_kind": best["classifier_kind"],
                "C": best["C"], "word_tfidf": {"ngram_range": [1, 2], "max_features": 150000, "min_df": 2},
                "char_tfidf": {"analyzer": "char_wb", "ngram_range": [3, 5], "max_features": 200000, "min_df": 2},
                "train_rows": len(train), "train_class_counts": dict(Counter(y_train)),
                "validation_rows": len(validation), "validation_class_counts": dict(Counter(y_val)),
                "validation_metrics": best["metrics"], "decision_thresholds": thresholds,
                "selection_metric": "expanded validation macro-F1; independent test/OOD data not used",
                "selected_svm_candidate": best["name"],
                "best_logistic_validation_candidate": best_by_kind["logistic"]["name"],
                "candidate_results": reports, "total_training_seconds": time.perf_counter() - started}
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {best['name']}; saved separate checkpoint to {args.checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
