from training.calibrate_image_model import select_thresholds
from training.evaluate_image_model import calculate_thresholded_metrics
from app.schemas.prediction import ImagePredictionResponse


def test_threshold_calibration_is_reproducible_and_respects_class_error_target() -> None:
    actual = [0] * 100 + [1] * 100
    scores = [0.05 + index * 0.004 for index in range(100)] + [0.55 + index * 0.004 for index in range(100)]

    result = select_thresholds(actual, scores, max_class_error_rate=0.01)
    assert result == select_thresholds(actual, scores, max_class_error_rate=0.01)
    assert result["real_max_ai_score"] < result["ai_min_ai_score"]
    assert result["validation_real_false_positive_rate"] <= 0.01
    assert result["validation_ai_generated_false_negative_rate"] <= 0.01
    assert result["validation_coverage"] >= 0.50


def test_thresholded_evaluation_reports_uncertain_and_covered_errors() -> None:
    metrics = calculate_thresholded_metrics(
        actual=[0, 0, 1, 1],
        positive_scores=[0.1, 0.5, 0.4, 0.9],
        thresholds={"real_max_ai_score": 0.2, "ai_min_ai_score": 0.8},
    )

    assert metrics["rows_true_columns_predicted"] == [[1, 0, 1], [0, 1, 1]]
    assert metrics["uncertain_count"] == 2
    assert metrics["coverage"] == 0.5
    assert metrics["accuracy_on_covered"] == 1.0


def test_prediction_schema_accepts_uncertain_label() -> None:
    response = ImagePredictionResponse(
        label="UNCERTAIN", predicted_label="UNCERTAIN", score=0.7,
        score_type="uncalibrated_softmax", class_scores={"REAL": 0.3, "AI_GENERATED": 0.7},
        architecture="resnet18", model_version="calibrated-test", warning="review manually",
    )

    assert response.label == "UNCERTAIN"
