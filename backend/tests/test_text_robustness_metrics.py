from training.evaluate_text_robustness import read_rows, score_report
from training.evaluate_text_additional_robustness import load_beemo_expert_edits, load_gradtex_test_c

import pytest


def test_mage_label_mapping_scores_ai_probability_in_correct_direction():
    report = score_report([0, 1], [0.9, 0.1])
    assert report["accuracy"] == 1.0
    assert report["roc_auc_ai_generated"] == 1.0
    assert report["confusion_matrix"] == [[1, 0], [0, 1]]
    assert report["class_support"] == {"AI_GENERATED": 1, "HUMAN_WRITTEN": 1}


def test_mage_single_class_paraphrase_slice_omits_undefined_auc(tmp_path):
    path = tmp_path / "paraphrase.csv"
    path.write_text("text,label,src\nexample AI text,0,cnn_gpt4_para\noriginal human text,1,cnn_human\n", encoding="utf-8")
    rows = read_rows(path)
    assert rows[0]["label"] == 0
    ai_rows = [row for row in rows if row["src"].endswith("_gpt4_para")]
    report = score_report([row["label"] for row in ai_rows], [0.8])
    assert report["ai_generated_recall"] == 1.0
    assert report["macro_f1"] is None
    assert report["accuracy"] is None
    assert report["ai_generated_precision"] is None
    assert report["roc_auc_ai_generated"] is None
    assert report["false_negative_rate_ai_as_human"] == 0.0


def test_gradtex_loader_only_uses_test_c_and_maps_mix_as_machine_involved(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "gradtex.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"text": "Pure human text from an independent editor source.", "binary_label": 1, "multiclass_label": "HWT",
         "domain": "news", "scenario": "human", "scenario_family": "human", "generator_model": None,
         "model_family": None, "test_split": "C", "evaluation_setup": "unseen_scenarios_evaluation"},
        {"text": "AI involved text after a held out model transformation.", "binary_label": 0, "multiclass_label": "MIX",
         "domain": "news", "scenario": "polish_sentence", "scenario_family": "polish", "generator_model": "g1",
         "model_family": "family1", "test_split": "C", "evaluation_setup": "unseen_scenarios_evaluation"},
        {"text": "An in-domain item must be ignored by this loader.", "binary_label": 0, "multiclass_label": "MIX",
         "domain": "news", "scenario": "paraphrase", "scenario_family": "paraphrase", "generator_model": "g1",
         "model_family": "family1", "test_split": "B", "evaluation_setup": "in_domain_evaluation"},
    ]), path)
    rows = load_gradtex_test_c(path)
    assert len(rows) == 2
    assert {row["label_name"] for row in rows} == {"HWT", "MIX"}
    assert {row["label"] for row in rows} == {0, 1}


def test_beemo_loader_selects_only_expert_edits_from_permissive_model_sources(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "beemo.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"id": 1, "category": "Rewrite", "model": "HuggingFaceH4/zephyr-7b-beta",
         "model_output": "AI original response", "human_edits": "Expert revised AI response"},
        {"id": 2, "category": "Open QA", "model": "meta-llama/Llama-2-7b-chat-hf",
         "model_output": "Restricted-source model output", "human_edits": "Restricted-source edit"},
    ]), path)
    rows = load_beemo_expert_edits(path)
    assert len(rows) == 1
    assert rows[0]["text"] == "Expert revised AI response"
    assert rows[0]["label"] == 0


def test_score_report_has_explicit_false_positive_and_negative_rates():
    report = score_report([0, 0, 1, 1], [.9, .1, .8, .2])
    assert report["false_negative_rate_ai_as_human"] == .5
    assert report["false_positive_rate_human_as_ai"] == .5
