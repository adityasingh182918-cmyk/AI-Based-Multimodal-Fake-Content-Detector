# Phase 2 architecture

Phase 2 adds a separate text path alongside the existing Phase 1 ResNet-18 image path:

```text
HC3 raw JSONL -> prepare + group split -> TF-IDF word/character -> Logistic Regression / calibrated Linear SVM
                       \-> DistilBERT tokenizer -> fine-tuned sequence classifier -> validation-selected checkpoint
POST /api/v1/predict/text -> selected TextDetector (lazy load) -> chunk/window probabilities -> HUMAN_WRITTEN / AI_GENERATED / UNCERTAIN
```

`POST /api/v1/predict/image` remains unchanged. `/text-model-status` reports text checkpoint availability. The browser page has independent image, text, and image-plus-text modes. Combined mode makes two independent requests and displays the separate outputs, then compares their labels. It explicitly says **PRELIMINARY COMBINED ASSESSMENT – NOT A TRAINED FUSION MODEL**; it does not average probabilities.

The default baseline uses word bigrams and character 3–5 grams with TF-IDF. Linear SVM probabilities use `CalibratedClassifierCV`; a validation-only quantile rule defines an abstention band. Long input is split in order into chunks up to 2,000 characters and scores are averaged by word count. Optional DistilBERT uses tokenizer overflow windows up to 256 tokens with overlap and token-count weighted aggregation. Neither aggregation is validated as a document-level classifier. Requests require at least 40 characters and five words and are limited to 50,000 characters.

The baseline checkpoint is a scikit-learn joblib artifact; DistilBERT uses Transformers `save_pretrained`. Load only trusted local checkpoints. Text inference is independent from the image service. Cross-generator evaluation, robustness attacks, frontend browser automation, and actual Phase 3 fusion remain incomplete.
