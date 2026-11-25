# Phase 3 paired data

No dataset has been downloaded or placed here. A fusion training example must
be an authentic, documented image/caption pair with independently verified
image origin (`REAL` or `AI_GENERATED`) and text origin (`HUMAN_WRITTEN` or
`AI_GENERATED`). Do not pair unrelated image and text samples.

Create `train.jsonl`, `validation.jsonl`, and `test.jsonl` under `paired/`.
Each line is JSON with these required fields:

```json
{"sample_id":"dataset-record-id","group_id":"original-document-or-prompt-id","split":"train","image_path":"images/record.jpg","image_origin":"REAL","text_origin":"HUMAN_WRITTEN","text":"caption content","source":"dataset/source split","dataset_provenance":"source URL, version, and license","relation":"documented original pair relationship"}
```

The JSON above is a schema example only, not a real or labeled training item.
Paths in `image_path` are relative to `phase3/data/`. Keep related records and
near duplicates within one group and one split. The preparation script refuses
duplicate sample IDs and group leakage across train/validation/test. Training
requires examples from all four origin combinations in the training split.

Data files are ignored by `phase3/.gitignore`; keep source data outside Git.
