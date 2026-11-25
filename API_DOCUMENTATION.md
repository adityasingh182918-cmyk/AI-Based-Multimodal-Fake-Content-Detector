# API documentation: unified application

Start the unified application from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn unified.app:app --host 127.0.0.1 --port 8000
```

The interactive OpenAPI page is `http://127.0.0.1:8000/docs`.

## `POST /api/v1/predict/multimodal`

Accepts `multipart/form-data` with:

- `file` (required): image upload accepted by the existing image endpoint.
- `text` (optional): related caption/text, up to 50,000 characters. Empty text
  is passed to the existing text detector; depending on its minimum-length
  rules this produces a partial result rather than a combined verdict.

Both existing detectors run independently using their current configured
checkpoints and preprocessing. Their complete result objects are returned
unchanged in `image_result` and `text_result` (and compatibility fields
`image_prediction` and `text_prediction`). A rule applies to their `label`
values; it does not combine their probabilities.

Successful response fields include:

- `analysis_type`: `rule_based_multimodal`.
- `fusion_type`: `rule_based`.
- `trained_fusion_model`: `false`.
- `image_result`, `text_result`: original modality response objects, or null
  if that detector failed.
- `combined_result`: `{label, explanation}` when both detectors succeed;
  null for partial analyses.
- `analysis_status`: `completed` or `partial`.
- `failed_detectors`: map of failed modality names to safe error descriptions.
- `limitations`: notes that this is not a trained model, has no separate
  combined confidence, and does not check semantic consistency or factuality.
- `fusion_prediction`: compatibility field holding the rule label when
  available; it is not a neural fusion output.
- `fusion_status`: `RULE_BASED`.
- `processing_time_ms`, `timestamp`, and history metadata.

The decision table is documented in
[PHASE3_IMPLEMENTATION_REPORT.md](PHASE3_IMPLEMENTATION_REPORT.md). If either
detector returns `UNCERTAIN`, `combined_result.label` is `INCONCLUSIVE`. If a
detector fails, the other result remains available but no combined verdict is
returned.

Possible errors include HTTP 422 for text over the character limit or an
invalid/missing upload, and HTTP 503 for shared application/model readiness
failures that prevent request handling. Per-detector errors inside a valid
multimodal request are reported as partial results where possible.

## Existing single-modality APIs

`POST /api/v1/predict/image` and `POST /api/v1/predict/text` remain the
Phase 1/2 endpoints. Their code and response semantics are not changed by the
rule-based combination. See the existing FastAPI OpenAPI schema at `/docs` for
their request details.

## Model status

`GET /api/v1/status` reports the active image/text checkpoint readiness
and separately identifies Phase 3 as `fusion_type: rule_based`,
`trained_fusion_model: false`. The separate experimental API
`phase3.app:app` is checkpoint-based and is not the route used by the unified
frontend's rule-based combined mode.

## Privacy and history

The unified app stores result metadata and input hashes locally. Raw text and
uploaded image bytes are not retained by default. History write failures do
not turn a completed detector inference into a server error; the response
indicates the history save status.
