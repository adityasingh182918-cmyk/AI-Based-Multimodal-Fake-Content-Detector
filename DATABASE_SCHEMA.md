# Unified history database schema

The unified application creates `prediction_history` in SQLite using
idempotent `CREATE TABLE IF NOT EXISTS` and indexed lookups. The default file is
`unified/data/history.sqlite3`; configure `UNIFIED_DB_PATH` to relocate it.
The directory is ignored by Git. Existing records are not recomputed when
detector checkpoints change.

| Column | Type | Purpose |
|---|---|---|
| `id` | integer primary key | Local row key |
| `analysis_id` | unique text | Public random UUID-like record identifier |
| `phase` | integer 1, 2, or 3 | Detector phase used |
| `detection_type` | text | `image`, `text`, or `multimodal` |
| `image_filename` | text nullable | Sanitized basename from upload metadata |
| `image_storage_path` | text nullable | Private upload path only when retention is enabled |
| `image_hash`, `text_hash` | text nullable | SHA-256 content hashes; no raw bytes/text required |
| `input_text` | text nullable | Original text only when retention is enabled |
| `primary_prediction` | text nullable | Main phase result or `FUSION_UNAVAILABLE` for a partial Phase 3 result |
| `image_prediction`, `text_prediction`, `fusion_prediction` | text nullable | Separate modality outputs |
| `confidence` | real nullable | Score attached to the main prediction if provided |
| `confidence_type` | text nullable | Calibrated probability or raw softmax designation |
| `model_name`, `model_version` | text nullable | Model identity captured at inference time |
| `processing_time_ms` | real nullable | Recorded API inference duration |
| `created_at` | ISO-8601 text | UTC creation time |
| `status` | text | `completed` or `partial`; errors are returned without a false successful record |
| `error_message` | text nullable | Reserved for an explicitly recorded failed analysis |
| `response_json` | text | Exact inference response at the time of analysis |

Indexes cover creation time, phase/type, and primary prediction. Writes use
`BEGIN IMMEDIATE`/commit and foreign keys plus a busy timeout. The implementation
uses SQLite WAL journaling. Each API read/write opens and closes its own short-
lived connection. Raw data retention defaults off; enabling it is an explicit
environment setting. The image storage directory is private and not mounted as
static content. CSV export excludes original inputs and hashes.

There is no account or authentication layer. Bind the server to `127.0.0.1`
unless authentication and access controls are added before network exposure.
