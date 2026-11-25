# Unified frontend implementation

## Application shape

`unified.app:app` is a new local FastAPI gateway and static single-page UI.
It calls the existing Phase 1/2 detector objects from `backend/app/main.py`
without editing that app or its source, and uses `Phase3Predictor` configured
to load only the frozen copies in `phase3/models/`. The original backend entry
point still works on its own; the unified app is a separate process entrypoint
and uses the same `/api/v1/predict/image` and `/api/v1/predict/text` paths.
The unified home page provides dashboard, image, text, image+text, history,
model-status, and privacy views in one origin and port.

When no fusion checkpoint exists, the unified image+text route uses the exact
live Phase 1 and Phase 2 detector instances used by the standalone modes. This
keeps the image/text outputs consistent with their individual modes even when
the user selected checkpoint overrides. `fusion_prediction` remains JSON
`null`, `fusion_status` is `NOT_TRAINED`, and the UI says “Multimodal fusion
model is not trained yet.” The separate legacy Phase 3 `/api/v1/predict/image`
and `/api/v1/predict/text` routes still use the isolated frozen copies; its
`/api/v1/predict/multimodal` endpoint remains HTTP 503 until fusion is trained.

## Persistence and privacy

History is stored in SQLite at `unified/data/history.sqlite3` by default. The
schema is documented in `DATABASE_SCHEMA.md`. A prediction row stores the
exact model response, prediction labels, model/checkpoint version, confidence
type, processing time, UTC timestamp, status, and SHA-256 hashes of supplied
inputs. SQLite uses short-lived connections, write transactions, busy timeout,
and WAL journaling. Schema creation is idempotent (`CREATE TABLE/INDEX IF NOT
EXISTS`).

Raw text and image bytes are **not retained by default**. To opt in before
starting the server, set `$env:UNIFIED_RETAIN_INPUTS='true'`. Then original
text is stored in SQLite and uploaded image bytes are stored under
`unified/storage/images/`, which is not served as static content. Detail and
image routes require a matching history record. Deleting a record or all
history removes retained uploads as well. CSV export intentionally excludes
raw inputs and hashes.

The application binds to loopback by default and has no login/authentication.
Keep it on a trusted local machine; do not expose the history API to a public
network. Changing retention does not retroactively change existing records.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | Unified responsive UI |
| `GET /health` | Process health |
| `GET /api/v1/status` | Phase 1/2/3 checkpoint status |
| `GET /api/v1/privacy` | Input-retention and local-path settings |
| `POST /api/v1/predict/image` | Existing Phase 1 image inference plus persisted history ID |
| `POST /api/v1/predict/text` | Existing Phase 2 text inference plus persisted history ID |
| `POST /api/v1/predict/multimodal` | Phase 3 independent outputs; fusion only if checkpoint exists |
| `GET /api/v1/dashboard` | Persistent analysis counts and recent items |
| `GET /api/v1/history` | Filtered, paginated history |
| `GET /api/v1/history/{analysis_id}` | Stored response and optional retained input details |
| `GET /api/v1/history/{analysis_id}/image` | Private retained-image retrieval |
| `GET /api/v1/history/export.csv` | CSV export without raw input data |
| `DELETE /api/v1/history/{analysis_id}` | Delete one record and its retained image |
| `DELETE /api/v1/history?confirm=true` | Delete all records and retained images |

## Start and verify

In Windows PowerShell, from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn unified.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. Check status at
`http://127.0.0.1:8000/api/v1/status`. The app calls detector services in the
same process; it does not require the old backend port or Phase 3 port to be
started separately.

Optional raw-input retention:

```powershell
$env:UNIFIED_RETAIN_INPUTS = 'true'
.\.venv\Scripts\python.exe -m uvicorn unified.app:app --host 127.0.0.1 --port 8000
```

Override the default database path if needed:

```powershell
$env:UNIFIED_DB_PATH = 'D:\imad-local\history.sqlite3'
```

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest unified\tests -q
.\.venv\Scripts\python.exe -m pytest phase3\tests -q
.\.venv\Scripts\python.exe -m pytest backend\tests -q
.\.venv\Scripts\python.exe phase3\tools\snapshot_originals.py --verify
```

The project still has no trained Phase 3 fusion checkpoint because no eligible
dataset with clearly licensed, aligned, independently labeled pairs has been
approved. Dataset screening is in `phase3/reports/dataset_comparison.md` and
`phase3/reports/dataset_license_audit.md`.
