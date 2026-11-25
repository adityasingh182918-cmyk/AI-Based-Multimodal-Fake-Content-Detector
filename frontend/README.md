# Image detection browser UI

The minimal responsive interface is served by FastAPI from `backend/app/static/index.html`; no separate Node.js installation or frontend build is required.

From the repository root, activate `.venv` and start the API:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/ui`. The page checks `/model-status` and submits selected image bytes only to the local `/api/v1/predict/image` endpoint. It does not make predictions in browser code. When there is no valid trained checkpoint, it disables analysis and explains that model training is required.
