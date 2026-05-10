# Hair Doctor AI

Hair Doctor AI is a Streamlit hair and scalp screening app with:

- LangGraph analysis workflow
- Gemini image/text analysis when configured
- Local LangGraph fallback when Gemini is unavailable
- Free SQLite storage for users, consultations, results, and uploaded photo paths
- Camera capture or image upload
- No login required
- Suggested deficiency/thyroid test checklist to discuss with a doctor
- Lab report upload for possible deficiency-pattern review
- Downloadable doctor visit summary as PDF or text

This is informational screening only. It is not a medical diagnosis, prescription, or replacement for a dermatologist or qualified clinician.
The app does not tell users to start medicines. It asks users to consult a dermatologist or qualified clinician before using medicines, supplements, or medicated treatments.
Even with uploaded lab reports, the app does not claim an exact cause; it summarizes possible patterns to discuss with a clinician.

## Run Locally

Streamlit version:

```bash
python -m pip install -r requirements.txt
python -m streamlit run hair_doctor_app.py
```

React version:

```bash
python -m pip install -r requirements.txt
python -m uvicorn react_app:app --reload --port 8504
```

Then open `http://localhost:8504`.

## Gemini

Set these in `.streamlit/secrets.toml` or as environment variables:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
GEMINI_MODEL = "gemini-2.5-flash"
```

Without Gemini, the app still runs with local fallback analysis.

## Storage

Runtime data is written to:

- `data/hair_doctor.db`
- `data/hair_uploads/`

The `data/` folder is ignored by git.
