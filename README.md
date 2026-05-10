# Hair Doctor AI

Hair Doctor AI is a Streamlit hair and scalp screening app with:

- Google authentication through Streamlit OIDC
- LangGraph analysis workflow
- Gemini image/text analysis when configured
- Local LangGraph fallback when Gemini is unavailable
- Free SQLite storage for users, consultations, results, and uploaded photo paths
- Camera capture or image upload

This is informational screening only. It is not a medical diagnosis, prescription, or replacement for a dermatologist or qualified clinician.

## Run Locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run hair_doctor_app.py
```

## Google Login

1. Create a Google OAuth web client in Google Cloud.
2. Add this redirect URI for local testing:

```text
http://localhost:8501/oauth2callback
```

3. Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`.
4. Fill in `client_id`, `client_secret`, and `cookie_secret`.

If Google auth is not configured, the app runs in local guest mode.

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
