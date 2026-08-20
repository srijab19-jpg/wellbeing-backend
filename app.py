"""
app.py

FastAPI backend for the Student Academic Burnout Early-Intervention System.

Flow:
  Google Form submission
    -> Apps Script (Code.gs) onFormSubmit trigger
    -> POST /checkin  (this file)
        1. Anonymise the incoming record (strip identifying fields,
           assign a rotating pseudonymous ID)
        2. Run Stage 1 classification (Logistic Regression, primary)
        3. Run Stage 2 GenAI guidance (constrained, validated, fails closed)
        4. Write ONLY the anonymised + classified + guidance output to the
           Google Sheet dashboard (never raw identifiers, never free text
           answers)
    -> Google Sheet ("Risk Dashboard") updates
    -> Counsellor views dashboard, human-in-the-loop review before any
       follow-up contact (H6)

Deployment target: Render free tier (see project's build guide).
Required environment variables (set in Render's dashboard, never in code):
    GOOGLE_AI_API_KEY         - Google AI Studio (Gemini) API key, free tier
    GOOGLE_SERVICE_ACCOUNT_JSON - contents of a Google service account
                                  JSON key, as a single-line string, with
                                  edit access to the "Risk Dashboard" sheet
    SHEET_ID                 - the target Google Sheet's ID (from its URL)
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import gspread
from fastapi import FastAPI, HTTPException
from google.oauth2.service_account import Credentials
from pydantic import BaseModel, Field

from classifier_genai_pipeline import (
    FEATURE_COLS,
    classify_student,
    generate_synthetic_dataset,
    get_genai_guidance,
    train_and_compare_models,
)

app = FastAPI(title="Student Wellbeing Check-in Backend")

# ---------------------------------------------------------------------------
# Startup: train the classifier once at boot (synthetic data placeholder —
# swap generate_synthetic_dataset() for a real anonymised dataset when
# available; see project open items).
# ---------------------------------------------------------------------------

_dataset = generate_synthetic_dataset()
_, MODEL_ARTIFACTS = train_and_compare_models(_dataset)


# ---------------------------------------------------------------------------
# Request schema — mirrors the Google Form fields
# ---------------------------------------------------------------------------


class CheckinSubmission(BaseModel):
    # Identifying field — used ONLY to derive an anonymised hash, then discarded.
    respondent_email: str = Field(..., description="Used only to derive a rotating pseudonymous ID; never stored raw.")

    # MBI-SS subscale responses (already averaged/summed client-side or raw items)
    exhaustion: float
    cynicism: float
    efficacy_reversed: float  # already reverse-scored before sending

    # ABC behavioural data
    attendance_pct: float
    missed_deadlines: int
    weekly_study_hours: float  # self-reported; institution has no LMS


class CheckinResponse(BaseModel):
    risk_level: str
    confidence: float
    top_contributing_features: list[str]
    guidance_message: str
    suggested_resources: list[str]
    is_fallback: bool


# ---------------------------------------------------------------------------
# Anonymisation helper
# ---------------------------------------------------------------------------

# Rotate this salt periodically (e.g. once per term) so the pseudonymous ID
# cannot be correlated indefinitely across terms, while still allowing
# within-term follow-up matching if a counsellor needs it.
ANON_SALT = os.environ.get("ANON_SALT", "change-me-per-term")


def anonymise_email(email: str) -> str:
    digest = hashlib.sha256((ANON_SALT + email.strip().lower()).encode()).hexdigest()
    return f"student_{digest[:12]}"


# ---------------------------------------------------------------------------
# Google Sheets writer
# ---------------------------------------------------------------------------


def get_sheet():
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id = os.environ.get("SHEET_ID")
    if not creds_json or not sheet_id:
        raise RuntimeError("Google Sheets credentials or SHEET_ID not configured")

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_key(sheet_id).sheet1


def append_to_dashboard(anon_id: str, result: dict, guidance: dict):
    sheet = get_sheet()
    sheet.append_row(
        [
            datetime.now(timezone.utc).isoformat(),
            anon_id,
            result["risk_level"],
            result["confidence"],
            ", ".join(result["top_contributing_features"]),
            guidance["message"],
            ", ".join(guidance["suggested_resources"]),
            guidance["is_fallback"],
        ]
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def health_check():
    return {"status": "ok", "service": "student-wellbeing-backend"}


@app.post("/checkin", response_model=CheckinResponse)
def handle_checkin(submission: CheckinSubmission):
    anon_id = anonymise_email(submission.respondent_email)

    record = {col: getattr(submission, col) for col in FEATURE_COLS}
    result = classify_student(record, MODEL_ARTIFACTS)
    guidance = get_genai_guidance(result)

    try:
        append_to_dashboard(anon_id, result, guidance)
    except Exception as e:
        # Classification + guidance still succeeded; log the sheet failure
        # but don't fail the whole request just because the dashboard
        # write failed (e.g. Sheets creds not yet configured during setup).
        print(f"[warning] failed to write to dashboard sheet: {e}")

    return CheckinResponse(
        risk_level=result["risk_level"],
        confidence=result["confidence"],
        top_contributing_features=result["top_contributing_features"],
        guidance_message=guidance["message"],
        suggested_resources=guidance["suggested_resources"],
        is_fallback=guidance["is_fallback"],
    )
