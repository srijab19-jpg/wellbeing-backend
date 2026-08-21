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
    DASHBOARD_ALLOWED_ORIGIN - the origin the React counsellor dashboard is
                                served from (e.g. "http://localhost:5173" in
                                dev, or your deployed dashboard's URL). Used
                                for CORS — GET/PATCH /dashboard requests
                                from any other origin are blocked by the
                                browser. Defaults to "*" (any origin) if
                                unset, which is fine for local development
                                but should be locked down before any real
                                deployment, since the dashboard shows
                                (anonymised) student risk data.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import gspread
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2.service_account import Credentials
from pydantic import BaseModel, Field

from classifier_genai_pipeline import (
    BIN_ORDERS,
    FEATURE_COLS,
    classify_student,
    encode_bin,
    generate_synthetic_dataset,
    get_genai_guidance,
    train_and_compare_models,
)

app = FastAPI(title="Student Wellbeing Check-in Backend")

# CORS: the React dashboard runs on a different origin than this API, so
# the browser blocks its fetch()/PATCH calls unless explicitly allowed
# here. See DASHBOARD_ALLOWED_ORIGIN in the module docstring above.
_allowed_origin = os.environ.get("DASHBOARD_ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_allowed_origin],
    allow_methods=["GET", "PATCH"],
    allow_headers=["*"],
)

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
    # Identifying-ish field — actually just Apps Script's unique per-response
    # ID (getId()), not a real identifier. The Form has email collection
    # turned off by design, so this is what gets hashed into anon_id instead.
    response_id: str = Field(..., description="Unique Apps-Script-assigned response ID, hashed into anon_id. Never a real student identifier.")

    # MBI-SS subscale responses (already averaged/summed client-side or raw items)
    exhaustion: float
    cynicism: float
    efficacy_reversed: float  # already reverse-scored before sending

    # ABC behavioural data — the Form collects these as multiple-choice
    # range bins (e.g. "61%-80%", "6-10"), not free-entry numbers, so the
    # raw selected label text is sent as-is and converted to an ordinal
    # bin index server-side (see BIN_ORDERS / encode_bin). Must exactly
    # match one of the labels in BIN_ORDERS for that field.
    attendance_pct: str  # e.g. "61%-80%"
    missed_deadlines: str  # e.g. "6-10"
    weekly_study_hours: str  # e.g. "21-40"; self-reported, institution has no LMS


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

# The exact tab name of your dashboard sheet within the spreadsheet
# (confirmed as "Sheet1" — if you ever rename that tab, update this or set
# the DASHBOARD_SHEET_NAME env var to match, or writes/reads will fail
# loudly with a WorksheetNotFound error instead of silently going to the
# wrong tab).
DASHBOARD_SHEET_NAME = os.environ.get("DASHBOARD_SHEET_NAME", "Sheet1")


def anonymise_response_id(response_id: str) -> str:
    """Hashes the Form's unique per-response ID (not an email — email
    collection is off by design) into a pseudonymous anon_id. Because
    Apps Script's getId() is unique per submission (not per student), this
    intentionally means the SAME student submitting twice in one term
    gets two DIFFERENT anon_ids — there is no cross-submission linking of
    one student's history, by design, since no student identifier is ever
    collected in the first place."""
    digest = hashlib.sha256((ANON_SALT + response_id.strip()).encode()).hexdigest()
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
    spreadsheet = client.open_by_key(sheet_id)
    # NOTE: deliberately NOT using spreadsheet.sheet1 here. .sheet1 means
    # "whichever tab is physically first/leftmost," not "the tab named
    # Sheet1" — and it silently breaks the moment another tab gets
    # inserted to its left (which is exactly what happened when Google
    # Forms auto-created its own "Form Responses" tab). Referencing by
    # name is position-independent and won't break again the same way.
    return spreadsheet.worksheet(DASHBOARD_SHEET_NAME)


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
            "FALSE",  # reviewed — new check-ins always start unreviewed (H6)
        ]
    )


# Column layout (1-indexed, matches append_to_dashboard's write order):
# A timestamp | B anon_id | C risk_level | D confidence |
# E top_contributing_features | F guidance_message | G suggested_resources |
# H is_fallback | I reviewed
REVIEWED_COLUMN = 9


def read_dashboard_rows() -> list[dict]:
    """Reads all check-in rows for the counsellor dashboard. Returns each
    row's actual Sheet row number (not just anon_id) as `row`, since
    anon_id is NOT a unique key — the same student can have multiple
    check-ins across the term (ANON_SALT only rotates per-term), and
    "mark reviewed" needs to target one specific check-in, not every row
    that student ever submitted.

    Tolerant of older rows written before the `reviewed` column existed
    (defaults them to not reviewed) rather than erroring."""
    sheet = get_sheet()
    all_values = sheet.get_all_values()
    if len(all_values) <= 1:
        return []  # header only, no check-ins yet

    rows = []
    for i, raw in enumerate(all_values[1:]):  # skip header row
        sheet_row_number = i + 2  # +2: 1-indexed, plus the header row
        # Pad defensively in case older rows have fewer columns than the
        # current schema (e.g. rows written before `reviewed` existed).
        padded = raw + [""] * (9 - len(raw))
        try:
            confidence = float(padded[3])
        except ValueError:
            confidence = 0.0
        rows.append(
            {
                "row": sheet_row_number,
                "id": padded[1],
                "timestamp": padded[0],
                "risk_level": padded[2],
                "confidence": confidence,
                "top_features": [f.strip() for f in padded[4].split(",") if f.strip()],
                "guidance_message": padded[5],
                "resources": [r.strip() for r in padded[6].split(",") if r.strip()],
                "reviewed": padded[8].strip().upper() == "TRUE",
            }
        )
    return rows


def set_row_reviewed(row_number: int):
    sheet = get_sheet()
    if row_number < 2:
        raise HTTPException(status_code=422, detail="Invalid row number")
    sheet.update_cell(row_number, REVIEWED_COLUMN, "TRUE")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def health_check():
    return {"status": "ok", "service": "student-wellbeing-backend"}


@app.get("/dashboard")
def get_dashboard():
    """Returns all check-in rows for the counsellor dashboard, most recent
    first. Never exposes raw identifiers — anon_id (as `id`) is already
    the anonymised pseudonym written by /checkin, and this only reads
    columns that were already anonymised at write time."""
    try:
        rows = read_dashboard_rows()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    rows.sort(key=lambda r: r["timestamp"], reverse=True)
    return rows


@app.patch("/dashboard/{row_number}")
def mark_dashboard_row_reviewed(row_number: int):
    """Marks a single check-in as reviewed by a counsellor (H6:
    human-in-the-loop before any follow-up contact). Targets the exact
    Sheet row — NOT the student's anon_id — since one student can have
    multiple check-in rows and only one should be marked here."""
    try:
        set_row_reviewed(row_number)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"row": row_number, "reviewed": True}


@app.post("/checkin", response_model=CheckinResponse)
def handle_checkin(submission: CheckinSubmission):
    anon_id = anonymise_response_id(submission.response_id)

    record = {}
    for col in FEATURE_COLS:
        value = getattr(submission, col)
        if col in BIN_ORDERS:
            try:
                record[col] = encode_bin(col, value)
            except ValueError as e:
                # A bin label that doesn't match BIN_ORDERS means either the
                # Form's option text changed or Code.gs sent something
                # unexpected. Reject loudly rather than silently coercing
                # to a wrong bin.
                raise HTTPException(status_code=422, detail=str(e))
        else:
            record[col] = value

    result = classify_student(record, MODEL_ARTIFACTS)
    guidance = get_genai_guidance(result)
    if guidance.get("is_fallback"):
        # Visible in Render's logs (Events / Logs tab) — without this, a
        # Gemini failure is invisible to the API consumer (CheckinResponse
        # never exposes fallback_reason) and to whoever's debugging.
        print(f"[warning] GenAI guidance fell back to static message. Reason: {guidance.get('fallback_reason')}")

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
