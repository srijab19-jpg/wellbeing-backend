"""
classifier_genai_pipeline.py

Standalone, run-in-Colab pipeline for the Student Academic Burnout
Early-Intervention System (course project prototype).

Stage 1 — Classification:
    Inputs: MBI-SS self-report subscales (Exhaustion, Cynicism, Reduced
    Academic Efficacy) + ABC academic-behavioural data (Attendance Rate,
    Missed Deadlines, Weekly Study Hours).
    Primary model: Multinomial Logistic Regression (interpretable — used
    for viva coefficient explanations).
    Secondary model: Random Forest (comparison only, per H9 — accuracy/F1/
    AUC reported side-by-side, LR remains primary regardless of which
    scores higher, for interpretability reasons).

Stage 2 — GenAI Guidance:
    Constrained call to Google Gemini (free tier, via Google AI Studio). The
    model NEVER sets the risk level itself, NEVER invents a support
    resource outside the fixed approved list, and must return valid
    structured JSON. If validation fails for any reason, the pipeline
    fails closed to a static fallback message — a student is never shown
    unvalidated model output.

NOTE ON DATA: This script uses synthetic data for demonstration. Sourcing
a real (anonymised) dataset is a listed open item for this project.

NOTE ON WEEKLY STUDY HOURS: The institution has no LMS, only a Student
Information System (SIS) used for attendance lookup. Weekly Study Hours
is therefore an honestly self-reported field (grounded in Cuevas-Caravaca,
Sánchez-Romero & Antón-Ruiz, 2024), NOT an "LMS engagement" proxy. Only
Attendance Rate and Missed Deadlines are treated as objective/behavioural
if sourced from SIS/registrar/grade-book records.
"""

import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
RISK_LEVELS = ["none", "low", "medium", "high"]

# ---------------------------------------------------------------------------
# Behavioral fields are collected on the live Google Form as multiple-choice
# RANGE BINS, not free-entry numbers (e.g. a student picks "61%-80%", not a
# specific percentage). Rather than fabricate a fake continuous number from
# a bin (midpoint/lower-bound guessing), these are treated as ordinal
# categories: each bin label maps to its position in the ordered list
# below (0 = lowest/first bin). This is what the data actually is, and it's
# what both training (synthetic data, below) and serving (classify_student)
# use consistently.
#
# NOTE: because these are now ordinal bin indices rather than real-world
# units, the Logistic Regression coefficients for these three features will
# read as "per bin step" (e.g. moving from the "0-5" to "6-10" missed-
# deadlines bin), not "per percentage point" or "per hour" as before. Keep
# that in mind for the viva coefficient explanation.
BIN_ORDERS = {
    # NOTE: the Form's option text is inconsistently formatted (verified
    # against screenshots of the live Form, not assumed) — some options
    # have spaces around the dash and a "%" after both numbers, others
    # don't. These strings must match EXACTLY, including that
    # inconsistency, or encode_bin() will correctly reject the answer.
    "attendance_pct": ["0-20%", "21% - 40%", "41% - 60%", "61%-80%", "81% - 100%"],
    "missed_deadlines": ["0-5", "6-10", "11-15", "16-20"],
    "weekly_study_hours": ["0-20", "21-40", "41-60", "61-80"],
}


def encode_bin(field: str, label: str) -> int:
    """Converts a Form answer's exact label text into its ordinal position
    within BIN_ORDERS[field]. Raises ValueError on an unrecognised label
    (e.g. the Form's option text was edited and this file wasn't updated
    to match) rather than silently guessing — a wrong ordinal position is
    worse than a loud failure here."""
    try:
        order = BIN_ORDERS[field]
    except KeyError:
        raise ValueError(f"Unknown bin field: {field!r}")
    label = label.strip()
    if label not in order:
        raise ValueError(
            f"Unrecognised bin label {label!r} for field {field!r}. "
            f"Expected one of: {order}. If the Form's option text changed, "
            f"update BIN_ORDERS to match."
        )
    return order.index(label)

# Approved, fixed support-resource list. The GenAI layer may only choose
# from this list — it can never invent a resource.
APPROVED_RESOURCES = [
    "Student Counselling Centre (drop-in hours, Mon-Fri 10am-4pm)",
    "Academic Advising Office (course-load / deadline extension support)",
    "Peer Mentoring Programme",
    "Time-Management & Study Skills Workshop (weekly, Wed 3pm)",
    "Crisis Support Line (24/7, institution-provided)",
]

STATIC_FALLBACK_GUIDANCE = {
    "risk_level_acknowledged": None,  # filled in at runtime
    "message": (
        "Thanks for completing your check-in. Based on your responses, we'd "
        "like to connect you with some support options. Please reach out to "
        "the Student Counselling Centre or Academic Advising Office when "
        "convenient — they're there to help, no judgement."
    ),
    "suggested_resources": [
        "Student Counselling Centre (drop-in hours, Mon-Fri 10am-4pm)",
        "Academic Advising Office (course-load / deadline extension support)",
    ],
    "is_fallback": True,
}

# ---------------------------------------------------------------------------
# 1. SYNTHETIC DATA (placeholder until a real dataset is sourced)
# ---------------------------------------------------------------------------


def generate_synthetic_dataset(n=600, seed=RANDOM_STATE):
    """Generates synthetic student check-in records for demo purposes."""
    rng = np.random.default_rng(seed)

    exhaustion = rng.integers(0, 7, n)  # 7-pt frequency scale, MBI-SS aligned
    cynicism = rng.integers(0, 7, n)
    efficacy = rng.integers(0, 7, n)  # NOTE: reverse-scored at analysis
    efficacy_reversed = 6 - efficacy

    # Ordinal bin indices — matches what the live Form actually collects
    # (a bin label, not a raw percentage/count/hour figure). 0 = the
    # lowest/first bin in BIN_ORDERS for that field.
    n_attendance_bins = len(BIN_ORDERS["attendance_pct"])  # 5
    n_deadline_bins = len(BIN_ORDERS["missed_deadlines"])  # 4
    n_study_bins = len(BIN_ORDERS["weekly_study_hours"])  # 4

    attendance_pct = rng.integers(0, n_attendance_bins, n)  # bin index; higher = better attendance
    missed_deadlines = rng.integers(0, n_deadline_bins, n)  # bin index; higher = more missed
    weekly_study_hours = rng.integers(0, n_study_bins, n)  # bin index; higher = more hours

    # crude synthetic burnout score to derive labels (for demo only —
    # a real dataset should replace this generation step entirely).
    # Attendance and study-hours bins run "better outcome = higher index",
    # so they're inverted (max_index - value) to contribute positively to
    # risk when LOW; missed-deadlines bin already runs "worse = higher index".
    burnout_score = (
        0.35 * exhaustion
        + 0.30 * cynicism
        + 0.20 * efficacy_reversed
        + 0.10 * (missed_deadlines * 1.2)
        + 0.05 * ((n_attendance_bins - 1 - attendance_pct) * 2)
        - 0.03 * (weekly_study_hours * 1.5)
    )
    burnout_score += rng.normal(0, 1.0, n)  # noise

    bins = np.quantile(burnout_score, [0.25, 0.5, 0.75])
    risk_level = np.digitize(burnout_score, bins)  # 0..3 -> none..high
    risk_label = np.array(RISK_LEVELS)[risk_level]

    df = pd.DataFrame(
        {
            "exhaustion": exhaustion,
            "cynicism": cynicism,
            "efficacy_reversed": efficacy_reversed,
            "attendance_pct": attendance_pct,
            "missed_deadlines": missed_deadlines,
            "weekly_study_hours": weekly_study_hours,
            "risk_level": risk_label,
        }
    )
    return df


# ---------------------------------------------------------------------------
# 2. STAGE 1: CLASSIFICATION (Logistic Regression primary, RF secondary)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "exhaustion",
    "cynicism",
    "efficacy_reversed",
    "attendance_pct",
    "missed_deadlines",
    "weekly_study_hours",
]


def train_and_compare_models(df):
    """Trains LR (primary) and RF (secondary comparison, H9) and reports
    accuracy / F1 / AUC for both, side by side."""

    X = df[FEATURE_COLS].values
    y_raw = df["risk_level"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)  # alphabetical: high, low, medium, none — reorder below
    class_order = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Primary model: Logistic Regression ---
    # Interpretable: coefficients map directly to "feature +1 unit -> risk
    # increases/decreases by X%" explanations for the viva.
    # NOTE: newer scikit-learn (1.5+) infers multinomial automatically for
    # solvers that support it (lbfgs, default) — no multi_class param needed.
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)

    # --- Secondary comparison model: Random Forest (H9) ---
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)

    def safe_auc(y_true, proba):
        try:
            return roc_auc_score(y_true, proba, multi_class="ovr")
        except ValueError:
            return float("nan")  # e.g. a class missing from the test split

    results = {
        "logistic_regression_primary": {
            "accuracy": accuracy_score(y_test, lr_pred),
            "f1_macro": f1_score(y_test, lr_pred, average="macro"),
            "auc_ovr": safe_auc(y_test, lr_proba),
        },
        "random_forest_secondary": {
            "accuracy": accuracy_score(y_test, rf_pred),
            "f1_macro": f1_score(y_test, rf_pred, average="macro"),
            "auc_ovr": safe_auc(y_test, rf_proba),
        },
    }

    # Coefficient-level explanation for the primary model (viva-ready)
    coef_table = pd.DataFrame(
        lr.coef_, columns=FEATURE_COLS, index=[class_order[i] for i in range(len(class_order))]
    )

    artifacts = {
        "lr_model": lr,
        "rf_model": rf,
        "scaler": scaler,
        "label_encoder": le,
        "class_order": class_order,
        "coef_table": coef_table,
    }

    return results, artifacts


def classify_student(record: dict, artifacts: dict):
    """Runs a single student's check-in through the PRIMARY (Logistic
    Regression) model and returns risk_level, confidence, and top
    contributing features. This — not the RF model — is what feeds the
    GenAI guidance layer, per the project's stated design (H9)."""

    x = np.array([[record[col] for col in FEATURE_COLS]])
    x_scaled = artifacts["scaler"].transform(x)

    proba = artifacts["lr_model"].predict_proba(x_scaled)[0]
    pred_idx = int(np.argmax(proba))
    class_order = artifacts["class_order"]
    risk_level = class_order[pred_idx]
    confidence = float(proba[pred_idx])

    # Top contributing features: coefficient * standardised feature value,
    # for the predicted class row.
    coef_row = artifacts["coef_table"].loc[risk_level]
    contributions = (coef_row.values * x_scaled[0])
    contrib_series = pd.Series(contributions, index=FEATURE_COLS)
    top_features = contrib_series.abs().sort_values(ascending=False).index[:3].tolist()

    return {
        "risk_level": risk_level,
        "confidence": round(confidence, 3),
        "top_contributing_features": top_features,
    }


# ---------------------------------------------------------------------------
# 3. STAGE 2: GENAI GUIDANCE LAYER (constrained, validated, fails closed)
# ---------------------------------------------------------------------------


def build_guidance_prompt(classification_result: dict) -> str:
    """Builds a constrained prompt. The model is told explicitly: do not
    set the risk level, only choose from the approved resource list,
    return ONLY valid JSON."""

    resources_block = "\n".join(f"- {r}" for r in APPROVED_RESOURCES)

    return f"""You are a supportive guidance-message generator for a student
wellbeing check-in system. You do NOT diagnose, you do NOT set or change
the risk level (it has already been determined by a separate classifier),
and you must ONLY recommend resources from the approved list below —
never invent a new resource or suggest anything outside this list.

Risk level (already determined, do not change): {classification_result['risk_level']}
Confidence: {classification_result['confidence']}
Top contributing factors: {', '.join(classification_result['top_contributing_features'])}

Approved resources (choose 1-2 most relevant, verbatim from this list):
{resources_block}

Return ONLY valid JSON, no markdown fences, no preamble, in exactly this shape:
{{
  "message": "<a warm, brief, non-clinical 2-3 sentence message to the student>",
  "suggested_resources": ["<resource 1 from the approved list>", "<resource 2 if relevant>"]
}}"""


def validate_guidance_response(raw_text: str) -> dict | None:
    """Validates the GenAI response. Returns the parsed dict if valid,
    or None if validation fails for any reason (triggering fail-closed
    fallback)."""

    try:
        parsed = json.loads(raw_text.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    if "message" not in parsed or "suggested_resources" not in parsed:
        return None
    if not isinstance(parsed["message"], str) or not parsed["message"].strip():
        return None
    if not isinstance(parsed["suggested_resources"], list):
        return None
    if not (1 <= len(parsed["suggested_resources"]) <= 2):
        return None
    for resource in parsed["suggested_resources"]:
        if resource not in APPROVED_RESOURCES:
            return None  # model invented or altered a resource -> reject

    return parsed


def get_genai_guidance(classification_result: dict) -> dict:
    """Calls the Google Gemini API (free tier: Flash / Flash-Lite models via
    Google AI Studio) for guidance text. Fails closed to a static fallback
    if the API call fails OR if the response fails validation.
    Requires GOOGLE_AI_API_KEY to be set as an environment variable
    (Colab secret / .env / Render env var — never hardcoded).

    NOTE: Google's free tier applies to specific models (Flash / Flash-Lite)
    with daily/per-minute request caps that Google can change without much
    notice — check https://ai.google.dev/gemini-api/docs/rate-limits for
    current limits before relying on this for anything beyond a course
    project demo."""

    prompt = build_guidance_prompt(classification_result)

    try:
        from google import genai  # pip install google-genai --break-system-packages

        api_key = os.environ.get("GOOGLE_AI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_AI_API_KEY not set")

        client = genai.Client(api_key=api_key)
        # gemini-2.5-flash-lite: currently the most generous free-tier model.
        # Swap to "gemini-2.5-flash" if you want slightly higher quality at a
        # lower daily request cap — check current limits before switching.
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt
        )
        raw_text = response.text

        validated = validate_guidance_response(raw_text)
        if validated is None:
            raise ValueError("GenAI response failed validation")

        validated["is_fallback"] = False
        validated["risk_level_acknowledged"] = classification_result["risk_level"]
        return validated

    except Exception as e:
        # Fail closed — a student is never shown an unvalidated response.
        fallback = dict(STATIC_FALLBACK_GUIDANCE)
        fallback["risk_level_acknowledged"] = classification_result["risk_level"]
        fallback["fallback_reason"] = str(e)
        return fallback


# ---------------------------------------------------------------------------
# 4. END-TO-END DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Stage 1: Training + comparing classifiers (H9) ===")
    dataset = generate_synthetic_dataset()
    results, artifacts = train_and_compare_models(dataset)

    print("\nModel comparison (test set):")
    print(json.dumps(results, indent=2))
    print(
        "\nNote: even if Random Forest scores marginally higher, Logistic "
        "Regression remains the PRIMARY model for this system, for "
        "interpretability (coefficient-level explanations for the viva)."
    )

    print("\nLogistic Regression coefficients (per risk class):")
    print(artifacts["coef_table"].round(3))

    print("\n=== Stage 2: Example end-to-end student check-in ===")
    # attendance_pct / missed_deadlines / weekly_study_hours come from the
    # Form as bin LABELS (what app.py receives) and must go through
    # encode_bin() before reaching the classifier — this mirrors what
    # app.py's handle_checkin() does for a real submission.
    example_student = {
        "exhaustion": 5,
        "cynicism": 4,
        "efficacy_reversed": 4,  # i.e. low reported efficacy
        "attendance_pct": encode_bin("attendance_pct", "41% - 60%"),
        "missed_deadlines": encode_bin("missed_deadlines", "11-15"),
        "weekly_study_hours": encode_bin("weekly_study_hours", "0-20"),
    }

    classification = classify_student(example_student, artifacts)
    print("\nClassification result:")
    print(json.dumps(classification, indent=2))

    guidance = get_genai_guidance(classification)
    print("\nGenAI guidance (validated, fails closed if needed):")
    print(json.dumps(guidance, indent=2))
