from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, TypedDict

import requests

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - lets the UI show a helpful setup hint.
    END = "__end__"
    StateGraph = None


@dataclass
class UserProfile:
    subject: str
    name: str
    age: int
    gender: str
    hair_fall_level: int
    duration: str
    pattern: str
    dandruff: str
    scalp_symptoms: str
    stress: int
    sleep_hours: float
    diet_quality: str
    family_history: str
    chemical_treatments: str
    recent_illness: str
    medications: str
    notes: str = ""


@dataclass
class AnalysisResult:
    risk_score: int
    risk_level: str
    confidence: int
    likely_causes: list[str]
    needed_actions: list[str]
    follow_up_questions: list[str]
    photo_observations: list[str]
    summary: str
    disclaimer: str
    analysis_engine: str


class AnalysisState(TypedDict, total=False):
    profile: UserProfile
    image_bytes: bytes | None
    image_mime: str | None
    image_note: str
    result: AnalysisResult
    needs_clinician_review: bool


SYSTEM_PROMPT = """
You are a cautious AI hair and scalp screening assistant. Return only JSON.

Required JSON schema:
{
  "risk_score": 0,
  "risk_level": "Low",
  "confidence": 0,
  "likely_causes": ["short cause 1", "short cause 2", "short cause 3"],
  "needed_actions": ["short action 1", "short action 2", "short action 3"],
  "follow_up_questions": ["question 1", "question 2"],
  "photo_observations": ["visible observation 1", "visible observation 2"],
  "summary": "2 short sentences"
}

Rules:
- This is not a diagnosis and must not replace a dermatologist.
- Keep the output short, practical, and specific.
- Say "likely" or "may" for causes.
- Never identify a person from the photo.
- Recommend urgent or clinician review for patchy loss, scalp pain, infection signs, sudden severe shedding,
  children, pregnancy/postpartum concerns, or medication-related shedding.
""".strip()


def analyze_hair_case(
    profile: UserProfile,
    image_bytes: bytes | None = None,
    image_mime: str | None = None,
) -> AnalysisResult:
    """Run the LangGraph hair screening workflow."""
    graph = build_hair_graph()
    final_state = graph.invoke(
        {
            "profile": profile,
            "image_bytes": image_bytes,
            "image_mime": image_mime,
        }
    )
    return final_state["result"]


def build_hair_graph():
    if StateGraph is None:
        return _DirectHairGraph()

    graph = StateGraph(AnalysisState)
    graph.add_node("prepare_photo_context", prepare_photo_context)
    graph.add_node("generate_analysis", generate_analysis)
    graph.add_node("review_safety", review_safety)
    graph.set_entry_point("prepare_photo_context")
    graph.add_edge("prepare_photo_context", "generate_analysis")
    graph.add_edge("generate_analysis", "review_safety")
    graph.add_edge("review_safety", END)
    return graph.compile()


class _DirectHairGraph:
    def invoke(self, state: AnalysisState) -> AnalysisState:
        state = prepare_photo_context(state)
        state = generate_analysis(state)
        return review_safety(state)


def prepare_photo_context(state: AnalysisState) -> AnalysisState:
    state["image_note"] = (
        "Hair/scalp photo was supplied for visual screening."
        if state.get("image_bytes")
        else "No photo was supplied; analysis is based on questionnaire only."
    )
    return state


def generate_analysis(state: AnalysisState) -> AnalysisState:
    api_key = get_config_value("GEMINI_API_KEY")
    if api_key:
        try:
            state["result"] = analyze_with_gemini(
                state["profile"],
                state.get("image_bytes"),
                state.get("image_mime"),
                api_key,
            )
            return state
        except Exception as exc:
            state["image_note"] = f"{state.get('image_note', '')} Gemini unavailable: {exc}"

    state["result"] = heuristic_analysis(
        state["profile"],
        has_photo=bool(state.get("image_bytes")),
    )
    return state


def review_safety(state: AnalysisState) -> AnalysisState:
    result = state["result"]
    profile = state["profile"]
    triggers = [
        profile.pattern in {"Patchy spots", "Sudden heavy shedding"},
        "pain" in profile.scalp_symptoms.lower(),
        "burn" in profile.scalp_symptoms.lower(),
        "pus" in profile.scalp_symptoms.lower(),
        profile.age < 16,
    ]
    state["needs_clinician_review"] = any(triggers) or result.risk_level == "High"
    if state["needs_clinician_review"]:
        action = "Book a dermatologist or qualified clinician review, especially if this is sudden, painful, patchy, or worsening."
        if action not in result.needed_actions:
            result.needed_actions = [action] + result.needed_actions[:3]
    state["result"] = result
    return state


def analyze_with_gemini(
    profile: UserProfile,
    image_bytes: bytes | None,
    image_mime: str | None,
    api_key: str,
) -> AnalysisResult:
    model = get_config_value("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    parts: list[dict[str, Any]] = [
        {
            "text": "Analyze this hair/scalp screening profile:\n"
            + json.dumps(asdict(profile), indent=2)
        }
    ]
    if image_bytes and image_mime:
        parts.append(
            {
                "inline_data": {
                    "mime_type": image_mime,
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }
            }
        )

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.25,
            "responseMimeType": "application/json",
        },
    }
    response = requests.post(endpoint, params={"key": api_key}, json=payload, timeout=45)
    response.raise_for_status()
    data = response.json()
    raw_text = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )
    return normalize_result(parse_json_block(raw_text), "LangGraph + Gemini")


def heuristic_analysis(profile: UserProfile, has_photo: bool) -> AnalysisResult:
    score = 10 + profile.hair_fall_level * 6 + profile.stress * 3
    causes: list[str] = []
    actions: list[str] = []
    questions: list[str] = []
    observations = ["Photo included; local fallback cannot inspect image details."] if has_photo else []

    if profile.duration in {"More than 6 months", "More than 1 year"}:
        score += 10
        causes.append("Long duration may suggest pattern thinning or an ongoing trigger.")
        questions.append("Has the hairline, crown, or part width changed over time?")
    if profile.pattern in {"Hairline / temples", "Crown thinning", "Part line widening"}:
        score += 12
        causes.append("The pattern may fit hereditary or hormone-sensitive thinning.")
        actions.append("Track monthly photos in the same lighting and discuss early treatment options with a dermatologist.")
    if profile.pattern in {"Patchy spots", "Sudden heavy shedding"}:
        score += 18
        causes.append("Sudden or patchy loss needs medical review to rule out inflammatory or autoimmune causes.")
    if profile.dandruff in {"Moderate", "Severe"} or profile.scalp_symptoms != "None":
        score += 10
        causes.append("Scalp irritation or dandruff may be increasing shedding or breakage.")
        actions.append("Use a gentle anti-dandruff routine and avoid scratching or harsh oils until irritation settles.")
    if profile.sleep_hours < 6:
        score += 8
        causes.append("Low sleep may be worsening stress-related shedding.")
        actions.append("Aim for a consistent sleep window with 7 or more hours most nights.")
    if profile.diet_quality in {"Poor", "Average"}:
        score += 8 if profile.diet_quality == "Poor" else 4
        causes.append("Nutrition gaps may affect protein, iron, zinc, vitamin D, or B12 status.")
        actions.append("Prioritize protein-rich meals and consider basic blood tests with a clinician.")
    if profile.family_history == "Yes":
        score += 10
        causes.append("Family history raises the chance of pattern-related thinning.")
    if profile.chemical_treatments in {"Often", "Sometimes"}:
        score += 7
        causes.append("Heat, color, or chemical styling may be causing breakage.")
        actions.append("Pause harsh treatments and reduce high-heat styling for 8 to 12 weeks.")
    if profile.recent_illness == "Yes":
        score += 10
        causes.append("Recent illness, fever, stress, or weight change may trigger temporary shedding.")
        questions.append("Did shedding start 6 to 12 weeks after illness, fever, crash diet, or major stress?")
    if profile.medications.strip():
        score += 5
        questions.append("Did hair fall begin after starting or changing any medicine or supplement?")

    if not causes:
        causes.append("Inputs suggest mild lifestyle or scalp contributors rather than an obvious high-risk pattern.")
    if not actions:
        actions.append("Keep the scalp routine simple and take monthly progress photos.")
    actions.append("Seek medical review if shedding is sudden, patchy, painful, or rapidly worsening.")
    questions.append("How many hairs do you see on wash days compared with normal days?")

    score = max(0, min(100, score))
    level = "Low" if score < 35 else "Medium" if score < 68 else "High"
    confidence = 58 if not has_photo else 64
    summary = (
        f"Likely {level.lower()} concern ({score}/100). "
        f"Main possible cause: {causes[0]}"
    )
    return AnalysisResult(
        risk_score=score,
        risk_level=level,
        confidence=confidence,
        likely_causes=dedupe_list(causes)[:3],
        needed_actions=dedupe_list(actions)[:4],
        follow_up_questions=dedupe_list(questions)[:3],
        photo_observations=observations[:3],
        summary=summary,
        disclaimer="Local fallback was used because Gemini is not configured or unavailable. This is informational screening, not medical diagnosis.",
        analysis_engine="LangGraph + Local Fallback",
    )


def normalize_result(data: dict[str, Any], engine_name: str) -> AnalysisResult:
    score = clamp_int(data.get("risk_score", 50), 0, 100)
    confidence = clamp_int(data.get("confidence", 70), 0, 100)
    level = str(data.get("risk_level", "Medium")).title()
    if level not in {"Low", "Medium", "High"}:
        level = "Medium"
    return AnalysisResult(
        risk_score=score,
        risk_level=level,
        confidence=confidence,
        likely_causes=ensure_list(data.get("likely_causes"), ["Lifestyle or scalp factors may be contributing."])[:3],
        needed_actions=ensure_list(data.get("needed_actions"), ["Book a clinician review if symptoms worsen."])[:4],
        follow_up_questions=ensure_list(data.get("follow_up_questions"), [])[:3],
        photo_observations=ensure_list(data.get("photo_observations"), [])[:3],
        summary=str(data.get("summary", "")).strip() or "Screening complete.",
        disclaimer="AI output is informational support only and is not a medical diagnosis.",
        analysis_engine=engine_name,
    )


def parse_json_block(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_text[start : end + 1])
        raise


def clamp_int(value: Any, low: int, high: int) -> int:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        numeric = low
    return max(low, min(high, numeric))


def ensure_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if cleaned:
            return cleaned
    return fallback


def dedupe_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def get_config_value(key: str, default: str = "") -> str:
    value = os.getenv(key, "").strip()
    if value:
        return value
    try:
        import streamlit as st

        secret_value = st.secrets.get(key, default)
        return str(secret_value).strip()
    except Exception:
        return default
