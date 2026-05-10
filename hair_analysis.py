from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
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


@dataclass
class LabReportResult:
    possible_findings: list[str]
    doctor_discussion: list[str]
    missing_tests: list[str]
    summary: str
    disclaimer: str


RECOMMENDED_TESTS = [
    "CBC with hemoglobin",
    "Ferritin and iron profile",
    "Vitamin D",
    "Vitamin B12",
    "TSH / thyroid profile",
    "Zinc",
    "Fasting glucose or HbA1c if clinically relevant",
    "Hormonal tests only if advised by a doctor, especially for irregular periods, acne, or excess facial hair",
]


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
- Never claim an exact cause. Say "possible", "likely", or "may" for causes.
- Do not recommend prescription medicines, doses, or starting/stopping medicines.
- If treatment or medicines may be relevant, tell the user to discuss options with a dermatologist/qualified clinician first.
- Never identify a person from the photo.
- If the photo is blurry, too dark, too close, too far, or does not show the scalp/hair clearly, ask for another clear photo.
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


def answer_follow_up(
    profile: UserProfile,
    result: AnalysisResult,
    question: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    question = question.strip()
    if not question:
        return "Ask me anything about your result, routine, symptoms, or what to do next."

    api_key = get_config_value("GEMINI_API_KEY")
    if api_key:
        try:
            return answer_follow_up_with_gemini(profile, result, question, history or [], api_key)
        except Exception:
            pass
    return heuristic_follow_up(profile, result, question)


def analyze_lab_report(
    profile: UserProfile,
    result: AnalysisResult,
    file_name: str,
    file_bytes: bytes,
    mime_type: str,
) -> LabReportResult:
    api_key = get_config_value("GEMINI_API_KEY")
    if api_key:
        try:
            return analyze_lab_report_with_gemini(profile, result, file_name, file_bytes, mime_type, api_key)
        except Exception:
            pass
    text = extract_report_text(file_name, file_bytes, mime_type)
    return heuristic_lab_report_analysis(text)


def analyze_lab_report_with_gemini(
    profile: UserProfile,
    result: AnalysisResult,
    file_name: str,
    file_bytes: bytes,
    mime_type: str,
    api_key: str,
) -> LabReportResult:
    model = get_config_value("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = {
        "profile": asdict(profile),
        "screening_result": asdict(result),
        "requested_output": {
            "possible_findings": "possible deficiency or thyroid/metabolic patterns only",
            "doctor_discussion": "what to ask a dermatologist/qualified clinician",
            "missing_tests": "important missing values from the recommended hair-loss workup",
            "summary": "2 short sentences; never claim exact cause",
        },
        "recommended_tests": RECOMMENDED_TESTS,
    }
    parts: list[dict[str, Any]] = [
        {
            "text": (
                "Review this lab report for possible deficiency patterns relevant to hair shedding. "
                "Do not diagnose. Do not claim exact cause. Do not recommend medicines or doses. "
                "Return only JSON with keys: possible_findings, doctor_discussion, missing_tests, summary.\n"
                + json.dumps(prompt, indent=2)
            )
        },
        {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(file_bytes).decode("ascii"),
            }
        },
    ]
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
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
    parsed = parse_json_block(raw_text)
    return normalize_lab_result(parsed)


def heuristic_lab_report_analysis(text: str) -> LabReportResult:
    lowered = text.lower()
    findings: list[str] = []
    doctor_discussion: list[str] = []
    present_tests: set[str] = set()

    hemoglobin = find_lab_value(lowered, [r"\bhb\b", r"hemoglobin"])
    ferritin = find_lab_value(lowered, [r"ferritin"])
    vitamin_d = find_lab_value(lowered, [r"vitamin\s*d", r"25[\s-]?oh"])
    b12 = find_lab_value(lowered, [r"b12", r"vitamin\s*b12"])
    tsh = find_lab_value(lowered, [r"\btsh\b"])
    zinc = find_lab_value(lowered, [r"zinc"])

    if hemoglobin is not None:
        present_tests.add("CBC with hemoglobin")
        if hemoglobin < 12:
            findings.append("Possible low hemoglobin/anemia pattern, which can contribute to shedding.")
            doctor_discussion.append("Ask whether anemia workup and iron status review are needed.")
    if ferritin is not None:
        present_tests.add("Ferritin and iron profile")
        if ferritin < 30:
            findings.append("Possible low ferritin/iron stores, a common contributor to hair shedding.")
            doctor_discussion.append("Ask how to correct iron stores safely and whether the cause of low iron needs evaluation.")
        elif ferritin < 50:
            findings.append("Ferritin appears borderline for hair concerns; clinical context matters.")
    if vitamin_d is not None:
        present_tests.add("Vitamin D")
        if vitamin_d < 20:
            findings.append("Possible vitamin D deficiency pattern.")
        elif vitamin_d < 30:
            findings.append("Possible vitamin D insufficiency pattern.")
    if b12 is not None:
        present_tests.add("Vitamin B12")
        if b12 < 300:
            findings.append("Possible low or borderline vitamin B12 pattern.")
    if tsh is not None:
        present_tests.add("TSH / thyroid profile")
        if tsh < 0.4 or tsh > 4.5:
            findings.append("Possible thyroid imbalance pattern that should be reviewed clinically.")
            doctor_discussion.append("Ask whether thyroid follow-up tests are needed.")
    if zinc is not None:
        present_tests.add("Zinc")
        if zinc < 70:
            findings.append("Possible low zinc pattern.")

    missing_tests = [test for test in RECOMMENDED_TESTS[:6] if test not in present_tests]
    if not text.strip():
        findings.append("I could not read enough text from this report. Please upload a clearer PDF/image or type the values.")
    elif not findings:
        findings.append("No obvious deficiency pattern was detected from the readable values, but a clinician should interpret the full report.")

    doctor_discussion.append("Bring this report to a dermatologist or qualified clinician before taking supplements or medicines.")
    summary = (
        "Report review can suggest possible deficiency patterns, but it cannot prove the exact cause of hair loss. "
        "Use these notes to guide a doctor visit."
    )
    return LabReportResult(
        possible_findings=dedupe_list(findings)[:5],
        doctor_discussion=dedupe_list(doctor_discussion)[:4],
        missing_tests=missing_tests[:6],
        summary=summary,
        disclaimer="This report review is informational only and is not a diagnosis. Do not start supplements or medicines without clinician guidance.",
    )


def normalize_lab_result(data: dict[str, Any]) -> LabReportResult:
    return LabReportResult(
        possible_findings=[soften_cause(item) for item in ensure_list(data.get("possible_findings"), ["No clear deficiency pattern was identified from the readable report."])[:5]],
        doctor_discussion=[sanitize_action(item) for item in ensure_list(data.get("doctor_discussion"), ["Discuss the report with a dermatologist or qualified clinician."])[:4]],
        missing_tests=ensure_list(data.get("missing_tests"), [])[:6],
        summary=str(data.get("summary", "")).strip() or "Report review completed. This cannot confirm an exact cause.",
        disclaimer="This report review is informational only and is not a diagnosis. Do not start supplements or medicines without clinician guidance.",
    )


def extract_report_text(file_name: str, file_bytes: bytes, mime_type: str) -> str:
    if mime_type == "text/plain" or file_name.lower().endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    if mime_type == "application/pdf" or file_name.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(file_bytes))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    return ""


def find_lab_value(text: str, labels: list[str]) -> float | None:
    for label in labels:
        pattern = rf"(?:{label})[^\d]{{0,30}}(\d+(?:\.\d+)?)"
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def answer_follow_up_with_gemini(
    profile: UserProfile,
    result: AnalysisResult,
    question: str,
    history: list[dict[str, str]],
    api_key: str,
) -> str:
    model = get_config_value("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    conversation = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in history[-8:]
    )
    prompt = {
        "profile": asdict(profile),
        "analysis": asdict(result),
        "recent_chat": conversation,
        "user_question": question,
    }
    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are a careful hair and scalp assistant. Answer follow-up questions in 2 to 5 short "
                        "sentences. Be practical and warm. Never diagnose or claim an exact cause. Do not tell the "
                        "user to start, stop, or dose any medicine. If medicines or procedures may be relevant, tell "
                        "the user to discuss them with a dermatologist or qualified clinician first. Recommend "
                        "clinician review for sudden, patchy, painful, infected, medication-related, pregnancy/"
                        "postpartum, or worsening symptoms."
                    )
                }
            ]
        },
        "contents": [{"role": "user", "parts": [{"text": json.dumps(prompt, indent=2)}]}],
        "generationConfig": {"temperature": 0.35},
    }
    response = requests.post(endpoint, params={"key": api_key}, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    answer = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )
    return answer or heuristic_follow_up(profile, result, question)


def heuristic_follow_up(profile: UserProfile, result: AnalysisResult, question: str) -> str:
    lowered = question.lower()
    if any(word in lowered for word in ["medicine", "medication", "minoxidil", "finasteride", "tablet", "dose", "treatment", "prescription"]):
        return (
            "Please do not start or change any hair-loss medicine based only on this AI screening. "
            "The safer next step is to take this result to a dermatologist or qualified clinician, who can confirm the cause and decide whether any medicine, test, or procedure is appropriate."
        )
    if any(word in lowered for word in ["exact", "cause", "diagnosis", "diagnose"]):
        return (
            "I cannot confirm the exact cause from a photo and questionnaire alone. "
            f"The most possible contributor from this screening is: {result.likely_causes[0] if result.likely_causes else 'not clear yet'}. "
            "A clinician can confirm with scalp examination, history, and tests if needed."
        )
    if any(word in lowered for word in ["oil", "shampoo", "wash", "routine"]):
        return (
            "Keep the routine simple for 4 to 6 weeks: gentle shampoo, avoid harsh scratching, and do not add many new products at once. "
            "If dandruff, itching, pain, or redness is present, a dermatologist can check whether medicated scalp care is needed."
        )
    if any(word in lowered for word in ["diet", "food", "protein", "vitamin", "iron", "b12", "zinc"]):
        return (
            "Focus on consistent protein, iron-rich foods, fruits, vegetables, and enough calories. "
            "If shedding is heavy or long-running, ask a clinician about checking ferritin/iron, vitamin D, B12, thyroid, and zinc rather than guessing supplements."
        )
    if any(word in lowered for word in ["doctor", "dermatologist", "serious", "worry", "urgent"]):
        return (
            "A clinician review is a good idea if hair loss is sudden, patchy, painful, rapidly worsening, or linked with illness, new medicine, pregnancy, or scalp infection signs. "
            f"Your current screening level is {result.risk_level.lower()}, so use the result as a guide, not a diagnosis."
        )
    if any(word in lowered for word in ["time", "recover", "regrow", "long"]):
        return (
            "Hair changes usually need patience: shedding from stress or illness may take 8 to 12 weeks to settle, and visible regrowth often takes 3 to 6 months. "
            "Monthly photos in the same lighting are more reliable than checking every day."
        )
    first_cause = result.likely_causes[0] if result.likely_causes else "your current symptoms"
    return (
        f"Based on this screening, the main thing to explore is: {first_cause} "
        "Tell me when it started, whether it is patchy or diffuse, and whether you have itching, pain, dandruff, recent illness, or new medicines. "
        "For medicines or treatment decisions, please consult a dermatologist first."
    )


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
    state["result"] = safety_filter_result(result)
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
    observations = ["Photo included. Keep future photos in the same lighting and angle for easier comparison."] if has_photo else []

    if profile.duration in {"More than 6 months", "More than 1 year", "More than 3 years", "More than 5 years", "More than 10 years"}:
        score += 10
        causes.append("Long duration may suggest pattern thinning or an ongoing trigger.")
        questions.append("Has the hairline, crown, or part width changed over time?")
    if profile.pattern in {"Hairline / temples", "Crown thinning", "Part line widening"}:
        score += 12
        causes.append("The pattern may fit hereditary or hormone-sensitive thinning.")
        actions.append("Track monthly photos in the same lighting and discuss treatment options with a dermatologist before using any medicine.")
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
        actions.append("Prioritize protein-rich meals and consider basic blood tests with a clinician before taking supplements.")
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
    actions.append("Consult a dermatologist or qualified clinician before starting any hair-loss medicine.")
    questions.append("How many hairs do you see on wash days compared with normal days?")

    score = max(0, min(100, score))
    level = "Low" if score < 35 else "Medium" if score < 68 else "High"
    confidence = 58 if not has_photo else 64
    summary = (
        f"{level} concern ({score}/100). "
        f"Most possible contributor: {causes[0]}"
    )
    return safety_filter_result(AnalysisResult(
        risk_score=score,
        risk_level=level,
        confidence=confidence,
        likely_causes=dedupe_list(causes)[:3],
        needed_actions=dedupe_list(actions)[:4],
        follow_up_questions=dedupe_list(questions)[:3],
        photo_observations=observations[:3],
        summary=summary,
        disclaimer="This is informational screening, not medical diagnosis. Consult a qualified clinician before using any medicine.",
        analysis_engine="LangGraph + Local Fallback",
    ))


def normalize_result(data: dict[str, Any], engine_name: str) -> AnalysisResult:
    score = clamp_int(data.get("risk_score", 50), 0, 100)
    confidence = clamp_int(data.get("confidence", 70), 0, 100)
    level = str(data.get("risk_level", "Medium")).title()
    if level not in {"Low", "Medium", "High"}:
        level = "Medium"
    return safety_filter_result(AnalysisResult(
        risk_score=score,
        risk_level=level,
        confidence=confidence,
        likely_causes=ensure_list(data.get("likely_causes"), ["Lifestyle or scalp factors may be contributing."])[:3],
        needed_actions=ensure_list(data.get("needed_actions"), ["Book a clinician review if symptoms worsen."])[:4],
        follow_up_questions=ensure_list(data.get("follow_up_questions"), [])[:3],
        photo_observations=ensure_list(data.get("photo_observations"), [])[:3],
        summary=str(data.get("summary", "")).strip() or "Screening complete.",
        disclaimer="AI output is informational support only and is not a medical diagnosis. Consult a qualified clinician before using any medicine.",
        analysis_engine=engine_name,
    ))


def safety_filter_result(result: AnalysisResult) -> AnalysisResult:
    result.likely_causes = [soften_cause(cause) for cause in result.likely_causes]
    result.needed_actions = [sanitize_action(action) for action in result.needed_actions]
    clinician_action = "Consult a dermatologist or qualified clinician before starting any medicine, supplement, or medicated treatment."
    if clinician_action not in result.needed_actions:
        result.needed_actions = [clinician_action] + result.needed_actions
    result.needed_actions = dedupe_list(result.needed_actions)[:4]
    result.summary = result.summary.replace("diagnosis", "screening result").replace("Diagnosis", "Screening result")
    if "exact cause" in result.summary.lower():
        result.summary = "This screening cannot confirm an exact cause. " + result.summary
    return result


def soften_cause(cause: str) -> str:
    cleaned = cause.strip()
    lowered = cleaned.lower()
    if lowered.startswith(("definite ", "confirmed ", "diagnosed ")):
        cleaned = "Possible " + cleaned.split(" ", 1)[-1]
    if not lowered.startswith(("possible", "likely", "may", "could")):
        cleaned = f"Possible {cleaned[0].lower() + cleaned[1:]}" if cleaned else cleaned
    return cleaned


def sanitize_action(action: str) -> str:
    medicine_terms = [
        "minoxidil",
        "finasteride",
        "dutasteride",
        "spironolactone",
        "ketoconazole",
        "steroid",
        "tablet",
        "capsule",
        "prescription",
        "dose",
        "mg",
        "medicine",
        "medication",
    ]
    lowered = action.lower()
    if any(term in lowered for term in medicine_terms):
        return "Discuss medicines, supplements, or medicated shampoos with a dermatologist before using them."
    return action


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
