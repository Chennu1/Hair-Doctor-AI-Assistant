from __future__ import annotations

import html
from io import BytesIO
from dataclasses import asdict

import streamlit as st
from PIL import Image, ImageStat, UnidentifiedImageError

from hair_analysis import (
    AnalysisResult,
    LabReportResult,
    UserProfile,
    analyze_hair_case,
    analyze_lab_report,
    answer_follow_up,
    RECOMMENDED_TESTS,
)
from hair_store import init_store, recent_consultations, save_consultation, upsert_user


st.set_page_config(
    page_title="Hair Doctor AI",
    page_icon="HD",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def main() -> None:
    init_store()
    inject_css()
    user = create_guest_user()
    render_shell(user)


def create_guest_user() -> dict[str, str]:
    user_id = "guest-local"
    upsert_user(user_id, "", "Guest", "")
    return {"id": user_id, "email": "", "name": "Guest", "mode": "guest"}


def render_shell(user: dict[str, str]) -> None:
    st.markdown(
        """
        <section class="hero">
          <div>
            <span class="eyebrow">Hair and scalp check</span>
            <h1>Hair Doctor AI</h1>
            <p>Upload or click a hair/scalp photo, answer a few details, and get a short cause-focused screening with practical next steps.</p>
          </div>
          <div class="hero-panel">
            <span>Quick check</span>
            <strong>No login</strong>
            <small>Private, simple, and focused</small>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        profile = render_profile_form()
    with right:
        image_bytes, image_mime = render_photo_panel()

    analyze = st.button("Analyse", type="primary", use_container_width=True)
    if analyze:
        errors = validate_profile(profile, image_bytes)
        if errors:
            for error in errors:
                st.error(error)
        else:
            with st.spinner("Checking your hair and scalp details..."):
                result = analyze_hair_case(profile, image_bytes, image_mime)
                save_consultation(user["id"], profile, result, image_bytes, image_mime)
            st.session_state["last_result"] = asdict(result)
            st.session_state["last_profile"] = asdict(profile)
            st.session_state["chat_messages"] = []

    if st.session_state.get("last_result"):
        active_profile = UserProfile(**st.session_state["last_profile"])
        render_result(AnalysisResult(**st.session_state["last_result"]), active_profile)
        render_tests_panel()
        render_lab_report_panel()
        render_follow_up_chat(active_profile)

    render_history(user)


def render_profile_form() -> UserProfile:
    st.markdown('<div class="section-title">Required Details</div>', unsafe_allow_html=True)
    subject = st.segmented_control("Who is this for?", ["Self", "Other"], default="Self")
    name_label = "Your name" if subject == "Self" else "Person's name"
    c1, c2, c3 = st.columns([1.2, 0.75, 0.85])
    name = c1.text_input(name_label, placeholder="Name")
    age = c2.number_input("Age", min_value=1, max_value=100, value=28, step=1)
    gender = c3.selectbox("Gender", ["Female", "Male", "Non-binary", "Prefer not to say"])

    c4, c5 = st.columns(2)
    duration = c4.selectbox(
        "How long?",
        [
            "Less than 1 month",
            "1 to 3 months",
            "3 to 6 months",
            "More than 6 months",
            "More than 1 year",
            "More than 3 years",
            "More than 5 years",
            "More than 10 years",
        ],
    )
    pattern = c5.selectbox("Main pattern", ["Overall shedding", "Hairline / temples", "Crown thinning", "Part line widening", "Patchy spots", "Sudden heavy shedding", "Breakage"])

    c6, c7, c8 = st.columns(3)
    hair_fall_level = c6.slider("Hair fall level", 0, 10, 5)
    dandruff = c7.selectbox("Dandruff/flakes", ["None", "Mild", "Moderate", "Severe"])
    scalp_symptoms = c8.selectbox("Scalp symptoms", ["None", "Itching", "Pain or tenderness", "Redness/burning", "Oily scalp", "Pus/crusting"])

    c9, c10, c11 = st.columns(3)
    stress = c9.slider("Stress", 0, 10, 5)
    sleep_hours = c10.number_input("Sleep hours", min_value=0.0, max_value=14.0, value=7.0, step=0.5)
    diet_quality = c11.selectbox("Diet quality", ["Good", "Average", "Poor"])

    c12, c13, c14 = st.columns(3)
    family_history = c12.selectbox("Family history", ["No", "Yes", "Not sure"])
    chemical_treatments = c13.selectbox("Heat/color/chemical styling", ["No", "Sometimes", "Often"])
    recent_illness = c14.selectbox("Recent illness/stress event", ["No", "Yes", "Not sure"])

    medications = st.text_input("Current medicines or supplements", placeholder="Optional, but important if hair fall started recently")
    notes = st.text_area("Anything else?", placeholder="Postpartum, thyroid, weight loss, new product, diet change, etc.", height=86)

    return UserProfile(
        subject=str(subject),
        name=name.strip(),
        age=int(age),
        gender=gender,
        hair_fall_level=int(hair_fall_level),
        duration=duration,
        pattern=pattern,
        dandruff=dandruff,
        scalp_symptoms=scalp_symptoms,
        stress=int(stress),
        sleep_hours=float(sleep_hours),
        diet_quality=diet_quality,
        family_history=family_history,
        chemical_treatments=chemical_treatments,
        recent_illness=recent_illness,
        medications=medications.strip(),
        notes=notes.strip(),
    )


def render_photo_panel() -> tuple[bytes | None, str | None]:
    st.markdown('<div class="section-title">Hair Photo</div>', unsafe_allow_html=True)
    tab_camera, tab_upload = st.tabs(["Camera", "Upload"])
    camera_file = None
    upload_file = None
    with tab_camera:
        camera_file = st.camera_input("Click a clear scalp or hair photo")
    with tab_upload:
        upload_file = st.file_uploader("Upload JPG or PNG", type=["jpg", "jpeg", "png"])

    selected = camera_file or upload_file
    if selected is None:
        st.info("Use bright light and keep the scalp area in focus.")
        return None, None
    image_bytes = selected.getvalue()
    st.image(image_bytes, caption="Selected photo", use_container_width=True)
    quality_notes = photo_quality_notes(image_bytes)
    if quality_notes:
        for note in quality_notes:
            st.warning(note)
    else:
        st.success("Photo looks clear enough for screening.")
    return image_bytes, selected.type


def validate_profile(profile: UserProfile, image_bytes: bytes | None) -> list[str]:
    errors: list[str] = []
    if not profile.name:
        errors.append("Name is required.")
    if not image_bytes:
        errors.append("A hair/scalp photo is required before analysis.")
    else:
        errors.extend(photo_quality_notes(image_bytes))
    if profile.age < 16:
        errors.append("For children or teenagers, use this only as notes for a clinician and book an in-person review.")
    return errors


def photo_quality_notes(image_bytes: bytes) -> list[str]:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("L")
    except (UnidentifiedImageError, OSError):
        return ["Please upload or take another clear JPG/PNG photo."]

    width, height = image.size
    stat = ImageStat.Stat(image)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]
    notes: list[str] = []

    if min(width, height) < 480:
        notes.append("Photo is too small or low-resolution. Please take another clear photo closer to the scalp/hair area.")
    if brightness < 45:
        notes.append("Photo looks too dark. Please retake it in brighter light.")
    if brightness > 238:
        notes.append("Photo looks overexposed. Please retake it with softer light so the scalp and hair are visible.")
    if contrast < 18:
        notes.append("Photo may be blurry or low-detail. Please retake it with the camera steady and the scalp/hair in focus.")
    return notes


def render_result(result: AnalysisResult, profile: UserProfile) -> None:
    level_class = result.risk_level.lower()
    display_name = html.escape(profile.name or "there")
    st.markdown(f'<div class="section-title">{display_name}\'s Short Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-grid">
          <div class="score-card {level_class}">
            <span>{display_name}'s concern</span>
            <strong>{html.escape(result.risk_level)}</strong>
            <small>{result.risk_score}/100 concern score &middot; {result.confidence}% confidence</small>
          </div>
          <div class="summary-card">
            <span>What this may mean</span>
            <p>{display_name}, {html.escape(result.summary[:1].lower() + result.summary[1:] if result.summary else "this screening is complete.")}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        render_list("Possible Causes", result.likely_causes)
    with c2:
        render_list("What To Do Next", result.needed_actions)
    with c3:
        render_list("Follow Up", result.follow_up_questions or ["No extra question needed right now."])

    if result.photo_observations:
        render_list("Photo Notes", result.photo_observations)
    st.caption("This is informational screening only, not a medical diagnosis. Please consult a dermatologist or qualified clinician before using any medicine.")


def render_follow_up_chat(profile: UserProfile) -> None:
    display_name = html.escape(profile.name or "this person")
    st.markdown(f'<div class="section-title">Ask Follow-Up Questions For {display_name}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="chat-intro">
          Ask about {display_name}'s routine, food, dandruff, regrowth time, warning signs, doctor visit, or what to do next.
        </div>
        """,
        unsafe_allow_html=True,
    )
    messages = st.session_state.setdefault("chat_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask a follow-up question about this analysis")
    if prompt:
        profile = UserProfile(**st.session_state["last_profile"])
        result = AnalysisResult(**st.session_state["last_result"])
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_follow_up(profile, result, prompt, messages)
            st.write(answer)
        messages.append({"role": "assistant", "content": answer})


def render_tests_panel() -> None:
    st.markdown('<div class="section-title">Step 2: Tests To Discuss With A Doctor</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chat-intro">
          Show this checklist to a dermatologist or qualified clinician. These tests can help check common deficiency, thyroid, and metabolic patterns linked with shedding.
          After the tests are done, come back to this app and upload the report in the next section.
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    midpoint = (len(RECOMMENDED_TESTS) + 1) // 2
    with cols[0]:
        render_list("Common Basics", RECOMMENDED_TESTS[:midpoint])
    with cols[1]:
        render_list("Context Based", RECOMMENDED_TESTS[midpoint:])


def render_lab_report_panel() -> None:
    st.markdown('<div class="section-title">Step 3: Upload Lab Reports Here</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="upload-callout">
          <strong>After tests are done:</strong> upload the lab report below as a PDF, JPG, PNG, or TXT file.
          The app will review possible deficiency patterns and list what to discuss with the doctor.
        </div>
        """,
        unsafe_allow_html=True,
    )
    report_file = st.file_uploader(
        "Upload completed lab report",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        key="lab_report_upload",
    )
    if report_file is None:
        st.caption("Upload reports here after testing. The app reviews possible patterns only; it does not diagnose the exact cause.")
        return

    if st.button("Review Test Report", use_container_width=True):
        profile = UserProfile(**st.session_state["last_profile"])
        result = AnalysisResult(**st.session_state["last_result"])
        with st.spinner("Reviewing report values..."):
            lab_result = analyze_lab_report(
                profile,
                result,
                report_file.name,
                report_file.getvalue(),
                report_file.type,
            )
        st.session_state["last_lab_result"] = asdict(lab_result)

    if st.session_state.get("last_lab_result"):
        render_lab_result(LabReportResult(**st.session_state["last_lab_result"]))


def render_lab_result(lab_result: LabReportResult) -> None:
    st.markdown(
        f"""
        <div class="summary-card">
          <span>Report Review</span>
          <p>{html.escape(lab_result.summary)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        render_list("Possible Findings", lab_result.possible_findings)
    with c2:
        render_list("Ask Your Doctor", lab_result.doctor_discussion)
    with c3:
        render_list("Still Missing", lab_result.missing_tests or ["No key missing tests detected from the readable report."])
    st.caption(lab_result.disclaimer)


def render_list(title: str, items: list[str]) -> None:
    safe_items = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    st.markdown(
        f"""
        <div class="mini-panel">
          <h3>{html.escape(title)}</h3>
          <ul>{safe_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history(user: dict[str, str]) -> None:
    history = recent_consultations(user["id"])
    if not history:
        return
    with st.expander("Recent saved consultations", expanded=False):
        for row in history:
            result = row["result"]
            profile = row["profile"]
            st.markdown(
                f"**{html.escape(profile.get('name', 'Unknown'))}** &middot; "
                f"{html.escape(row['created_at'])} &middot; "
                f"{html.escape(result.get('risk_level', ''))} ({result.get('risk_score', 0)}/100)"
            )
            st.caption(html.escape(result.get("summary", "")))


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #18211f;
          --soft: #5d6c68;
          --line: #dce7e4;
          --panel: #ffffff;
          --page: #f4f8f6;
          --mint: #007f6d;
          --mint-soft: #dff6f0;
          --coral: #c94f4f;
          --amber: #a56a16;
          --blue: #325b93;
        }
        .stApp {
          background:
            linear-gradient(180deg, #eaf4f1 0, var(--page) 300px),
            var(--page);
          color: var(--ink);
        }
        .main .block-container {
          max-width: 1240px;
          padding: 1.2rem 2rem 3rem;
        }
        .hero {
          min-height: 230px;
          display: grid;
          grid-template-columns: minmax(0, 1fr) 250px;
          gap: 24px;
          align-items: end;
          padding: 28px;
          border: 1px solid #203532;
          border-radius: 8px;
          color: white;
          background:
            linear-gradient(120deg, rgba(0, 127, 109, .95), rgba(24, 33, 31, .94)),
            url("https://images.unsplash.com/photo-1516975080664-ed2fc6a32937?auto=format&fit=crop&w=1600&q=80");
          background-size: cover;
          background-position: center;
          box-shadow: 0 18px 42px rgba(24, 33, 31, .12);
          margin-bottom: 18px;
        }
        .hero h1 {
          margin: 4px 0 8px;
          font-size: 3.4rem;
          line-height: 1;
          letter-spacing: 0;
        }
        .hero p {
          max-width: 760px;
          margin: 0;
          color: #d8ebe7;
          font-size: 1.05rem;
        }
        .eyebrow {
          color: #aee3d6;
          font-size: .78rem;
          text-transform: uppercase;
          font-weight: 800;
        }
        .hero-panel {
          border: 1px solid rgba(255,255,255,.2);
          background: rgba(255,255,255,.09);
          border-radius: 8px;
          padding: 16px;
        }
        .hero-panel span, .hero-panel small {
          display: block;
          color: #c7dcd7;
        }
        .hero-panel strong {
          display: block;
          font-size: 1.45rem;
          margin: 4px 0;
        }
        .section-title {
          margin: 12px 0 10px;
          font-weight: 800;
          color: var(--ink);
          font-size: 1.06rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"], .mini-panel, .summary-card, .score-card {
          border-radius: 8px;
        }
        .result-grid {
          display: grid;
          grid-template-columns: 260px minmax(0, 1fr);
          gap: 12px;
          margin-bottom: 12px;
        }
        .score-card, .summary-card, .mini-panel {
          border: 1px solid var(--line);
          background: var(--panel);
          padding: 16px;
          box-shadow: 0 10px 26px rgba(24, 33, 31, .06);
        }
        .score-card span, .summary-card span {
          color: var(--soft);
          font-size: .78rem;
          text-transform: uppercase;
          font-weight: 800;
        }
        .score-card strong {
          display: block;
          font-size: 2rem;
          margin-top: 4px;
        }
        .score-card small {
          color: var(--soft);
        }
        .score-card.low strong { color: var(--mint); }
        .score-card.medium strong { color: var(--amber); }
        .score-card.high strong { color: var(--coral); }
        .summary-card p {
          margin: 6px 0 0;
          font-size: 1.05rem;
          color: var(--ink);
        }
        .mini-panel h3 {
          margin: 0 0 8px;
          color: var(--blue);
          font-size: .95rem;
        }
        .mini-panel ul {
          margin: 0;
          padding-left: 18px;
        }
        .mini-panel li {
          margin: 6px 0;
          color: var(--ink);
        }
        .stButton > button {
          border-radius: 8px;
          min-height: 46px;
          font-weight: 800;
        }
        .chat-intro {
          border: 1px solid var(--line);
          background: #eef7f4;
          border-radius: 8px;
          color: var(--soft);
          padding: 10px 12px;
          margin-bottom: 10px;
        }
        .upload-callout {
          border: 1px solid #b8d7cf;
          background: #ffffff;
          border-left: 5px solid var(--mint);
          border-radius: 8px;
          color: var(--ink);
          padding: 12px 14px;
          margin: 4px 0 12px;
          box-shadow: 0 8px 20px rgba(24, 33, 31, .05);
        }
        .upload-callout strong {
          color: var(--mint);
        }
        section[data-testid="stSidebar"] {
          display: none;
        }
        div[data-testid="collapsedControl"] {
          display: none;
        }
        @media (max-width: 820px) {
          .main .block-container { padding: 1rem; }
          .hero, .result-grid { grid-template-columns: 1fr; }
          .hero h1 { font-size: 2.35rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
