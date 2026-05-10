from __future__ import annotations

import html
from datetime import datetime
from io import BytesIO

from hair_analysis import AnalysisResult, LabReportResult, RECOMMENDED_TESTS, UserProfile


def build_doctor_summary_text(
    profile: UserProfile,
    result: AnalysisResult,
    lab_result: LabReportResult | None,
) -> str:
    lines = [
        "Hair Doctor AI - Doctor Visit Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Patient / Person",
        f"Name: {profile.name}",
        f"For: {profile.subject}",
        f"Age: {profile.age}",
        f"Gender: {profile.gender}",
        "",
        "Hair Concern Details",
        f"Duration: {profile.duration}",
        f"Main pattern: {profile.pattern}",
        f"Hair fall level: {profile.hair_fall_level}/10",
        f"Dandruff/flakes: {profile.dandruff}",
        f"Scalp symptoms: {profile.scalp_symptoms}",
        f"Family history: {profile.family_history}",
        f"Recent illness/stress event: {profile.recent_illness}",
        f"Heat/color/chemical styling: {profile.chemical_treatments}",
        f"Current medicines/supplements noted by user: {profile.medications or 'None noted'}",
        f"Additional notes: {profile.notes or 'None'}",
        "",
        "Screening Result",
        f"Concern level: {result.risk_level} ({result.risk_score}/100)",
        f"Summary: {result.summary}",
        "",
        "Possible Contributors To Discuss",
        *format_summary_items(result.likely_causes),
        "",
        "Next Steps To Discuss With Doctor",
        *format_summary_items(result.needed_actions),
        "",
        "Suggested Tests To Discuss",
        *format_summary_items(RECOMMENDED_TESTS),
    ]
    if lab_result:
        lines.extend(
            [
                "",
                "Uploaded Report Review",
                f"Summary: {lab_result.summary}",
                "",
                "Possible Report Findings",
                *format_summary_items(lab_result.possible_findings),
                "",
                "Doctor Discussion Points From Report",
                *format_summary_items(lab_result.doctor_discussion),
                "",
                "Missing / Unread Tests",
                *format_summary_items(lab_result.missing_tests or ["No key missing tests detected from the readable report."]),
            ]
        )
    lines.extend(
        [
            "",
            "Important Safety Note",
            "This summary is informational only. It is not a diagnosis and should not be used to start, stop, or dose medicines or supplements without a dermatologist or qualified clinician.",
        ]
    )
    return "\n".join(lines)


def format_summary_items(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- None listed"]


def build_doctor_summary_pdf(name: str, summary_text: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
        title=f"{name or 'Hair'} Doctor Summary",
    )
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#007f6d")
    styles["Heading2"].textColor = colors.HexColor("#18211f")
    story = []
    headings = {
        "Patient / Person",
        "Hair Concern Details",
        "Screening Result",
        "Possible Contributors To Discuss",
        "Next Steps To Discuss With Doctor",
        "Suggested Tests To Discuss",
        "Uploaded Report Review",
        "Possible Report Findings",
        "Doctor Discussion Points From Report",
        "Missing / Unread Tests",
        "Important Safety Note",
    }
    for line in summary_text.splitlines():
        escaped = html.escape(line)
        if not line:
            story.append(Spacer(1, 8))
        elif line == "Hair Doctor AI - Doctor Visit Summary":
            story.append(Paragraph(escaped, styles["Title"]))
            story.append(Spacer(1, 8))
        elif not line.startswith("- ") and line in headings:
            story.append(Paragraph(escaped, styles["Heading2"]))
        else:
            story.append(Paragraph(escaped, styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()


def safe_file_name(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "hair-summary"
