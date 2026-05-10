from __future__ import annotations

import json
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageStat, UnidentifiedImageError

from hair_analysis import (
    AnalysisResult,
    LabReportResult,
    UserProfile,
    analyze_hair_case,
    analyze_lab_report,
    answer_follow_up,
)
from hair_store import init_store, save_consultation, upsert_user
from summary_utils import build_doctor_summary_pdf, build_doctor_summary_text


WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Hair Doctor AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.on_event("startup")
def startup() -> None:
    init_store()
    upsert_user("guest-local", "", "Guest", "")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/analyze")
async def analyze(
    image: Annotated[UploadFile, File()],
    profile_json: Annotated[str, Form()],
) -> JSONResponse:
    profile = UserProfile(**json.loads(profile_json))
    image_bytes = await image.read()
    quality_notes = photo_quality_notes(image_bytes)
    if quality_notes:
        return JSONResponse({"errors": quality_notes}, status_code=400)
    result = analyze_hair_case(profile, image_bytes, image.content_type)
    save_consultation("guest-local", profile, result, image_bytes, image.content_type)
    return JSONResponse({"result": asdict(result)})


@app.post("/api/chat")
async def chat(payload: dict) -> JSONResponse:
    profile = UserProfile(**payload["profile"])
    result = AnalysisResult(**payload["result"])
    answer = answer_follow_up(
        profile,
        result,
        str(payload.get("message", "")),
        payload.get("history", []),
    )
    return JSONResponse({"answer": answer})


@app.post("/api/lab-report")
async def lab_report(
    report: Annotated[UploadFile, File()],
    profile_json: Annotated[str, Form()],
    result_json: Annotated[str, Form()],
) -> JSONResponse:
    profile = UserProfile(**json.loads(profile_json))
    result = AnalysisResult(**json.loads(result_json))
    file_bytes = await report.read()
    lab_result = analyze_lab_report(
        profile,
        result,
        report.filename or "report",
        file_bytes,
        report.content_type or "application/octet-stream",
    )
    return JSONResponse({"labResult": asdict(lab_result)})


@app.post("/api/doctor-summary.pdf")
async def doctor_summary_pdf(payload: dict) -> Response:
    profile = UserProfile(**payload["profile"])
    result = AnalysisResult(**payload["result"])
    lab_result = None
    if payload.get("labResult"):
        lab_result = LabReportResult(**payload["labResult"])
    summary_text = build_doctor_summary_text(profile, result, lab_result)
    pdf_bytes = build_doctor_summary_pdf(profile.name, summary_text)
    return Response(
        pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=doctor-summary.pdf"},
    )


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
