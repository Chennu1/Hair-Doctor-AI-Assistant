from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hair_analysis import AnalysisResult, UserProfile


DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "hair_uploads"
DB_PATH = DATA_DIR / "hair_doctor.db"


def init_store() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT,
                name TEXT,
                picture TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS consultations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                image_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
            """
        )
        conn.commit()


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def upsert_user(user_id: str, email: str, name: str, picture: str = "") -> None:
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO users (user_id, email, name, picture, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                email = excluded.email,
                name = excluded.name,
                picture = excluded.picture,
                updated_at = excluded.updated_at
            """,
            (user_id, email, name, picture, now, now),
        )
        conn.commit()


def save_consultation(
    user_id: str,
    profile: UserProfile,
    result: AnalysisResult,
    image_bytes: bytes | None,
    image_mime: str | None,
) -> str:
    consultation_id = str(uuid.uuid4())
    image_path = save_image(consultation_id, image_bytes, image_mime) if image_bytes else None
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO consultations
                (id, user_id, subject, profile_json, result_json, image_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                consultation_id,
                user_id,
                profile.subject,
                json.dumps(asdict(profile)),
                json.dumps(asdict(result)),
                image_path,
                utc_now(),
            ),
        )
        conn.commit()
    return consultation_id


def save_image(consultation_id: str, image_bytes: bytes | None, image_mime: str | None) -> str | None:
    if not image_bytes:
        return None
    suffix = ".jpg"
    if image_mime == "image/png":
        suffix = ".png"
    digest = hashlib.sha256(image_bytes).hexdigest()[:12]
    path = UPLOAD_DIR / f"{consultation_id}-{digest}{suffix}"
    path.write_bytes(image_bytes)
    return str(path)


def recent_consultations(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, subject, profile_json, result_json, image_path, created_at
            FROM consultations
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "subject": row["subject"],
            "profile": json.loads(row["profile_json"]),
            "result": json.loads(row["result_json"]),
            "image_path": row["image_path"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
