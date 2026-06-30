import hashlib
import hmac
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt_value = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_value, 100_000)
    return f"{salt_value.hex()}:{digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    salt_hex, digest_hex = stored_hash.split(":")
    test_hash = _hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(test_hash.split(":")[1], digest_hex)


def init_db(db_path: Path, admin_username: str, admin_password: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        admin = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (admin_username,),
        ).fetchone()
        if admin is None:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, is_admin, created_at)
                VALUES (?, ?, 1, ?)
                """,
                (
                    admin_username,
                    _hash_password(admin_password),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )


def create_user(db_path: Path, username: str, password: str) -> bool:
    try:
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, is_admin, created_at)
                VALUES (?, ?, 0, ?)
                """,
                (
                    username,
                    _hash_password(password),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate_user(db_path: Path, username: str, password: str) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, is_admin FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        return None
    if not _verify_password(password, row["password_hash"]):
        return None
    return {"id": row["id"], "username": row["username"], "is_admin": bool(row["is_admin"])}


def save_chat(
    db_path: Path,
    user_id: int,
    question: str,
    answer: str,
    sources: list[dict],
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chat_history (user_id, question, answer, sources_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                question,
                answer,
                json.dumps(sources),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_user_chat_history(db_path: Path, user_id: int, limit: int = 100) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, question, answer, sources_json, created_at
            FROM chat_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "sources": json.loads(row["sources_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_all_chat_history(db_path: Path, limit: int = 500) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT ch.id, u.username, ch.question, ch.answer, ch.sources_json, ch.created_at
            FROM chat_history ch
            JOIN users u ON u.id = ch.user_id
            ORDER BY ch.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "username": row["username"],
            "question": row["question"],
            "answer": row["answer"],
            "sources": json.loads(row["sources_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_chat_history_grouped_by_user(db_path: Path, limit: int = 500) -> dict[str, list[dict]]:
    rows = get_all_chat_history(db_path, limit=limit)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["username"]].append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "sources": row["sources"],
                "created_at": row["created_at"],
            }
        )
    return dict(grouped)
