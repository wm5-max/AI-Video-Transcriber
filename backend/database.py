import aiosqlite
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "tasks.db"
DB_PATH.parent.mkdir(exist_ok=True)

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    url TEXT,
    title TEXT,
    video_title TEXT,
    transcript TEXT,
    translation TEXT,
    summary TEXT,
    summary_language TEXT,
    summary_style TEXT,
    detected_language TEXT,
    status TEXT DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    message TEXT,
    error TEXT,
    script_path TEXT,
    summary_path TEXT,
    translation_path TEXT,
    raw_script_file TEXT,
    translation_filename TEXT,
    short_id TEXT,
    safe_title TEXT,
    transcription_model_name TEXT,
    transcription_model_version TEXT,
    summarization_model_name TEXT,
    summarization_model_version TEXT,
    api_provider TEXT,
    api_endpoint TEXT,
    transcription_input_tokens INTEGER,
    transcription_output_tokens INTEGER,
    summarization_input_tokens INTEGER,
    summarization_output_tokens INTEGER,
    processing_time_seconds REAL,
    total_tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
"""

MIGRATE_COLS = [
    "ALTER TABLE tasks ADD COLUMN transcription_model_name TEXT",
    "ALTER TABLE tasks ADD COLUMN transcription_model_version TEXT",
    "ALTER TABLE tasks ADD COLUMN summarization_model_name TEXT",
    "ALTER TABLE tasks ADD COLUMN summarization_model_version TEXT",
    "ALTER TABLE tasks ADD COLUMN api_provider TEXT",
    "ALTER TABLE tasks ADD COLUMN api_endpoint TEXT",
    "ALTER TABLE tasks ADD COLUMN transcription_input_tokens INTEGER",
    "ALTER TABLE tasks ADD COLUMN transcription_output_tokens INTEGER",
    "ALTER TABLE tasks ADD COLUMN summarization_input_tokens INTEGER",
    "ALTER TABLE tasks ADD COLUMN summarization_output_tokens INTEGER",
    "ALTER TABLE tasks ADD COLUMN processing_time_seconds REAL",
    "ALTER TABLE tasks ADD COLUMN total_tokens_used INTEGER",
]


class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(CREATE_TABLES_SQL)
            await db.commit()
            for sql in MIGRATE_COLS:
                try:
                    await db.execute(sql)
                    await db.commit()
                except Exception:
                    pass  # column already exists

    async def _connect(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        return db

    async def _safe_execute(self, operation):
        try:
            return await operation()
        except Exception as e:
            logger.warning(f"DB op failed: {e}")
            return None

    async def create_task(self, task_id: str, url: str = "", title: str = "",
                          video_title: str = "", summary_language: Optional[str] = None,
                          summary_style: Optional[str] = None) -> None:
        async def _op():
            db = await self._connect()
            try:
                await db.execute(
                    "INSERT INTO tasks (id,url,title,video_title,summary_language,summary_style,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (task_id, url, title, video_title, summary_language, summary_style, datetime.now(), datetime.now()),
                )
                await db.commit()
            finally:
                await db.close()
        await self._safe_execute(_op)

    async def update_task(self, task_id: str, **kwargs) -> None:
        """Update any task columns by keyword argument."""
        if not kwargs:
            return
        async def _op():
            db = await self._connect()
            try:
                fields = [f"{k} = ?" for k in kwargs]
                values = list(kwargs.values())
                fields.append("updated_at = ?")
                values.append(datetime.now())
                values.append(task_id)
                await db.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?", values)
                await db.commit()
            finally:
                await db.close()
        await self._safe_execute(_op)

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        async def _op():
            db = await self._connect()
            try:
                async with db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)) as cur:
                    row = await cur.fetchone()
                    if row:
                        data = dict(row)
                        data["task_id"] = data.get("id")
                        return data
                    return None
            finally:
                await db.close()
        return await self._safe_execute(_op)

    async def get_history(self, limit: int = 50, offset: int = 0,
                          summary_language: Optional[str] = None,
                          summary_style: Optional[str] = None,
                          status: Optional[str] = None) -> List[Dict[str, Any]]:
        async def _op():
            db = await self._connect()
            try:
                query = "SELECT * FROM tasks"
                params: list = []
                conds = []
                if summary_language:
                    conds.append("summary_language = ?"); params.append(summary_language)
                if summary_style:
                    conds.append("summary_style = ?"); params.append(summary_style)
                if status:
                    conds.append("status = ?"); params.append(status)
                if conds:
                    query += " WHERE " + " AND ".join(conds)
                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                async with db.execute(query, params) as cur:
                    rows = await cur.fetchall()
                    result = []
                    for row in rows:
                        data = dict(row)
                        data["task_id"] = data.get("id")
                        result.append(data)
                    return result
            finally:
                await db.close()
        return await self._safe_execute(_op) or []

    async def delete_task(self, task_id: str) -> bool:
        async def _op():
            db = await self._connect()
            try:
                cur = await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                await db.commit()
                return cur.rowcount > 0
            finally:
                await db.close()
        result = await self._safe_execute(_op)
        return bool(result)

    async def update_transcript(self, task_id: str, transcript: str) -> bool:
        async def _op():
            db = await self._connect()
            try:
                cur = await db.execute(
                    "UPDATE tasks SET transcript = ?, updated_at = ? WHERE id = ?",
                    (transcript, datetime.now(), task_id),
                )
                await db.commit()
                return cur.rowcount > 0
            finally:
                await db.close()
        result = await self._safe_execute(_op)
        return bool(result)


db = Database()
