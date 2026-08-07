import aiosqlite
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Database file path
DB_PATH = Path(__file__).parent.parent / "data" / "tasks.db"

# Ensure data directory exists
DB_PATH.parent.mkdir(exist_ok=True)

# SQL to create tables
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
"""

class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    async def init(self):
        """Initialize the database and create tables."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(CREATE_TABLES_SQL)
            await db.commit()

    async def create_task(self, task_id: str, url: str = "", title: str = "", video_title: str = "") -> None:
        """Create a new task record."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO tasks (id, url, title, video_title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task_id, url, title, video_title, datetime.now(), datetime.now())
            )
            await db.commit()

    async def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        video_title: Optional[str] = None,
        transcript: Optional[str] = None,
        translation: Optional[str] = None,
        summary: Optional[str] = None,
        summary_language: Optional[str] = None,
        summary_style: Optional[str] = None,
        detected_language: Optional[str] = None,
        script_path: Optional[str] = None,
        summary_path: Optional[str] = None,
        translation_path: Optional[str] = None,
        raw_script_file: Optional[str] = None,
        translation_filename: Optional[str] = None,
        short_id: Optional[str] = None,
        safe_title: Optional[str] = None
    ) -> None:
        """Update a task with various fields."""
        async with aiosqlite.connect(self.db_path) as db:
            # Build dynamic update query
            fields = []
            values = []
            
            if status is not None:
                fields.append("status = ?")
                values.append(status)
            if progress is not None:
                fields.append("progress = ?")
                values.append(progress)
            if message is not None:
                fields.append("message = ?")
                values.append(message)
            if error is not None:
                fields.append("error = ?")
                values.append(error)
            if video_title is not None:
                fields.append("video_title = ?")
                values.append(video_title)
            if transcript is not None:
                fields.append("transcript = ?")
                values.append(transcript)
            if translation is not None:
                fields.append("translation = ?")
                values.append(translation)
            if summary is not None:
                fields.append("summary = ?")
                values.append(summary)
            if summary_language is not None:
                fields.append("summary_language = ?")
                values.append(summary_language)
            if summary_style is not None:
                fields.append("summary_style = ?")
                values.append(summary_style)
            if detected_language is not None:
                fields.append("detected_language = ?")
                values.append(detected_language)
            if script_path is not None:
                fields.append("script_path = ?")
                values.append(script_path)
            if summary_path is not None:
                fields.append("summary_path = ?")
                values.append(summary_path)
            if translation_path is not None:
                fields.append("translation_path = ?")
                values.append(translation_path)
            if raw_script_file is not None:
                fields.append("raw_script_file = ?")
                values.append(raw_script_file)
            if translation_filename is not None:
                fields.append("translation_filename = ?")
                values.append(translation_filename)
            if short_id is not None:
                fields.append("short_id = ?")
                values.append(short_id)
            if safe_title is not None:
                fields.append("safe_title = ?")
                values.append(safe_title)
            
            if not fields:
                return
            
            fields.append("updated_at = ?")
            values.append(datetime.now())
            values.append(task_id)
            
            query = f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?"
            await db.execute(query, values)
            await db.commit()

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a task by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return dict(row)
                return None

    async def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        summary_language: Optional[str] = None,
        summary_style: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get task history with optional filters."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            query = "SELECT * FROM tasks"
            params = []
            conditions = []

            if summary_language:
                conditions.append("summary_language = ?")
                params.append(summary_language)
            if summary_style:
                conditions.append("summary_style = ?")
                params.append(summary_style)
            if status:
                conditions.append("status = ?")
                params.append(status)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            await db.commit()
            return cursor.rowcount > 0

    async def update_transcript(self, task_id: str, transcript: str) -> bool:
        """Update just the transcript (for editing)."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                UPDATE tasks
                SET transcript = ?, updated_at = ?
                WHERE id = ?
                """,
                (transcript, datetime.now(), task_id)
            )
            await db.commit()
            return cursor.rowcount > 0

# Global database instance
db = Database()