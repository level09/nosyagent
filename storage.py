#!/usr/bin/env python3
"""
Simple SQLite storage with brain versioning
Built from scratch for production - no backward compatibility cruft
"""

import sqlite3
import aiosqlite
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Message:
    chat_id: str
    user_message: str
    agent_response: str
    timestamp: datetime

@dataclass
class Reminder:
    id: Optional[int]
    chat_id: str
    message: str
    scheduled_time: datetime
    created_time: datetime
    delivered: bool = False


@dataclass
class UserSettings:
    chat_id: str
    companion_level: str = "light"
    nudge_frequency: str = "weekly"
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "07:00"
    last_reflection_at: Optional[datetime] = None
    short_reply_streak: int = 0
    reflections_paused_until: Optional[datetime] = None
    last_template_id: Optional[str] = None
    last_nudge_at: Optional[datetime] = None


@dataclass
class CompanionMetric:
    chat_id: str
    template_id: Optional[str]
    shown_at: datetime
    muted: bool = False
    line_count: int = 0

class Storage:
    """
    Simple SQLite storage for NosyAgent
    
    Schema:
    - conversations: chat_id FK, user/agent messages
    - brain: current brain content per user (chat_id FK)
    - brain_history: versioned brain history (chat_id FK)
    - reminders: scheduled reminders (chat_id FK)
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize clean SQLite schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            
            # Conversations - keep existing production table structure
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    user TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_time ON messages(chat_id, timestamp)")
            
            # Brain - current content per user
            conn.execute("""
                CREATE TABLE IF NOT EXISTS brain (
                    chat_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL DEFAULT '',
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Brain history - version history per user
            conn.execute("""
                CREATE TABLE IF NOT EXISTS brain_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    reason TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES brain(chat_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_brain_history_chat ON brain_history(chat_id, created_at)")
            
            # Reminders - scheduled tasks per user
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    scheduled_time DATETIME NOT NULL,
                    created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    delivered BOOLEAN DEFAULT FALSE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_schedule ON reminders(scheduled_time, delivered)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_chat ON reminders(chat_id)")

            # Companion preferences per user
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    chat_id TEXT PRIMARY KEY,
                    companion_level TEXT NOT NULL DEFAULT 'light',
                    nudge_frequency TEXT NOT NULL DEFAULT 'weekly',
                    quiet_hours_start TEXT NOT NULL DEFAULT '22:00',
                    quiet_hours_end TEXT NOT NULL DEFAULT '07:00',
                    last_reflection_at DATETIME,
                    short_reply_streak INTEGER NOT NULL DEFAULT 0,
                    reflections_paused_until DATETIME,
                    last_template_id TEXT,
                    last_nudge_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Companion metrics log for lightweight telemetry
            conn.execute("""
                CREATE TABLE IF NOT EXISTS companion_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    shown_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    template_id TEXT,
                    muted INTEGER DEFAULT 0,
                    line_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_companion_metrics_chat ON companion_metrics(chat_id, shown_at)")
    
    # === CONVERSATIONS ===
    
    async def store_conversation(self, chat_id: str, user_message: str, agent_response: str):
        """Store conversation exchange"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO messages (chat_id, user, agent) VALUES (?, ?, ?)",
                (chat_id, user_message, agent_response)
            )
            await db.commit()
    
    async def get_recent_conversations(self, chat_id: str, limit: int = 10) -> List[Message]:
        """Get recent conversations for context"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT user, agent, timestamp FROM messages WHERE chat_id = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (chat_id, limit)
            )
            rows = await cursor.fetchall()
            return [
                Message(chat_id, row[0], row[1], datetime.fromisoformat(row[2]))
                for row in reversed(rows)  # Return in chronological order
            ]

    async def get_recent_user_messages(self, chat_id: str, limit: int = 3) -> List[str]:
        """Return the latest user utterances only."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT user FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?",
                (chat_id, limit)
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    # === COMPANION SETTINGS ===

    @staticmethod
    def _from_iso(value: Optional[str]) -> Optional[datetime]:
        return datetime.fromisoformat(value) if value else None

    @staticmethod
    def _to_iso(value: Optional[datetime]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return value.replace(microsecond=0).isoformat()

    async def get_user_settings(self, chat_id: str) -> UserSettings:
        """Fetch companion preferences for a user, falling back to defaults."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                "SELECT chat_id, companion_level, nudge_frequency, quiet_hours_start, quiet_hours_end, "
                "last_reflection_at, short_reply_streak, reflections_paused_until, last_template_id, last_nudge_at "
                "FROM user_settings WHERE chat_id = ?",
                (chat_id,)
            )
            row = await cursor.fetchone()

        if not row:
            return UserSettings(chat_id=chat_id)

        return UserSettings(
            chat_id=row["chat_id"],
            companion_level=row["companion_level"],
            nudge_frequency=row["nudge_frequency"],
            quiet_hours_start=row["quiet_hours_start"],
            quiet_hours_end=row["quiet_hours_end"],
            last_reflection_at=self._from_iso(row["last_reflection_at"]),
            short_reply_streak=row["short_reply_streak"],
            reflections_paused_until=self._from_iso(row["reflections_paused_until"]),
            last_template_id=row["last_template_id"],
            last_nudge_at=self._from_iso(row["last_nudge_at"]),
        )

    async def upsert_user_settings(self, settings: UserSettings):
        """Insert or update companion settings for a user."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO user_settings (
                    chat_id, companion_level, nudge_frequency, quiet_hours_start, quiet_hours_end,
                    last_reflection_at, short_reply_streak, reflections_paused_until, last_template_id, last_nudge_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM user_settings WHERE chat_id = ?), datetime('now')), datetime('now'))
                ON CONFLICT(chat_id) DO UPDATE SET
                    companion_level=excluded.companion_level,
                    nudge_frequency=excluded.nudge_frequency,
                    quiet_hours_start=excluded.quiet_hours_start,
                    quiet_hours_end=excluded.quiet_hours_end,
                    last_reflection_at=excluded.last_reflection_at,
                    short_reply_streak=excluded.short_reply_streak,
                    reflections_paused_until=excluded.reflections_paused_until,
                    last_template_id=excluded.last_template_id,
                    last_nudge_at=excluded.last_nudge_at,
                    updated_at=datetime('now')
                """,
                (
                    settings.chat_id,
                    settings.companion_level,
                    settings.nudge_frequency,
                    settings.quiet_hours_start,
                    settings.quiet_hours_end,
                    self._to_iso(settings.last_reflection_at),
                    settings.short_reply_streak,
                    self._to_iso(settings.reflections_paused_until),
                    settings.last_template_id,
                    self._to_iso(settings.last_nudge_at),
                    settings.chat_id,
                ),
            )
            await db.commit()

    async def record_companion_metric(self, metric: CompanionMetric) -> int:
        """Persist reflection telemetry for later review."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO companion_metrics (chat_id, shown_at, template_id, muted, line_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    metric.chat_id,
                    self._to_iso(metric.shown_at) or datetime.utcnow().replace(microsecond=0).isoformat(),
                    metric.template_id,
                    1 if metric.muted else 0,
                    metric.line_count,
                ),
            )
            await db.commit()
            return cursor.lastrowid

    async def get_recent_companion_metrics(self, chat_id: str, limit: int = 20) -> List[CompanionMetric]:
        """Return recent companion reflection events for diagnostics."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                """
                SELECT chat_id, shown_at, template_id, muted, line_count
                FROM companion_metrics
                WHERE chat_id = ?
                ORDER BY shown_at DESC
                LIMIT ?
                """,
                (chat_id, limit),
            )
            rows = await cursor.fetchall()

        metrics: List[CompanionMetric] = []
        for row in rows:
            metrics.append(
                CompanionMetric(
                    chat_id=row["chat_id"],
                    template_id=row["template_id"],
                    shown_at=self._from_iso(row["shown_at"]) or datetime.utcnow(),
                    muted=bool(row["muted"]),
                    line_count=row["line_count"],
                )
            )
        return metrics
    
    # === BRAIN (with automatic versioning) ===
    
    async def read_user_context(self, chat_id: str) -> str:
        """Read current brain content for user"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT content FROM brain WHERE chat_id = ?",
                (chat_id,)
            )
            row = await cursor.fetchone()
            return row[0] if row else ""
    
    async def update_user_context(self, chat_id: str, content: str, reason: str = None):
        """
        Update brain content with automatic versioning
        
        Process:
        1. Save current content to brain_history
        2. Update brain with new content
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Get current content
            cursor = await db.execute(
                "SELECT content FROM brain WHERE chat_id = ?",
                (chat_id,)
            )
            row = await cursor.fetchone()
            current_content = row[0] if row else ""
            
            # Only update if content actually changed
            if current_content.strip() != content.strip():
                # Save current to history (if exists)
                if current_content:
                    await db.execute(
                        "INSERT INTO brain_history (chat_id, content, reason) VALUES (?, ?, ?)",
                        (chat_id, current_content, reason or "Auto-versioned before update")
                    )
                
                # Update current brain
                await db.execute(
                    "INSERT OR REPLACE INTO brain (chat_id, content, updated_at) VALUES (?, ?, datetime('now'))",
                    (chat_id, content)
                )
                
                await db.commit()
    
    async def get_brain_history(self, chat_id: str, limit: int = 10) -> List[dict]:
        """Get brain version history for user"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT content, reason, created_at FROM brain_history "
                "WHERE chat_id = ? ORDER BY created_at DESC LIMIT ?",
                (chat_id, limit)
            )
            rows = await cursor.fetchall()
            return [
                {
                    "content": row[0],
                    "reason": row[1],
                    "created_at": datetime.fromisoformat(row[2])
                }
                for row in rows
            ]
    
    # === REMINDERS ===
    
    async def store_reminder(self, chat_id: str, message: str, scheduled_time: datetime) -> int:
        """Store reminder and return ID"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO reminders (chat_id, message, scheduled_time) VALUES (?, ?, ?)",
                (chat_id, message, scheduled_time.isoformat())
            )
            await db.commit()
            return cursor.lastrowid
    
    async def get_pending_reminders(self) -> List[Reminder]:
        """Get all pending reminders"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, chat_id, message, scheduled_time, created_time, delivered "
                "FROM reminders WHERE delivered = FALSE ORDER BY scheduled_time"
            )
            rows = await cursor.fetchall()
            return [
                Reminder(
                    id=row[0],
                    chat_id=row[1],
                    message=row[2],
                    scheduled_time=datetime.fromisoformat(row[3]),
                    created_time=datetime.fromisoformat(row[4]),
                    delivered=row[5]
                )
                for row in rows
            ]
    
    async def mark_reminder_delivered(self, reminder_id: int):
        """Mark reminder as delivered"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE reminders SET delivered = TRUE WHERE id = ?",
                (reminder_id,)
            )
            await db.commit()

if __name__ == '__main__':
    """Test the storage system"""
    import asyncio
    
    async def test_storage():
        print("🧪 Testing new storage system...")
        
        # Test with temporary database
        test_db = Path("test_storage.db")
        if test_db.exists():
            test_db.unlink()
        
        storage = Storage(test_db)
        chat_id = "test_user"
        
        # Test brain updates with versioning
        print("\n1. Testing brain with auto-versioning...")
        await storage.update_user_context(chat_id, "# My Brain\n\nI like coffee.", "Initial brain")
        content1 = await storage.read_user_context(chat_id)
        print(f"   Initial: {content1[:20]}...")
        
        await storage.update_user_context(chat_id, "# My Brain\n\nI like coffee and tea.", "Added tea")
        content2 = await storage.read_user_context(chat_id)
        print(f"   Updated: {content2[:20]}...")
        
        # Check history
        history = await storage.get_brain_history(chat_id)
        print(f"   History: {len(history)} versions")
        
        # Test conversations
        print("\n2. Testing conversations...")
        await storage.store_conversation(chat_id, "Hello!", "Hi there!")
        conversations = await storage.get_recent_conversations(chat_id)
        print(f"   Conversations: {len(conversations)}")
        
        # Cleanup
        test_db.unlink()
        print("\n✅ All tests passed!")
    
    asyncio.run(test_storage())
