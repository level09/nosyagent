#!/usr/bin/env python3
"""
Simple SQLite storage with brain versioning
Built from scratch for production - no backward compatibility cruft
"""

import aiosqlite
import json
import math
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


SINGLE_VALUE_MEMORY_FACTS = {"lives_in", "works_at"}


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


@dataclass
class MemoryItem:
    id: Optional[int]
    chat_id: str
    layer: str
    content: str
    tags: List[str]
    confidence: str
    strength: float
    half_life_days: float
    retrieval_count: int
    last_retrieved_at: Optional[datetime]
    supersedes_id: Optional[int]
    created_at: datetime
    updated_at: datetime


@dataclass
class MemoryConflict:
    id: Optional[int]
    chat_id: str
    memory_a_id: int
    memory_b_id: int
    reason: str
    resolved: bool = False
    created_at: Optional[datetime] = None


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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_chat_time ON messages(chat_id, timestamp)"
            )

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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_brain_history_chat ON brain_history(chat_id, created_at)"
            )

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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_schedule ON reminders(scheduled_time, delivered)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_chat ON reminders(chat_id)"
            )

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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_companion_metrics_chat ON companion_metrics(chat_id, shown_at)"
            )

            # Entity triples for structured fact lookup
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_chat ON entities(chat_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_lookup ON entities(chat_id, predicate)"
            )

            # Structured memory lifecycle. The brain remains the readable summary;
            # memory_items is the durable, queryable source of truth for v1.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    layer TEXT NOT NULL CHECK(layer IN ('working', 'episodic', 'semantic')),
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    confidence TEXT NOT NULL DEFAULT 'observed'
                        CHECK(confidence IN ('verified', 'observed', 'inferred', 'stale')),
                    strength REAL NOT NULL DEFAULT 1.0,
                    half_life_days REAL NOT NULL DEFAULT 7.0,
                    retrieval_count INTEGER NOT NULL DEFAULT 0,
                    last_retrieved_at DATETIME,
                    supersedes_id INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (supersedes_id) REFERENCES memory_items(id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_chat ON memory_items(chat_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_layer ON memory_items(chat_id, layer)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_confidence ON memory_items(chat_id, confidence)"
            )

            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    memory_a_id INTEGER NOT NULL,
                    memory_b_id INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    resolved INTEGER NOT NULL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(chat_id, memory_a_id, memory_b_id),
                    FOREIGN KEY (memory_a_id) REFERENCES memory_items(id),
                    FOREIGN KEY (memory_b_id) REFERENCES memory_items(id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_conflicts_chat ON memory_conflicts(chat_id, resolved)"
            )

    # === CONVERSATIONS ===

    async def store_conversation(
        self, chat_id: str, user_message: str, agent_response: str
    ):
        """Store conversation exchange"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO messages (chat_id, user, agent) VALUES (?, ?, ?)",
                (chat_id, user_message, agent_response),
            )
            await db.commit()

    async def get_recent_conversations(
        self, chat_id: str, limit: int = 10
    ) -> List[Message]:
        """Get recent conversations for context"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT user, agent, timestamp FROM messages WHERE chat_id = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (chat_id, limit),
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
                (chat_id, limit),
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
                (chat_id,),
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
                    self._to_iso(metric.shown_at)
                    or datetime.utcnow().replace(microsecond=0).isoformat(),
                    metric.template_id,
                    1 if metric.muted else 0,
                    metric.line_count,
                ),
            )
            await db.commit()
            return cursor.lastrowid

    async def get_recent_companion_metrics(
        self, chat_id: str, limit: int = 20
    ) -> List[CompanionMetric]:
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
                "SELECT content FROM brain WHERE chat_id = ?", (chat_id,)
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
                "SELECT content FROM brain WHERE chat_id = ?", (chat_id,)
            )
            row = await cursor.fetchone()
            current_content = row[0] if row else ""

            # Only update if content actually changed
            if current_content.strip() != content.strip():
                # Save current to history (if exists)
                if current_content:
                    await db.execute(
                        "INSERT INTO brain_history (chat_id, content, reason) VALUES (?, ?, ?)",
                        (
                            chat_id,
                            current_content,
                            reason or "Auto-versioned before update",
                        ),
                    )

                # Update current brain
                await db.execute(
                    "INSERT OR REPLACE INTO brain (chat_id, content, updated_at) VALUES (?, ?, datetime('now'))",
                    (chat_id, content),
                )

                await db.commit()

    async def get_brain_history(self, chat_id: str, limit: int = 10) -> List[dict]:
        """Get brain version history for user"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT content, reason, created_at FROM brain_history "
                "WHERE chat_id = ? ORDER BY created_at DESC LIMIT ?",
                (chat_id, limit),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "content": row[0],
                    "reason": row[1],
                    "created_at": datetime.fromisoformat(row[2]),
                }
                for row in rows
            ]

    # === STRUCTURED MEMORY ITEMS ===

    @staticmethod
    def _normalize_memory_text(content: str) -> str:
        text = content.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9\s:_-]", "", text)
        return text

    @staticmethod
    def _tokenize_memory_query(query: str) -> set[str]:
        return {
            term for term in re.findall(r"[a-z0-9_]+", query.lower()) if len(term) > 2
        }

    @staticmethod
    def _memory_from_row(row) -> MemoryItem:
        tags = []
        try:
            tags = json.loads(row["tags"] or "[]")
        except json.JSONDecodeError:
            tags = []
        return MemoryItem(
            id=row["id"],
            chat_id=row["chat_id"],
            layer=row["layer"],
            content=row["content"],
            tags=tags if isinstance(tags, list) else [],
            confidence=row["confidence"],
            strength=float(row["strength"]),
            half_life_days=float(row["half_life_days"]),
            retrieval_count=int(row["retrieval_count"]),
            last_retrieved_at=Storage._from_iso(row["last_retrieved_at"]),
            supersedes_id=row["supersedes_id"],
            created_at=Storage._from_iso(row["created_at"]) or datetime.utcnow(),
            updated_at=Storage._from_iso(row["updated_at"]) or datetime.utcnow(),
        )

    async def store_memory_item(
        self,
        chat_id: str,
        content: str,
        layer: str = "episodic",
        tags: Optional[List[str]] = None,
        confidence: str = "observed",
        strength: float = 1.0,
        half_life_days: float = 7.0,
        supersedes_id: Optional[int] = None,
    ) -> int:
        """Store one structured memory item and return its ID."""
        content = content.strip()
        if not content:
            raise ValueError("memory content cannot be empty")
        if layer not in {"working", "episodic", "semantic"}:
            raise ValueError("layer must be working|episodic|semantic")
        if confidence not in {"verified", "observed", "inferred", "stale"}:
            raise ValueError("confidence must be verified|observed|inferred|stale")

        cleaned_tags = sorted(
            {tag.strip().lower() for tag in tags or [] if tag.strip()}
        )
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO memory_items (
                    chat_id, layer, content, tags, confidence, strength,
                    half_life_days, supersedes_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    layer,
                    content,
                    json.dumps(cleaned_tags),
                    confidence,
                    max(0.0, float(strength)),
                    max(1.0, float(half_life_days)),
                    supersedes_id,
                ),
            )
            if supersedes_id:
                await db.execute(
                    """
                    UPDATE memory_items
                    SET confidence = 'stale',
                        strength = MIN(strength, 0.25),
                        updated_at = datetime('now')
                    WHERE id = ? AND chat_id = ?
                    """,
                    (supersedes_id, chat_id),
                )
            await db.commit()
            return cursor.lastrowid

    async def list_memory_items(
        self,
        chat_id: str,
        limit: int = 50,
        include_stale: bool = False,
    ) -> List[MemoryItem]:
        """Return recent/high-signal memory items."""
        where_stale = "" if include_stale else "AND confidence != 'stale'"
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                f"""
                SELECT * FROM memory_items
                WHERE chat_id = ? {where_stale}
                ORDER BY strength DESC, updated_at DESC
                LIMIT ?
                """,
                (chat_id, limit),
            )
            rows = await cursor.fetchall()
        return [self._memory_from_row(row) for row in rows]

    async def search_memory_items(
        self,
        chat_id: str,
        query: str,
        limit: int = 5,
        strengthen: bool = True,
        include_stale: bool = False,
    ) -> List[MemoryItem]:
        """Search memory items with deterministic keyword scoring."""
        terms = self._tokenize_memory_query(query)
        if not terms:
            return []

        candidates = await self.list_memory_items(
            chat_id,
            limit=200,
            include_stale=include_stale,
        )
        scored: list[tuple[float, MemoryItem]] = []
        now = datetime.utcnow()
        for item in candidates:
            text = " ".join([item.content, " ".join(item.tags)]).lower()
            matches = sum(1 for term in terms if term in text)
            if not matches:
                continue
            age_days = max((now - item.updated_at).total_seconds() / 86400, 0.0)
            recency = 1.0 / (1.0 + (age_days / max(item.half_life_days, 1.0)))
            confidence_boost = {
                "verified": 1.2,
                "observed": 1.0,
                "inferred": 0.8,
                "stale": 0.45,
            }.get(item.confidence, 1.0)
            relevance = matches / max(len(terms), 1)
            score = relevance * item.strength * recency * confidence_boost
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [item for _, item in scored[:limit]]
        if strengthen and results:
            await self.strengthen_memory_items([item.id for item in results if item.id])
            results = await self.get_memory_items_by_ids(
                [item.id for item in results if item.id]
            )
        return results

    async def get_memory_items_by_ids(self, ids: List[int]) -> List[MemoryItem]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                f"SELECT * FROM memory_items WHERE id IN ({placeholders})",
                ids,
            )
            rows = await cursor.fetchall()
        by_id = {row["id"]: self._memory_from_row(row) for row in rows}
        return [by_id[item_id] for item_id in ids if item_id in by_id]

    async def strengthen_memory_items(self, ids: List[int]):
        """Apply Hippo-style retrieval strengthening."""
        if not ids:
            return
        async with aiosqlite.connect(self.db_path) as db:
            for item_id in ids:
                await db.execute(
                    """
                    UPDATE memory_items
                    SET retrieval_count = retrieval_count + 1,
                        last_retrieved_at = datetime('now'),
                        strength = MIN(strength + 0.12, 2.0),
                        half_life_days = MIN(half_life_days + 2.0, 90.0),
                        confidence = CASE WHEN confidence = 'stale' THEN 'observed' ELSE confidence END,
                        updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    (item_id,),
                )
            await db.commit()

    async def get_memory_status(self, chat_id: str) -> dict:
        """Return lightweight memory health counters for commands."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                """
                SELECT layer, confidence, COUNT(*) AS count, AVG(strength) AS avg_strength
                FROM memory_items
                WHERE chat_id = ?
                GROUP BY layer, confidence
                ORDER BY layer, confidence
                """,
                (chat_id,),
            )
            rows = await cursor.fetchall()
            conflict_cursor = await db.execute(
                "SELECT COUNT(*) FROM memory_conflicts WHERE chat_id = ? AND resolved = 0",
                (chat_id,),
            )
            conflicts = (await conflict_cursor.fetchone())[0]

        total = sum(row["count"] for row in rows)
        return {
            "total": total,
            "conflicts": conflicts,
            "groups": [
                {
                    "layer": row["layer"],
                    "confidence": row["confidence"],
                    "count": row["count"],
                    "avg_strength": round(row["avg_strength"] or 0.0, 2),
                }
                for row in rows
            ],
        }

    async def get_memory_conflicts(self, chat_id: str, limit: int = 10) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                """
                SELECT
                    c.id,
                    c.reason,
                    c.created_at,
                    a.content AS memory_a,
                    b.content AS memory_b,
                    a.confidence AS memory_a_confidence,
                    b.confidence AS memory_b_confidence
                FROM memory_conflicts c
                JOIN memory_items a ON a.id = c.memory_a_id
                JOIN memory_items b ON b.id = c.memory_b_id
                WHERE c.chat_id = ? AND c.resolved = 0
                ORDER BY c.created_at DESC
                LIMIT ?
                """,
                (chat_id, limit),
            )
            rows = await cursor.fetchall()

        conflicts = []
        for row in rows:
            if (
                row["memory_a_confidence"] == "stale"
                or row["memory_b_confidence"] == "stale"
            ):
                continue
            reason = self._memory_conflict_reason(row["memory_a"], row["memory_b"])
            if not reason:
                continue
            conflicts.append(
                {
                    "id": row["id"],
                    "reason": reason,
                    "created_at": row["created_at"],
                    "memory_a": row["memory_a"],
                    "memory_b": row["memory_b"],
                }
            )
        return conflicts

    async def get_memory_review(self, chat_id: str, limit: int = 5) -> dict:
        """Return memories and conflicts worth explicit user review."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                """
                SELECT * FROM memory_items
                WHERE chat_id = ?
                  AND confidence IN ('observed', 'inferred')
                ORDER BY
                  CASE confidence WHEN 'inferred' THEN 0 ELSE 1 END,
                  updated_at DESC
                LIMIT ?
                """,
                (chat_id, limit),
            )
            rows = await cursor.fetchall()

        return {
            "memories": [self._memory_from_row(row) for row in rows],
            "conflicts": await self.get_memory_conflicts(chat_id, limit=limit),
        }

    async def confirm_memory_item(self, chat_id: str, memory_id: int) -> bool:
        """Mark a memory as user-verified."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                UPDATE memory_items
                SET confidence = 'verified',
                    strength = MAX(strength, 1.5),
                    half_life_days = MAX(half_life_days, 30.0),
                    updated_at = datetime('now')
                WHERE chat_id = ? AND id = ? AND confidence != 'stale'
                """,
                (chat_id, memory_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def stale_memory_item(self, chat_id: str, memory_id: int) -> bool:
        """Keep a memory for audit/history but remove it from normal recall."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                UPDATE memory_items
                SET confidence = 'stale',
                    strength = MIN(strength, 0.2),
                    updated_at = datetime('now')
                WHERE chat_id = ? AND id = ?
                """,
                (chat_id, memory_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def forget_memory_item(self, chat_id: str, memory_id: int) -> bool:
        """Delete a memory and any unresolved conflict records that reference it."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE memory_items
                SET supersedes_id = NULL
                WHERE chat_id = ? AND supersedes_id = ?
                """,
                (chat_id, memory_id),
            )
            await db.execute(
                """
                DELETE FROM memory_conflicts
                WHERE chat_id = ? AND (memory_a_id = ? OR memory_b_id = ?)
                """,
                (chat_id, memory_id, memory_id),
            )
            cursor = await db.execute(
                "DELETE FROM memory_items WHERE chat_id = ? AND id = ?",
                (chat_id, memory_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def correct_memory_item(
        self,
        chat_id: str,
        memory_id: int,
        corrected_content: str,
    ) -> Optional[int]:
        """Create a verified replacement and stale the original memory."""
        existing = await self.get_memory_items_by_ids([memory_id])
        if not existing or existing[0].chat_id != chat_id:
            return None

        old = existing[0]
        replacement_id = await self.store_memory_item(
            chat_id,
            corrected_content,
            layer=old.layer,
            tags=old.tags,
            confidence="verified",
            strength=max(old.strength, 1.5),
            half_life_days=max(old.half_life_days, 30.0),
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE memory_items
                SET confidence = 'stale',
                    strength = MIN(strength, 0.2),
                    supersedes_id = ?,
                    updated_at = datetime('now')
                WHERE chat_id = ? AND id = ?
                """,
                (replacement_id, chat_id, memory_id),
            )
            await db.execute(
                """
                UPDATE memory_conflicts
                SET resolved = 1
                WHERE chat_id = ? AND (memory_a_id = ? OR memory_b_id = ?)
                """,
                (chat_id, memory_id, memory_id),
            )
            await db.commit()
        return replacement_id

    async def run_memory_sleep(self, chat_id: str, dry_run: bool = False) -> dict:
        """Consolidate memory deterministically: decay, stale, merge, promote, conflict-scan."""
        items = await self.list_memory_items(chat_id, limit=1000, include_stale=True)
        now = datetime.utcnow()
        updates: list[tuple[float, str, int]] = []
        stale_ids: set[int] = set()
        duplicate_pairs: list[tuple[int, int]] = []
        promotions: list[str] = []
        conflicts: list[tuple[int, int, str]] = []

        for item in items:
            if not item.id:
                continue
            last_activity = item.last_retrieved_at or item.updated_at or item.created_at
            age_days = max((now - last_activity).total_seconds() / 86400, 0.0)
            decayed_strength = item.strength * math.pow(
                0.5, age_days / max(item.half_life_days, 1.0)
            )
            new_confidence = item.confidence
            if item.confidence != "stale" and (
                decayed_strength < 0.25 or age_days >= 30
            ):
                new_confidence = "stale"
                stale_ids.add(item.id)
            updates.append((max(decayed_strength, 0.0), new_confidence, item.id))

        by_normalized: dict[str, MemoryItem] = {}
        duplicate_groups: dict[str, list[MemoryItem]] = {}
        for item in items:
            if not item.id or item.confidence == "stale":
                continue
            normalized = self._normalize_memory_text(item.content)
            if not normalized:
                continue
            duplicate_groups.setdefault(normalized, []).append(item)
            existing = by_normalized.get(normalized)
            if existing and existing.id:
                winner, loser = (
                    (existing, item)
                    if existing.strength >= item.strength
                    else (item, existing)
                )
                duplicate_pairs.append((winner.id, loser.id))
                by_normalized[normalized] = winner
            else:
                by_normalized[normalized] = item

        for normalized, group in duplicate_groups.items():
            episodic = [item for item in group if item.layer == "episodic"]
            if len(episodic) >= 3:
                content = episodic[0].content
                semantic_exists = any(
                    item.layer == "semantic"
                    and self._normalize_memory_text(item.content) == normalized
                    for item in items
                )
                if not semantic_exists:
                    promotions.append(content)

        fact_index: dict[str, MemoryItem] = {}
        for item in items:
            if not item.id or item.confidence == "stale":
                continue
            fact = self._extract_memory_fact(item.content)
            if not fact:
                continue
            key, value = fact
            if key not in SINGLE_VALUE_MEMORY_FACTS:
                continue
            existing = fact_index.get(key)
            if existing and existing.id and self._extract_memory_fact(existing.content):
                existing_value = self._extract_memory_fact(existing.content)[1]
                if existing_value != value:
                    older, newer = (
                        (existing, item)
                        if existing.created_at <= item.created_at
                        else (item, existing)
                    )
                    reason = self._memory_conflict_reason(
                        older.content,
                        newer.content,
                    )
                    if reason:
                        conflicts.append(
                            (
                                older.id,
                                newer.id,
                                reason,
                            )
                        )
                    fact_index[key] = newer
            else:
                fact_index[key] = item

        result = {
            "dry_run": dry_run,
            "checked": len(items),
            "decayed": len(updates),
            "staled": len(stale_ids),
            "duplicates": len(duplicate_pairs),
            "promotions": len(promotions),
            "conflicts": len(conflicts),
        }
        if dry_run:
            return result

        async with aiosqlite.connect(self.db_path) as db:
            for strength, confidence, item_id in updates:
                await db.execute(
                    """
                    UPDATE memory_items
                    SET strength = ?, confidence = ?, updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    (strength, confidence, item_id),
                )
            for winner_id, loser_id in duplicate_pairs:
                await db.execute(
                    """
                    UPDATE memory_items
                    SET confidence = 'stale',
                        strength = MIN(strength, 0.2),
                        supersedes_id = ?,
                        updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    (winner_id, loser_id),
                )
            for content in promotions:
                await db.execute(
                    """
                    INSERT INTO memory_items (
                        chat_id, layer, content, tags, confidence, strength, half_life_days
                    ) VALUES (?, 'semantic', ?, ?, 'observed', 1.25, 21.0)
                    """,
                    (
                        chat_id,
                        f"Repeated pattern: {content}",
                        json.dumps(["consolidated"]),
                    ),
                )
            for a_id, b_id, reason in conflicts:
                first, second = sorted([a_id, b_id])
                await db.execute(
                    """
                    INSERT OR IGNORE INTO memory_conflicts (
                        chat_id, memory_a_id, memory_b_id, reason
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (chat_id, first, second, reason),
                )
            await db.commit()

        return result

    @staticmethod
    def _extract_memory_fact(content: str) -> Optional[tuple[str, str]]:
        text = content.lower().strip()
        patterns = [
            (r"\buser\s+lives\s+in\s+([^.;,\n]+)", "lives_in"),
            (r"\buser\s+works\s+(?:at|for)\s+([^.;,\n]+)", "works_at"),
            (r"\buser\s+takes\s+([^.;,\n]+)", "takes"),
            (r"\buser\s+prefers\s+([^.;,\n]+)", "prefers"),
        ]
        for pattern, key in patterns:
            match = re.search(pattern, text)
            if match:
                return key, match.group(1).strip()
        return None

    @classmethod
    def _memory_conflict_reason(cls, memory_a: str, memory_b: str) -> Optional[str]:
        fact_a = cls._extract_memory_fact(memory_a)
        fact_b = cls._extract_memory_fact(memory_b)
        if not fact_a or not fact_b:
            return None

        key_a, value_a = fact_a
        key_b, value_b = fact_b
        if key_a != key_b or key_a not in SINGLE_VALUE_MEMORY_FACTS:
            return None
        if value_a == value_b:
            return None
        return f"conflicting {key_a}: {value_a} vs {value_b}"

    # === REMINDERS ===

    async def store_reminder(
        self, chat_id: str, message: str, scheduled_time: datetime
    ) -> int:
        """Store reminder and return ID"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO reminders (chat_id, message, scheduled_time) VALUES (?, ?, ?)",
                (chat_id, message, scheduled_time.isoformat()),
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
                    delivered=row[5],
                )
                for row in rows
            ]

    # === ENTITIES ===

    async def replace_entities(self, chat_id: str, triples: list[dict]):
        """Replace all entities for a user with new set."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM entities WHERE chat_id = ?", (chat_id,))
            for t in triples:
                subj = t.get("subject", "").strip()
                pred = t.get("predicate", "").strip()
                obj = t.get("object", "").strip()
                if subj and pred and obj:
                    await db.execute(
                        "INSERT INTO entities (chat_id, subject, predicate, object) VALUES (?, ?, ?, ?)",
                        (chat_id, subj.lower(), pred.lower(), obj),
                    )
            await db.commit()

    async def search_entities(self, chat_id: str, query: str) -> list[dict]:
        """Search entities by matching query against all columns."""
        q = f"%{query.lower()}%"
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT subject, predicate, object FROM entities "
                "WHERE chat_id = ? AND (subject LIKE ? OR predicate LIKE ? OR object LIKE ?)",
                (chat_id, q, q, q),
            )
            rows = await cursor.fetchall()
            return [{"subject": r[0], "predicate": r[1], "object": r[2]} for r in rows]

    async def get_all_entities(self, chat_id: str) -> list[dict]:
        """Get all entities for a user."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT subject, predicate, object FROM entities WHERE chat_id = ?",
                (chat_id,),
            )
            rows = await cursor.fetchall()
            return [{"subject": r[0], "predicate": r[1], "object": r[2]} for r in rows]

    async def mark_reminder_delivered(self, reminder_id: int):
        """Mark reminder as delivered"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE reminders SET delivered = TRUE WHERE id = ?", (reminder_id,)
            )
            await db.commit()


if __name__ == "__main__":
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
        await storage.update_user_context(
            chat_id, "# My Brain\n\nI like coffee.", "Initial brain"
        )
        content1 = await storage.read_user_context(chat_id)
        print(f"   Initial: {content1[:20]}...")

        await storage.update_user_context(
            chat_id, "# My Brain\n\nI like coffee and tea.", "Added tea"
        )
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
