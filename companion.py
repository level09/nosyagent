import json
import logging
import random
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from reminder_scheduler import schedule_reminder_task
from storage import Storage, UserSettings

logger = logging.getLogger(__name__)

DEFAULT_SPARKS = {
    "life": [
        "Write a two-sentence win log at night—patterns emerge fast.",
        "Call the person you've been meaning to thank for months.",
        "List the three conversations that energized you last week.",
    ],
    "work": [
        "Block 90 minutes for deep work before noon—guard it.",
        "Draft a one-page brief for your toughest project—clarity beats speed.",
        "Pick the decision that is stuck and list two facts you still need.",
    ],
    "health": [
        "Walk while taking your next call to freshen the loop.",
        "Drink water before coffee tomorrow and note if anything shifts.",
        "Stretch your back for 90 seconds between sessions today.",
    ],
    "finance": [
        "Review the last three discretionary purchases for glow vs. meh.",
        "Check fees on one recurring subscription and renegotiate or cut.",
        "Write a sentence on how you want next month's money to feel.",
    ],
}

DEFAULT_BLINDSPOTS = {
    "life": [
        "What would you postpone if focus dropped tomorrow?",
        "Which relationship do you want to steward more actively?",
    ],
    "work": [
        "Where are decisions waiting for you to choose?",
        "What does success look like for stakeholders this month?",
    ],
    "health": [
        "Did sleep or movement drive yesterday's energy?",
        "What recovery practice is missing this week?",
    ],
    "finance": [
        "How would a surprise expense hit your cash flow?",
        "What's the plan if income dips 15% for a quarter?",
    ],
}


class CompanionService:
    """Companion settings and scheduled nudges (no per-reply injection)."""

    def __init__(self, storage: Storage, cards_path: Path, enabled: bool = True):
        self.storage = storage
        self.cards_path = cards_path
        self.enabled = enabled
        self.sparks: Dict[str, List[str]] = {}
        self.blindspots: Dict[str, List[str]] = {}
        if enabled:
            self._load_cards()
        else:
            logger.info("Companion mode disabled via configuration")

    # === Public API ===

    async def set_companion_level(self, chat_id: str, level: str) -> UserSettings:
        level = level.lower()
        if level not in {"off", "light", "standard"}:
            raise ValueError("companion_level must be off|light|standard")
        settings = await self.storage.get_user_settings(chat_id)
        settings.companion_level = level
        await self.storage.upsert_user_settings(settings)
        return settings

    async def set_quiet_hours(self, chat_id: str, start: str, end: str) -> UserSettings:
        self._validate_hhmm(start)
        self._validate_hhmm(end)
        settings = await self.storage.get_user_settings(chat_id)
        settings.quiet_hours_start = start
        settings.quiet_hours_end = end
        await self.storage.upsert_user_settings(settings)
        return settings

    async def set_nudge_frequency(self, chat_id: str, frequency: str) -> UserSettings:
        frequency = frequency.lower()
        if frequency not in {"off", "weekly", "standard"}:
            raise ValueError("nudge frequency must be off|weekly|standard")
        settings = await self.storage.get_user_settings(chat_id)
        settings.nudge_frequency = frequency
        await self.storage.upsert_user_settings(settings)
        return settings

    async def schedule_next_nudge(self, chat_id: str) -> Optional[datetime]:
        settings = await self.storage.get_user_settings(chat_id)
        if (
            not self.enabled
            or settings.companion_level == "off"
            or settings.nudge_frequency == "off"
        ):
            return None
        now = datetime.utcnow()
        if settings.last_nudge_at and settings.last_nudge_at > now:
            return settings.last_nudge_at

        target_time = self._compute_next_nudge_time(settings, now)
        spark = self._pick_spark()
        if not spark:
            return None

        message = f"Spark: {spark} Reply stop to mute."
        scheduled = await schedule_reminder_task(chat_id, message, target_time)
        if not scheduled:
            return None

        settings.last_nudge_at = target_time
        await self.storage.upsert_user_settings(settings)
        return target_time

    # === Helpers ===

    def _load_cards(self):
        data = self._read_cards_file()
        self.sparks = data.get("sparks", DEFAULT_SPARKS)
        if not self.sparks:
            self.sparks = DEFAULT_SPARKS
        self.blindspots = data.get("blindspots", DEFAULT_BLINDSPOTS)
        if not self.blindspots:
            self.blindspots = DEFAULT_BLINDSPOTS
        logger.info("Companion cards loaded: %s spark topics", len(self.sparks))

    def _read_cards_file(self) -> Dict:
        if not self.cards_path.exists():
            logger.warning(
                "Companion cards file missing at %s, using defaults", self.cards_path
            )
            return {}
        try:
            with self.cards_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception as exc:
            logger.error(f"Failed to load companion cards: {exc}")
            return {}

    def _validate_hhmm(self, value: str):
        try:
            datetime.strptime(value, "%H:%M")
        except ValueError as exc:
            raise ValueError("time must be HH:MM in 24h format") from exc

    def _compute_next_nudge_time(
        self, settings: UserSettings, now: datetime
    ) -> datetime:
        delta = timedelta(days=7)
        if settings.nudge_frequency == "standard":
            delta = timedelta(days=3)
        candidate = (now + delta).replace(second=0, microsecond=0)

        quiet_start, quiet_end = self._quiet_bounds(settings)
        candidate = candidate.replace(hour=quiet_end.hour, minute=quiet_end.minute)
        candidate += timedelta(minutes=60)

        while not self._is_after_now(candidate, now) or self._is_quiet_time(
            candidate.time(), quiet_start, quiet_end
        ):
            candidate += timedelta(hours=1)
            if candidate - now > timedelta(days=2):
                break
        return candidate

    def _quiet_bounds(self, settings: UserSettings) -> Tuple[time, time]:
        start = datetime.strptime(settings.quiet_hours_start, "%H:%M").time()
        end = datetime.strptime(settings.quiet_hours_end, "%H:%M").time()
        return start, end

    def _is_quiet_time(self, current: time, start: time, end: time) -> bool:
        if start < end:
            return start <= current < end
        return current >= start or current < end

    def _is_after_now(self, candidate: datetime, now: datetime) -> bool:
        return candidate > now + timedelta(minutes=1)

    def _pick_spark(self) -> Optional[str]:
        if not self.sparks:
            return None
        topic = random.choice(list(self.sparks.keys()))
        sparks = self.sparks.get(topic, [])
        if not sparks:
            return None
        return random.choice(sparks)
