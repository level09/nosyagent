import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from reminder_scheduler import schedule_reminder_task
from storage import CompanionMetric, Storage, UserSettings

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_SET = [
    {
        "id": "focus",
        "summary": "You're homing in on {focus}.",
        "question": "What would move {topic} forward today?",
        "stretch": "Try writing the smallest next step for {topic}.",
    },
    {
        "id": "tension",
        "summary": "I hear tension around {focus}.",
        "question": "What constraint is shaping this for you?",
        "stretch": "Explore one assumption you could test fast.",
    },
    {
        "id": "momentum",
        "summary": "Momentum shows up in how you describe {focus}.",
        "question": "Where do you already have leverage here?",
        "stretch": "Name a quick win worth locking in this week.",
    },
]

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

TOPIC_MAP = {
    "health": ["sleep", "run", "gym", "diet", "protein", "fast", "steps", "walk"],
    "finance": ["budget", "money", "cash", "invest", "spend", "savings", "debt", "tax"],
    "work": ["deploy", "client", "meeting", "deadline", "project", "team", "code", "launch"],
}

STOP_WORDS = {"stop", "mute", "quiet"}


@dataclass
class ReflectionResult:
    text: str
    template_id: str
    line_count: int


class CompanionService:
    """Small helper that decides when and how to add companion reflections."""

    def __init__(self, storage: Storage, cards_path: Path, enabled: bool = True):
        self.storage = storage
        self.cards_path = cards_path
        self.enabled = enabled
        self.templates: List[Dict[str, str]] = []
        self.sparks: Dict[str, List[str]] = {}
        self.blindspots: Dict[str, List[str]] = {}
        if enabled:
            self._load_cards()
        else:
            logger.info("Companion mode disabled via configuration")

    # === Public API ===

    async def wrap_response(
        self,
        chat_id: str,
        user_message: str,
        base_response: str,
        recent_messages,
    ) -> str:
        """Apply companion reflection when the guardrails allow it."""
        if not self.enabled:
            return base_response

        settings = await self.storage.get_user_settings(chat_id)
        decision = await self._should_reflect(chat_id, user_message, recent_messages, settings)
        if not decision:
            await self.storage.upsert_user_settings(settings)
            return base_response

        reflection = self._build_reflection(user_message, base_response, settings)
        if not reflection:
            await self.storage.upsert_user_settings(settings)
            return base_response

        composed = self._merge_response(base_response, reflection.text)
        now = datetime.utcnow()
        settings.last_reflection_at = now
        settings.last_template_id = reflection.template_id
        settings.reflections_paused_until = None
        settings.short_reply_streak = 0
        await self.storage.upsert_user_settings(settings)

        try:
            await self.storage.record_companion_metric(
                CompanionMetric(
                    chat_id=chat_id,
                    template_id=reflection.template_id,
                    shown_at=now,
                    muted=False,
                    line_count=reflection.line_count,
                )
            )
        except Exception as exc:  # pragma: no cover - metrics failure should not block reply
            logger.warning(f"Failed to record companion metric: {exc}")

        return composed

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
        if not self.enabled or settings.companion_level == "off" or settings.nudge_frequency == "off":
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
        self.templates = data.get("templates", DEFAULT_TEMPLATE_SET)
        if not self.templates:
            self.templates = DEFAULT_TEMPLATE_SET
        self.sparks = data.get("sparks", DEFAULT_SPARKS)
        if not self.sparks:
            self.sparks = DEFAULT_SPARKS
        self.blindspots = data.get("blindspots", DEFAULT_BLINDSPOTS)
        if not self.blindspots:
            self.blindspots = DEFAULT_BLINDSPOTS
        logger.info(
            "Companion cards loaded: %s templates, %s spark topics",
            len(self.templates),
            len(self.sparks),
        )

    def _read_cards_file(self) -> Dict:
        if not self.cards_path.exists():
            logger.warning("Companion cards file missing at %s, using defaults", self.cards_path)
            return {}
        try:
            with self.cards_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception as exc:
            logger.error(f"Failed to load companion cards: {exc}")
            return {}

    async def _should_reflect(self, chat_id: str, user_message: str, recent_messages, settings: UserSettings) -> bool:
        if settings.companion_level == "off":
            return False

        normalized = user_message.strip().lower()
        if not normalized:
            return False

        tokens = {token.strip(",.?!") for token in normalized.split()}
        if STOP_WORDS.intersection(tokens):
            logger.debug("Companion skip: user requested silence")
            return False

        now = datetime.utcnow()
        if settings.reflections_paused_until and settings.reflections_paused_until > now:
            logger.debug("Companion skip: reflections paused until %s", settings.reflections_paused_until)
            return False

        if recent_messages:
            last_user = recent_messages[-1].user_message.strip().lower()
            if last_user and last_user == normalized:
                logger.debug("Companion skip: repeating prompt")
                return False

        # Track engagement streaks
        short_reply = len(normalized.split()) < 5
        if short_reply:
            settings.short_reply_streak += 1
        else:
            settings.short_reply_streak = 0

        if settings.short_reply_streak >= 3:
            settings.reflections_paused_until = now + timedelta(hours=24)
            logger.debug("Companion pause: three short replies, pausing until %s", settings.reflections_paused_until)
            return False

        return True

    def _merge_response(self, base_response: str, reflection_block: str) -> str:
        base_clean = base_response.rstrip()
        if not base_clean:
            return reflection_block
        return f"{base_clean}\n{reflection_block}"

    def _build_reflection(self, user_message: str, base_response: str, settings: UserSettings) -> Optional[ReflectionResult]:
        template = self._next_template(settings.last_template_id)
        if not template:
            return None

        topic = self._infer_topic(user_message)
        focus = self._focus(user_message)

        # Build candidate elements (no prefixes for natural flow)
        elements = []

        if template.get("question"):
            formatted = self._format_line(template["question"], topic, focus)
            if formatted:
                elements.append(formatted)

        if template.get("stretch"):
            formatted = self._format_line(template["stretch"], topic, focus)
            if formatted:
                elements.append(formatted)

        if not elements:
            return None

        # Pick just ONE element randomly for natural, non-robotic flow
        text = random.choice(elements)

        return ReflectionResult(text=text, template_id=template["id"], line_count=1)

    def _format_line(self, template: str, topic: str, focus: str) -> str:
        if not template:
            return ""
        try:
            text = template.format(topic=topic, focus=focus)
        except KeyError:
            text = template
        return self._trim_line(text)

    def _next_template(self, last_template_id: Optional[str]) -> Optional[Dict[str, str]]:
        if not self.templates:
            return None
        if not last_template_id:
            return self.templates[0]
        for idx, template in enumerate(self.templates):
            if template.get("id") == last_template_id:
                return self.templates[(idx + 1) % len(self.templates)]
        return self.templates[0]

    def _infer_topic(self, user_message: str) -> str:
        text = user_message.lower()
        for topic, keywords in TOPIC_MAP.items():
            if any(keyword in text for keyword in keywords):
                return topic
        return "life"

    def _focus(self, user_message: str) -> str:
        clean = " ".join(user_message.strip().split())
        if len(clean) <= 64:
            return clean
        return clean[:61] + "..."

    def _trim_line(self, text: str) -> str:
        words = text.split()
        if len(words) <= 30:
            return text
        return " ".join(words[:30])

    def _validate_hhmm(self, value: str):
        try:
            datetime.strptime(value, "%H:%M")
        except ValueError as exc:
            raise ValueError("time must be HH:MM in 24h format") from exc

    def _compute_next_nudge_time(self, settings: UserSettings, now: datetime) -> datetime:
        delta = timedelta(days=7)
        if settings.nudge_frequency == "standard":
            delta = timedelta(days=3)
        candidate = (now + delta).replace(second=0, microsecond=0)

        quiet_start, quiet_end = self._quiet_bounds(settings)
        candidate = candidate.replace(hour=quiet_end.hour, minute=quiet_end.minute)
        candidate += timedelta(minutes=60)

        while not self._is_after_now(candidate, now) or self._is_quiet_time(candidate.time(), quiet_start, quiet_end):
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
