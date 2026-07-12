#!/usr/bin/env python3
"""
ARQ Worker: reminders + daily digest
"""

import logging
from datetime import datetime
from typing import Any, Dict

import anthropic
import httpx
from arq.connections import RedisSettings
from arq.cron import cron

from config import Config
from storage import Storage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# httpx INFO logs full request URLs, which include the bot token
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

REDIS_SETTINGS = RedisSettings(host="localhost", port=6379, database=0)
STORAGE = None
CONFIG = None


def get_storage():
    global STORAGE, CONFIG
    if STORAGE is None:
        CONFIG = Config()
        STORAGE = Storage(CONFIG.DB_PATH)
    return STORAGE, CONFIG


async def send_telegram(config, chat_id: str, text: str):
    """Send a message via Telegram bot API."""
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            logger.error(f"Telegram send failed: {resp.status_code} {resp.text}")
            return False
    return True


# === Reminder task (existing) ===


async def send_reminder(
    ctx: Dict[str, Any], reminder_id: int, chat_id: str, message: str, **kwargs
) -> str:
    storage, config = get_storage()
    logger.info(f"reminder: delivering {reminder_id} to {chat_id}")

    await storage.mark_reminder_delivered(reminder_id)

    if chat_id.startswith("cli_"):
        try:
            import subprocess

            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification "{message}" with title "NosyAgent Reminder" sound name "Glass"',
                ],
                check=True,
            )
        except Exception:
            logger.info(f"CLI reminder: {message}")
    else:
        await send_telegram(config, chat_id, f"🔔 {message}")

    return f"reminder {reminder_id} delivered"


# === Daily digest ===

DIGEST_PROMPT = """You are NosyAgent, a personal life optimization assistant. Generate a brief morning digest for your user based on their context below.

Rules:
- 3-5 lines max. Telegram format.
- Reference specific facts from their brain (weight, goals, habits).
- Note patterns from recent conversations (what they mentioned, commitments made, mood signals).
- One actionable nudge based on their actual goals, not generic advice.
- Be warm but direct. No fluff.
- If they mentioned going to sleep late, note it. If weight is stalling, note it. If they skipped something, note it.
- End with one concrete question or challenge for the day.

[Brain]
{brain}

[Entities]
{entities}

[Last 24h conversations]
{recent}

[Current time]
{now}

Write the digest now. No preamble."""


async def generate_digest(ctx: Dict[str, Any]) -> str:
    """Generate and send morning digest to all active users."""
    storage, config = get_storage()
    logger.info("digest: starting daily generation")

    for chat_id in config.ALLOWED_CHAT_IDS:
        chat_id_str = str(chat_id)
        try:
            brain = await storage.read_user_context(chat_id_str)
            if not brain:
                logger.info(f"digest: skipping {chat_id_str}, no brain content")
                continue

            recent = await storage.get_recent_conversations(chat_id_str, limit=20)
            entities = await storage.get_all_entities(chat_id_str)

            # Format recent conversations (last 24h worth).
            # Exclude prior digests: feeding them back makes the digest
            # echo its own stale claims (streak counters, old weight).
            recent_text = ""
            real = [m for m in recent if m.user_message != "[daily digest]"]
            for msg in real[-10:]:
                recent_text += (
                    f"User: {msg.user_message}\nAgent: {msg.agent_response[:200]}\n\n"
                )

            entities_text = (
                "\n".join(
                    f"- {e['subject']} {e['predicate']} {e['object']}" for e in entities
                )
                if entities
                else "No structured facts yet."
            )

            now = datetime.now().strftime("%A %B %d, %Y %H:%M")

            prompt = DIGEST_PROMPT.format(
                brain=brain,
                entities=entities_text,
                recent=recent_text or "No recent conversations.",
                now=now,
            )

            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=config.SONNET_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            digest_text = response.content[0].text

            await send_telegram(config, chat_id_str, digest_text)
            await storage.store_conversation(chat_id_str, "[daily digest]", digest_text)
            logger.info(f"digest: sent to {chat_id_str} ({len(digest_text)} chars)")

        except Exception as e:
            logger.error(f"digest: failed for {chat_id_str}: {e}")

    return "digest complete"


# === Memory sleep consolidation ===


async def memory_sleep(ctx: Dict[str, Any]) -> str:
    """Run deterministic memory lifecycle maintenance for all active users."""
    storage, config = get_storage()
    logger.info("memory_sleep: starting")

    summaries = []
    for chat_id in config.ALLOWED_CHAT_IDS:
        chat_id_str = str(chat_id)
        try:
            result = await storage.run_memory_sleep(chat_id_str, dry_run=False)
            summaries.append(f"{chat_id_str}: {result}")
            logger.info(f"memory_sleep: {chat_id_str} {result}")
        except Exception as e:
            logger.error(f"memory_sleep: failed for {chat_id_str}: {e}")

    return "memory sleep complete: " + "; ".join(summaries)


# === Worker config ===


class WorkerSettings:
    functions = [send_reminder, memory_sleep]
    cron_jobs = [
        cron(memory_sleep, hour=6, minute=30),
        cron(generate_digest, hour=7, minute=0),  # 7am UTC = 8am CET
    ]
    redis_settings = REDIS_SETTINGS


async def startup(ctx: Dict[str, Any]) -> None:
    logger.info("worker starting")
    get_storage()
    logger.info("storage initialized")


async def shutdown(ctx: Dict[str, Any]) -> None:
    logger.info("worker shutting down")


WorkerSettings.on_startup = startup
WorkerSettings.on_shutdown = shutdown

if __name__ == "__main__":
    from arq import run_worker

    run_worker(WorkerSettings)
