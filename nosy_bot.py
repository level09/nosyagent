import json
import logging
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from http import HTTPStatus

from fastapi import FastAPI, Request, Response
from chatgpt_md_converter import telegram_format
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent import AIAgent as NosyAgent
from config import get_config
from companion import CompanionService
from router import Router
from storage import Storage

# Configure logging: app at INFO, silence noisy libraries
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("aiosqlite").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

RECENT_PHOTOS = {}
RECENT_PHOTO_TTL_SECONDS = 600
IMAGE_FOLLOWUP_TERMS = {
    "ocr",
    "image",
    "photo",
    "picture",
    "screenshot",
    "attached",
    "transcribe",
    "read the image",
    "read this image",
    "read the photo",
}


# Using chatgpt-md-converter library for proper Telegram HTML formatting


def convert_markdown_to_html(text):
    """Convert markdown text to Telegram-compatible HTML using specialized library"""
    try:
        return telegram_format(text)
    except Exception as e:
        logger.warning(f"Markdown conversion failed: {e}, using plain text")
        return text


def split_telegram_chunks(text: str, limit: int) -> list[str]:
    """Split long Telegram messages without hard truncating."""
    if len(text) <= limit:
        return [text]

    chunks = []
    current = ""
    for block in text.split("\n\n"):
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(block) > limit:
            chunks.append(block[:limit])
            block = block[limit:]
        current = block
    if current:
        chunks.append(current)
    return chunks


def validate_input(text: str, chat_id: int) -> bool:
    """Validate user input for security and limits"""
    if not text or not text.strip():
        return False

    # Length limits
    if len(text) > config.MAX_MESSAGE_LENGTH:
        logger.warning(f"Message too long from chat {chat_id}: {len(text)} chars")
        return False

    # Basic security: no suspicious patterns
    suspicious_patterns = ["<script", "<?php", "javascript:", "data:"]
    text_lower = text.lower()
    if any(pattern in text_lower for pattern in suspicious_patterns):
        logger.warning(f"Suspicious content detected from chat {chat_id}")
        return False

    return True


def clean_expired_updates():
    """Remove expired update IDs from cache to prevent memory bloat"""
    current_time = time.time()
    expired_keys = [
        update_id
        for update_id, timestamp in processed_updates.items()
        if current_time - timestamp > CACHE_EXPIRY_SECONDS
    ]
    for key in expired_keys:
        del processed_updates[key]

    if expired_keys:
        logger.debug(f"Cleaned {len(expired_keys)} expired update IDs from cache")


def is_duplicate_update(update_id: int) -> bool:
    """Check if this update_id has already been processed"""
    clean_expired_updates()  # Clean expired entries
    return update_id in processed_updates


def mark_update_processed(update_id: int):
    """Mark an update_id as processed"""
    processed_updates[update_id] = time.time()


async def send_or_edit_message(update, thinking_message, text):
    """Send or edit message with chunking and plain-text fallback."""
    chunks = split_telegram_chunks(text, config.TELEGRAM_MAX_LENGTH - 96)

    for index, chunk in enumerate(chunks):
        html_text = convert_markdown_to_html(chunk)
        try:
            if thinking_message is not None and index == 0:
                await thinking_message.edit_text(html_text, parse_mode=ParseMode.HTML)
            else:
                await update.message.reply_text(html_text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.warning(f"HTML message failed: {e}, sending plain text chunk")
            if thinking_message is not None and index == 0:
                await thinking_message.edit_text(chunk)
            else:
                await update.message.reply_text(chunk)


def is_image_followup(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in IMAGE_FOLLOWUP_TERMS)


def remember_recent_photo(chat_id: int, image_bytes: bytes, context: str):
    RECENT_PHOTOS[chat_id] = {
        "bytes": bytes(image_bytes),
        "context": context,
        "stored_at": time.time(),
    }


def get_recent_photo(chat_id: int):
    record = RECENT_PHOTOS.get(chat_id)
    if not record:
        return None
    if time.time() - record["stored_at"] > RECENT_PHOTO_TTL_SECONDS:
        RECENT_PHOTOS.pop(chat_id, None)
        return None
    return record


# Initialize config and validate
config = get_config()
config.validate()

# Initialize storage
storage = Storage(config.DB_PATH)

# Companion service + agent
companion_service = CompanionService(
    storage,
    config.COMPANION_CARDS_PATH,
    enabled=config.COMPANION_MODE_ENABLED,
)

agent = NosyAgent(config, storage, companion_service)

# Router for smart message classification
router = Router(config)

# Whitelist of allowed chat IDs
ALLOWED_CHAT_IDS = config.ALLOWED_CHAT_IDS

# Startup info
logger.info(
    f"NosyAgent starting: model={config.SONNET_MODEL} haiku={config.HAIKU_MODEL} "
    f"companion={'on' if config.COMPANION_MODE_ENABLED else 'off'} "
    f"notion={'on' if config.NOTION_TOKEN else 'off'} "
    f"context_budget={config.CONTEXT_TOKEN_BUDGET} "
    f"users={len(ALLOWED_CHAT_IDS)}"
)

# Webhook deduplication cache - stores processed update_ids with timestamps
# Format: {update_id: timestamp}
processed_updates = {}
CACHE_EXPIRY_SECONDS = 3600  # Keep processed IDs for 1 hour

# Create telegram application
ptb = (
    Application.builder()
    .updater(None)  # We handle updates manually via webhook
    .token(config.TELEGRAM_BOT_TOKEN)
    .read_timeout(7)
    .get_updates_read_timeout(42)
    .build()
)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command"""
    chat_id = update.effective_chat.id

    if chat_id not in ALLOWED_CHAT_IDS:
        await update.message.reply_text(
            "🚫 Access Restricted\n\n"
            "This AI agent is currently available only to specific members. "
            "If you believe you should have access, please contact the administrator."
        )
        return

    await update.message.reply_text(
        "👋 Hello! I'm your Nosy Agent.\n\n"
        "I'm here to help with life optimization, scheduling, and personal assistance. "
        "Just send me a message and I'll do my best to help!"
    )


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id_raw = update.effective_chat.id
    if chat_id_raw not in ALLOWED_CHAT_IDS:
        await update.message.reply_text("🚫 Access Restricted")
        return

    chat_id = str(chat_id_raw)
    args = context.args if context.args else []

    if not args:
        settings = await companion_service.storage.get_user_settings(chat_id)
        await update.message.reply_text(
            f"Companion mode is {settings.companion_level}. Use /mode off|light|standard to change."
        )
        return

    level = args[0].lower()
    try:
        settings = await companion_service.set_companion_level(chat_id, level)
    except ValueError:
        await update.message.reply_text("Usage: /mode off|light|standard")
        return

    response = f"Companion mode set to {settings.companion_level}."
    if settings.companion_level != "off":
        scheduled = await companion_service.schedule_next_nudge(chat_id)
        if scheduled:
            response += f" Next spark queued for {scheduled.strftime('%a %H:%M UTC')}"

    await update.message.reply_text(response)


async def quiet_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id_raw = update.effective_chat.id
    if chat_id_raw not in ALLOWED_CHAT_IDS:
        await update.message.reply_text("🚫 Access Restricted")
        return

    chat_id = str(chat_id_raw)
    args = context.args if context.args else []

    if len(args) != 2:
        await update.message.reply_text("Usage: /quiet HH:MM HH:MM (24h format)")
        return

    try:
        settings = await companion_service.set_quiet_hours(chat_id, args[0], args[1])
    except ValueError:
        await update.message.reply_text("Usage: /quiet HH:MM HH:MM")
        return

    await update.message.reply_text(
        f"Quiet hours set to {settings.quiet_hours_start}–{settings.quiet_hours_end}."
    )


async def nudge_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id_raw = update.effective_chat.id
    if chat_id_raw not in ALLOWED_CHAT_IDS:
        await update.message.reply_text("🚫 Access Restricted")
        return

    chat_id = str(chat_id_raw)
    args = context.args if context.args else []

    if not args:
        settings = await companion_service.storage.get_user_settings(chat_id)
        await update.message.reply_text(
            f"Nudges are {settings.nudge_frequency}. Use /nudge off|weekly|standard."
        )
        return

    choice = args[0].lower()
    if choice == "on":
        choice = "weekly"

    try:
        settings = await companion_service.set_nudge_frequency(chat_id, choice)
    except ValueError:
        await update.message.reply_text("Usage: /nudge off|weekly|standard")
        return

    response = f"Nudges set to {settings.nudge_frequency}."
    if settings.nudge_frequency != "off":
        scheduled = await companion_service.schedule_next_nudge(chat_id)
        if scheduled:
            response += f" Next spark queued for {scheduled.strftime('%a %H:%M UTC')}"

    await update.message.reply_text(response)


async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id_raw = update.effective_chat.id
    if chat_id_raw not in ALLOWED_CHAT_IDS:
        await update.message.reply_text("🚫 Access Restricted")
        return

    chat_id = str(chat_id_raw)
    args = context.args if context.args else []
    action = args[0].lower() if args else "status"

    if action == "status":
        status = await storage.get_memory_status(chat_id)
        lines = [f"Memory items: {status['total']} | conflicts: {status['conflicts']}"]
        if status["groups"]:
            for group in status["groups"]:
                lines.append(
                    f"- {group['layer']}/{group['confidence']}: "
                    f"{group['count']} avg={group['avg_strength']}"
                )
        else:
            lines.append("No structured memories yet.")
        await update.message.reply_text("\n".join(lines))
        return

    if action == "review":
        review = await storage.get_memory_review(chat_id)
        memories = review["memories"]
        conflicts = review["conflicts"]
        if not memories and not conflicts:
            await update.message.reply_text("No memories need review.")
            return

        lines = ["Review these memories:"]
        for memory in memories:
            lines.append(
                f"#{memory.id} [{memory.confidence}/{memory.layer}] "
                f"strength={memory.strength:.2f}: {memory.content}"
            )
        for conflict in conflicts:
            lines.append(
                f"Conflict #{conflict['id']}: {conflict['reason']}\n"
                f"A: {conflict['memory_a']}\n"
                f"B: {conflict['memory_b']}"
            )
        lines.append(
            "Actions: /memory confirm <id> | stale <id> | forget <id> | "
            "correct <id> <text>"
        )
        await update.message.reply_text("\n\n".join(lines))
        return

    if action in {"confirm", "stale", "forget"}:
        if len(args) < 2 or not args[1].isdigit():
            await update.message.reply_text(f"Usage: /memory {action} <id>")
            return
        memory_id = int(args[1])
        if action == "confirm":
            ok = await storage.confirm_memory_item(chat_id, memory_id)
            text = (
                f"Confirmed memory #{memory_id}"
                if ok
                else f"Memory #{memory_id} not found"
            )
        elif action == "stale":
            ok = await storage.stale_memory_item(chat_id, memory_id)
            text = (
                f"Staled memory #{memory_id}"
                if ok
                else f"Memory #{memory_id} not found"
            )
        else:
            ok = await storage.forget_memory_item(chat_id, memory_id)
            text = (
                f"Forgot memory #{memory_id}"
                if ok
                else f"Memory #{memory_id} not found"
            )
        await update.message.reply_text(text)
        return

    if action == "correct":
        if len(args) < 3 or not args[1].isdigit():
            await update.message.reply_text(
                "Usage: /memory correct <id> <corrected text>"
            )
            return
        memory_id = int(args[1])
        replacement_id = await storage.correct_memory_item(
            chat_id,
            memory_id,
            " ".join(args[2:]).strip(),
        )
        if replacement_id:
            await update.message.reply_text(
                f"Corrected memory #{memory_id} -> #{replacement_id}"
            )
        else:
            await update.message.reply_text(f"Memory #{memory_id} not found")
        return

    if action == "search":
        query = " ".join(args[1:]).strip()
        if not query:
            await update.message.reply_text("Usage: /memory search <query>")
            return
        memories = await storage.search_memory_items(chat_id, query, limit=5)
        if not memories:
            await update.message.reply_text("No matching memories.")
            return
        lines = []
        for memory in memories:
            lines.append(
                f"#{memory.id} [{memory.confidence}/{memory.layer}] "
                f"strength={memory.strength:.2f}: {memory.content}"
            )
        await update.message.reply_text("\n".join(lines))
        return

    if action == "conflicts":
        conflicts = await storage.get_memory_conflicts(chat_id)
        if not conflicts:
            await update.message.reply_text("No open memory conflicts.")
            return
        lines = []
        for conflict in conflicts:
            lines.append(
                f"Conflict #{conflict['id']}: {conflict['reason']}\n"
                f"A: {conflict['memory_a']}\n"
                f"B: {conflict['memory_b']}"
            )
        await update.message.reply_text("\n\n".join(lines))
        return

    if action == "sleep":
        run = "--run" in args
        result = await storage.run_memory_sleep(chat_id, dry_run=not run)
        mode = "applied" if run else "preview"
        text = f"Memory sleep {mode}: {result}"
        if not run:
            text += "\nAdd --run to apply these changes."
        await update.message.reply_text(text)
        return

    await update.message.reply_text(
        "Usage: /memory [status|review|confirm <id>|stale <id>|forget <id>|"
        "correct <id> <text>|search <query>|conflicts|sleep [--run]]"
    )


async def stream_to_draft(bot, chat_id: int, agent_stream) -> str:
    """Stream agent response to Telegram via sendMessageDraft, return final text."""
    draft_id = random.randint(1, 2**31)
    accumulated = ""
    last_sent = ""
    MIN_DELTA = 20  # min chars between draft updates to avoid rate limits

    async for chunk in agent_stream:
        accumulated += chunk
        if len(accumulated) - len(last_sent) >= MIN_DELTA:
            try:
                await bot.send_message_draft(
                    chat_id=chat_id, draft_id=draft_id, text=accumulated
                )
                last_sent = accumulated
            except Exception as e:
                logger.debug(f"Draft update skipped: {e}")

    # Send final draft if there's unsent text
    if accumulated and accumulated != last_sent:
        try:
            await bot.send_message_draft(
                chat_id=chat_id, draft_id=draft_id, text=accumulated
            )
        except Exception:
            pass

    return accumulated


def make_status_callback(bot, chat_id: int):
    """Create a callback that shows tool activity as draft messages."""
    draft_id = random.randint(1, 2**31)
    statuses = []

    async def on_status(label: str):
        statuses.append(label)
        status_text = " > ".join(statuses) + "..."
        try:
            await bot.send_message_draft(
                chat_id=chat_id, draft_id=draft_id, text=status_text
            )
        except Exception:
            pass

    return on_status


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages with smart routing: Haiku for simple, Sonnet streams via drafts."""
    try:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        user_text = update.message.text

        if not validate_input(user_text, chat_id):
            await update.message.reply_text(
                "Sorry, I can't process this message. Please check the content and try again."
            )
            return

        if chat_id not in ALLOWED_CHAT_IDS:
            logger.warning(
                f"Unauthorized access from chat_id: {chat_id}, user: {username}"
            )
            await update.message.reply_text(
                "🚫 Access Restricted\n\n"
                "This AI agent is currently available only to specific members. "
                "If you believe you should have access, please contact the administrator."
            )
            return

        logger.info(f"Message from {username} ({chat_id}): {user_text[:80]}")

        if is_image_followup(user_text):
            recent_photo = get_recent_photo(chat_id)
            if recent_photo:
                thinking_message = await update.message.reply_text(
                    "🤔 Reading image..."
                )
                response_chunks = []
                stream = agent.stream_chat_with_image(
                    user_text,
                    chat_id,
                    recent_photo["bytes"],
                    recent_photo["context"],
                )
                async for chunk in stream:
                    response_chunks.append(chunk)
                final_response = "".join(response_chunks[1:]).lstrip("\n")
                if not final_response.strip():
                    final_response = "I couldn't read useful text from that image."
                await send_or_edit_message(update, thinking_message, final_response)
                return

        # Route: simple messages go to Haiku, complex to Sonnet.
        # A conversation is "warm" if the last exchange was minutes ago —
        # short follow-ups then belong to the full agent, not the quick path.
        last = await storage.get_recent_conversations(str(chat_id), limit=1)
        warm = bool(last) and (
            datetime.utcnow() - last[-1].timestamp < timedelta(minutes=5)
        )
        classification = await router.classify(user_text, warm=warm)
        logger.debug(f"Classified as {classification}: {user_text[:50]}")

        if classification == "simple":
            # Fast path: Haiku direct reply
            recent = await storage.get_recent_conversations(str(chat_id), limit=3)
            recent_ctx = ""
            if recent:
                recent_ctx = "\n".join(
                    f"User: {m.user_message}\nAssistant: {m.agent_response}"
                    for m in recent[-2:]
                )
            final_response = await router.quick_reply(user_text, recent_ctx)
            await storage.store_conversation(str(chat_id), user_text, final_response)
            await send_or_edit_message(update, None, final_response)
        else:
            # Complex path: show tool status as drafts, then stream response
            on_status = make_status_callback(ptb.bot, chat_id)
            stream = agent.stream_response(str(chat_id), user_text, on_status=on_status)
            final_response = await stream_to_draft(ptb.bot, chat_id, stream)

            if not final_response or not final_response.strip():
                final_response = "✓ Done"

            # Finalize with a real message (draft disappears, message persists)
            await send_or_edit_message(update, None, final_response)

        logger.debug(f"Response sent for chat {chat_id}")

    except Exception as e:
        logger.error(
            f"Error processing message for chat {update.effective_chat.id}: {e}"
        )
        try:
            await update.message.reply_text(
                "Sorry, I encountered an error. Please try again."
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages"""
    try:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"

        # Get message timestamp
        message_timestamp = None
        if update.message and update.message.date:
            message_timestamp = update.message.date

        logger.info(
            f"Processing photo message for chat {chat_id}, user {username}, timestamp: {message_timestamp}"
        )

        # Check whitelist protection
        if chat_id not in ALLOWED_CHAT_IDS:
            logger.warning(
                f"🚫 SECURITY: Unauthorized photo access attempt from chat_id: {chat_id}, "
                f"user_id: {user_id}, username: {username}"
            )
            await update.message.reply_text(
                "🚫 Access Restricted\n\n"
                "This AI agent is currently available only to specific members. "
                "If you believe you should have access, please contact the administrator."
            )
            return

        # Send initial thinking message
        thinking_message = None
        try:
            thinking_message = await update.message.reply_text("🤔 Analyzing image...")
            logger.debug("Successfully sent thinking message for photo")
        except Exception as thinking_error:
            logger.error(f"Failed to send thinking message for photo: {thinking_error}")

        # Get the largest photo (highest resolution)
        photo = update.message.photo[-1]

        # Download the photo
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        logger.debug(f"Downloaded photo: {len(photo_bytes)} bytes")

        # Get caption text if any
        caption_text = update.message.caption or ""

        # Build context with timestamp if available
        context = ""
        if message_timestamp:
            utc_time = message_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
            context = f"Photo received at: {utc_time}"

        # Create message for agent (combine caption with image indicator)
        remember_recent_photo(chat_id, photo_bytes, context)
        if caption_text:
            user_message = (
                f"[IMAGE ATTACHED]\n"
                f"User request: {caption_text}\n"
                "OCR any visible text exactly before answering when relevant."
            )
        else:
            user_message = (
                "[IMAGE ATTACHED]\n"
                "Task: OCR any visible text exactly, then briefly describe the image."
            )

        # Get agent response with image
        response_chunks = []
        async for chunk in agent.stream_chat_with_image(
            user_message, chat_id, photo_bytes, context
        ):
            response_chunks.append(chunk)
            logger.debug(f"Received photo chunk: {chunk[:100]!r}")

        logger.debug(f"Total photo chunks received: {len(response_chunks)}")

        # Build the final response from chunks
        if response_chunks:
            # First chunk is always "🤔 Analyzing image...", skip it
            final_response = (
                "".join(response_chunks[1:]) if len(response_chunks) > 1 else ""
            )
            # Clean up any leading newlines from concatenation
            final_response = final_response.lstrip("\n")
            logger.debug(f"Final photo response length: {len(final_response)}")
        else:
            final_response = ""
            logger.warning("No response chunks received from agent for photo")

        if not final_response.strip():
            final_response = (
                "I'm having trouble analyzing this image. Please try again."
            )

        # Send response with simplified error handling
        await send_or_edit_message(update, thinking_message, final_response)

        logger.debug(f"Photo response processing completed for chat {chat_id}")

    except Exception as e:
        logger.error(f"Error processing photo for chat {chat_id}: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        error_message = (
            "Sorry, I encountered an error processing your image. Please try again."
        )
        try:
            await send_or_edit_message(update, thinking_message, error_message)
        except Exception as send_error:
            logger.error(f"Failed to send photo error message: {send_error}")


# Add handlers to the application
ptb.add_handler(CommandHandler("start", start_command))
ptb.add_handler(CommandHandler("mode", mode_command))
ptb.add_handler(CommandHandler("quiet", quiet_command))
ptb.add_handler(CommandHandler("nudge", nudge_command))
ptb.add_handler(CommandHandler("memory", memory_command))
ptb.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
ptb.add_handler(MessageHandler(filters.PHOTO, handle_photo))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - set webhook and start/stop bot"""
    webhook_url = config.WEBHOOK_URL
    logger.info(f"Setting webhook URL: {webhook_url}")

    await ptb.bot.setWebhook(webhook_url)
    async with ptb:
        await ptb.start()
        logger.info("Bot started successfully")
        yield
        logger.info("Shutting down bot")
        await ptb.stop()


# Create FastAPI app with lifecycle management
app = FastAPI(lifespan=lifespan)


@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming webhook updates with deduplication"""
    try:
        logger.info(f"Webhook received from {request.client.host}")
        req = await request.json()

        # Check for duplicate updates before processing
        update_id = req.get("update_id")
        if update_id is not None:
            if is_duplicate_update(update_id):
                logger.info(
                    f"⚠️ Duplicate update {update_id} ignored - already processed"
                )
                return Response(status_code=HTTPStatus.OK)

            # Mark as processed immediately to prevent race conditions
            mark_update_processed(update_id)
            logger.debug(f"Processing new update {update_id}")

        logger.debug(f"Webhook payload: {json.dumps(req, indent=2)}")
        update = Update.de_json(req, ptb.bot)
        logger.debug(f"Successfully parsed update: {update}")
        await ptb.process_update(update)
        logger.debug("Successfully processed update")
        return Response(status_code=HTTPStatus.OK)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.debug(f"Request headers: {dict(request.headers)}")
        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "NosyAgent"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
