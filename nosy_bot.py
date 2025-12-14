import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
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
from storage import Storage

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Using chatgpt-md-converter library for proper Telegram HTML formatting


def convert_markdown_to_html(text):
    """Convert markdown text to Telegram-compatible HTML using specialized library"""
    try:
        return telegram_format(text)
    except Exception as e:
        logger.warning(f"Markdown conversion failed: {e}, using plain text")
        return text


def validate_input(text: str, chat_id: int) -> bool:
    """Validate user input for security and limits"""
    if not text or not text.strip():
        return False
    
    # Length limits
    if len(text) > config.MAX_MESSAGE_LENGTH:
        logger.warning(f"Message too long from chat {chat_id}: {len(text)} chars")
        return False
    
    # Basic security: no suspicious patterns
    suspicious_patterns = ['<script', '<?php', 'javascript:', 'data:']
    text_lower = text.lower()
    if any(pattern in text_lower for pattern in suspicious_patterns):
        logger.warning(f"Suspicious content detected from chat {chat_id}")
        return False
    
    return True


def clean_expired_updates():
    """Remove expired update IDs from cache to prevent memory bloat"""
    current_time = time.time()
    expired_keys = [
        update_id for update_id, timestamp in processed_updates.items()
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
    """Send or edit message with single fallback to plain text"""
    # Validate and truncate if needed
    if len(text) > config.TELEGRAM_MAX_LENGTH:
        text = text[:config.TELEGRAM_MAX_LENGTH-6] + "..."
    
    html_text = convert_markdown_to_html(text)
    
    try:
        if thinking_message is not None:
            await thinking_message.edit_text(html_text, parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text(html_text, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.warning(f"HTML message failed: {e}, sending as plain text")
        # Single fallback: send as plain text
        if thinking_message is not None:
            await thinking_message.edit_text(text)
        else:
            await update.message.reply_text(text)


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

# Initialize agent with optional semantic memory
semantic_memory_path = config.SEMANTIC_MEMORY_PATH if config.SEMANTIC_MEMORY_ENABLED else None
agent = NosyAgent(config, storage, companion_service, semantic_memory_path=semantic_memory_path)

# Whitelist of allowed chat IDs
ALLOWED_CHAT_IDS = config.ALLOWED_CHAT_IDS

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





async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages"""
    try:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        user_text = update.message.text
        
        # Validate input
        if not validate_input(user_text, chat_id):
            await update.message.reply_text(
                "Sorry, I can't process this message. Please check the content and try again."
            )
            return

        # Get message timestamp (already in UTC)
        message_timestamp = None
        if update.message and update.message.date:
            message_timestamp = update.message.date

        logger.info(
            f"Processing message for chat {chat_id}, user {username}, timestamp: {message_timestamp}"
        )

        # Check whitelist protection
        if chat_id not in ALLOWED_CHAT_IDS:
            logger.warning(
                f"🚫 SECURITY: Unauthorized access attempt from chat_id: {chat_id}, "
                f"user_id: {user_id}, username: {username}, message_length: {len(user_text)}"
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
            thinking_message = await update.message.reply_text("🤔 Thinking...")
            logger.debug("Successfully sent thinking message")
        except Exception as thinking_error:
            logger.error(f"Failed to send thinking message: {thinking_error}")
            # Continue without thinking message - we'll send a new message instead

        # Build context with timestamp if available
        context = ""
        if message_timestamp:
            utc_time = message_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
            context = f"Message received at: {utc_time}"

        # Get agent response - collect all chunks
        response_chunks = []
        async for chunk in agent.stream_chat(user_text, chat_id, context):
            response_chunks.append(chunk)
            logger.debug(f"Received chunk: {chunk[:100]!r}")

        logger.debug(f"Total chunks received: {len(response_chunks)}")

        # Build the final response from chunks
        if response_chunks:
            # First chunk is always "🤔 Thinking...", skip it
            final_response = (
                "".join(response_chunks[1:]) if len(response_chunks) > 1 else ""
            )
            # Clean up any leading newlines from concatenation
            final_response = final_response.lstrip("\n")
            logger.debug(f"Final response length: {len(final_response)}")
        else:
            final_response = ""
            logger.warning("No response chunks received from agent")

        if not final_response.strip():
            final_response = (
                "I'm having trouble generating a response. Please try again."
            )

        # Send response with simplified error handling
        await send_or_edit_message(update, thinking_message, final_response)

        logger.debug(f"Response processing completed for chat {chat_id}")

    except Exception as e:
        logger.error(f"Error processing message for chat {chat_id}: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        error_message = "Sorry, I encountered an error processing your request. Please try again."
        try:
            await send_or_edit_message(update, thinking_message, error_message)
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
        user_message = f"[IMAGE ATTACHED]{': ' + caption_text if caption_text else ''}"

        # Get agent response with image
        response_chunks = []
        async for chunk in agent.stream_chat_with_image(user_message, chat_id, photo_bytes, context):
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
        error_message = "Sorry, I encountered an error processing your image. Please try again."
        try:
            await send_or_edit_message(update, thinking_message, error_message)
        except Exception as send_error:
            logger.error(f"Failed to send photo error message: {send_error}")


# Add handlers to the application
ptb.add_handler(CommandHandler("start", start_command))
ptb.add_handler(CommandHandler("mode", mode_command))
ptb.add_handler(CommandHandler("quiet", quiet_command))
ptb.add_handler(CommandHandler("nudge", nudge_command))
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
        update_id = req.get('update_id')
        if update_id is not None:
            if is_duplicate_update(update_id):
                logger.info(f"⚠️ Duplicate update {update_id} ignored - already processed")
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
