#!/usr/bin/env python3
"""
ARQ Worker for processing scheduled reminders
Similar to Celery but simpler - runs background tasks from Redis queue
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from arq import create_pool
from arq.connections import RedisSettings
from config import Config
from storage import Storage

logger = logging.getLogger(__name__)

# Redis settings for ARQ
REDIS_SETTINGS = RedisSettings(host='localhost', port=6379, database=0)

# Global storage instance
STORAGE = None

async def send_reminder(ctx: Dict[str, Any], reminder_id: int, chat_id: str, message: str, **kwargs) -> str:
    """
    ARQ task function to send a scheduled reminder
    This gets called by the worker when a reminder is due
    """
    logger.info(f"Processing reminder {reminder_id} for chat {chat_id}: {message}")
    
    try:
        # Get storage instance 
        global STORAGE
        if STORAGE is None:
            config = Config()
            STORAGE = Storage(config.DB_PATH)
            
        storage = STORAGE
        
        # Mark reminder as delivered
        await storage.mark_reminder_delivered(reminder_id)
        
        # Handle different delivery methods based on chat_id
        if chat_id.startswith("cli_"):
            # CLI user - create desktop notification
            try:
                import subprocess
                # macOS desktop notification
                subprocess.run([
                    "osascript", "-e", 
                    f'display notification "{message}" with title "NosyAgent Reminder" sound name "Glass"'
                ], check=True)
                logger.info(f"✅ CLI reminder delivered via desktop notification: {message}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                # Fallback to system bell and log
                logger.info(f"🔔 CLI REMINDER: {message}")
                try:
                    print(f"\a🔔 REMINDER: {message}")  # Terminal bell + message
                except:
                    pass
        else:
            # Telegram user - send actual message via Telegram API
            try:
                import httpx
                config = Config()
                bot_token = config.TELEGRAM_BOT_TOKEN
                
                if bot_token:
                    telegram_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                    payload = {
                        "chat_id": chat_id,
                        "text": f"🔔 Reminder: {message}",
                        "parse_mode": "HTML"
                    }
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.post(telegram_url, json=payload, timeout=10.0)
                        if response.status_code == 200:
                            logger.info(f"✅ Telegram reminder delivered: {message}")
                        else:
                            logger.error(f"❌ Telegram API error {response.status_code}: {response.text}")
                else:
                    logger.error("❌ No Telegram bot token configured")
                    
            except Exception as e:
                logger.error(f"❌ Failed to send Telegram reminder: {e}")
                # Fallback to logging
                logger.info(f"🔔 REMINDER (fallback): {message}")
        
        return f"Reminder {reminder_id} delivered successfully"
        
    except Exception as e:
        logger.error(f"❌ Failed to deliver reminder {reminder_id}: {e}")
        raise

class WorkerSettings:
    """ARQ worker configuration"""
    functions = [send_reminder]
    redis_settings = REDIS_SETTINGS
    # Use default ARQ queue name
    # queue_name = 'nosyagent:reminders'
    
async def startup(ctx: Dict[str, Any]) -> None:
    """Worker startup - called when worker starts"""
    logger.info("🚀 ARQ reminder worker starting up...")
    
    # Initialize global storage
    global STORAGE
    config = Config()
    STORAGE = Storage(config.DB_PATH)
    logger.info("✅ Storage initialized")
    
async def shutdown(ctx: Dict[str, Any]) -> None:
    """Worker shutdown - called when worker stops"""  
    logger.info("🛑 ARQ reminder worker shutting down...")

# Update settings with startup/shutdown
WorkerSettings.on_startup = startup
WorkerSettings.on_shutdown = shutdown

if __name__ == '__main__':
    """Run the ARQ worker directly like: python worker.py"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔧 Starting ARQ worker for reminder processing...")
    print("📝 Use Ctrl+C to stop")
    
    # This is equivalent to: arq worker.WorkerSettings
    from arq import run_worker
    run_worker(WorkerSettings)