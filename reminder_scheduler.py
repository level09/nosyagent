#!/usr/bin/env python3
"""
Reminder scheduling utility
Handles enqueueing reminders to ARQ for future delivery
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from arq import create_pool
from arq.connections import RedisSettings
from storage import Storage
from config import Config

logger = logging.getLogger(__name__)

# Redis settings for ARQ (same as worker)
REDIS_SETTINGS = RedisSettings(host='localhost', port=6379, database=0)

class ReminderScheduler:
    """Handles scheduling reminders via ARQ"""
    
    def __init__(self):
        self.redis_pool = None
        
    async def connect(self):
        """Connect to Redis pool"""
        if not self.redis_pool:
            self.redis_pool = await create_pool(REDIS_SETTINGS)
            
    async def close(self):
        """Close Redis connection"""
        if self.redis_pool:
            await self.redis_pool.close()
            
    async def schedule_reminder(self, chat_id: str, message: str, scheduled_time: datetime) -> bool:
        """
        Schedule a reminder for future delivery
        
        Args:
            chat_id: Chat ID to send reminder to
            message: Reminder message
            scheduled_time: When to send the reminder
            
        Returns:
            bool: True if scheduled successfully
        """
        try:
            # Store reminder in database first
            config = Config()
            storage = Storage(config.DB_PATH)
            reminder_id = await storage.store_reminder(chat_id, message, scheduled_time)
            
            # Connect to Redis if needed
            await self.connect()
            
            # Calculate delay until scheduled time (use fresh timestamp to avoid drift)
            current_time = datetime.now()
            delay_seconds = (scheduled_time - current_time).total_seconds()
            
            # Add small buffer to account for processing time
            if delay_seconds < 0.5:
                logger.warning(f"Reminder scheduled for past/immediate time (delay: {delay_seconds:.2f}s), adding 1 second buffer")
                delay_seconds = 1.0
                
            # Enqueue the reminder task
            job = await self.redis_pool.enqueue_job(
                'send_reminder',
                reminder_id,
                chat_id, 
                message,
                _defer_by=timedelta(seconds=delay_seconds)
            )
            
            logger.info(f"✅ Scheduled reminder {reminder_id} for {scheduled_time} (in {delay_seconds:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to schedule reminder: {e}")
            return False

# Global scheduler instance
_scheduler = None

async def get_scheduler() -> ReminderScheduler:
    """Get global scheduler instance"""
    global _scheduler
    if not _scheduler:
        _scheduler = ReminderScheduler()
    return _scheduler

async def schedule_reminder_task(chat_id: str, message: str, scheduled_time: datetime) -> bool:
    """Convenience function to schedule a reminder"""
    scheduler = await get_scheduler()
    return await scheduler.schedule_reminder(chat_id, message, scheduled_time)

if __name__ == '__main__':
    """Test scheduling a reminder"""
    async def test():
        print("🧪 Testing reminder scheduling...")
        
        # Schedule a test reminder for 10 seconds from now
        test_time = datetime.now() + timedelta(seconds=10)
        success = await schedule_reminder_task("test_chat", "Test reminder!", test_time)
        
        if success:
            print(f"✅ Test reminder scheduled for {test_time}")
        else:
            print("❌ Failed to schedule test reminder")
            
    asyncio.run(test())