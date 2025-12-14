import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    
    # Webhook
    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
    PORT = int(os.getenv("PORT", 8000))
    
    # Security
    ALLOWED_CHAT_IDS = [int(x) for x in os.getenv("ALLOWED_CHAT_IDS", "").split(",") if x.strip()]
    
    # Data paths
    DATA_DIR = Path("data")
    DB_PATH = DATA_DIR / "nosyagent.db"
    USERS_DIR = DATA_DIR / "users"
    LOGS_DIR = DATA_DIR / "logs"
    
    # Conversation settings
    MAX_CONTEXT_MESSAGES = 10
    MAX_USER_CONTEXT_LENGTH = 2000
    
    # Message limits
    MAX_MESSAGE_LENGTH = 4000
    TELEGRAM_MAX_LENGTH = 4096
    
    # API settings
    CLAUDE_MAX_RETRIES = 3
    CLAUDE_BASE_DELAY = 1.0
    CLAUDE_MAX_TOKENS = 1500  # Balanced for Telegram - not too short, not too long

    # Companion mode
    COMPANION_MODE = os.getenv("COMPANION_MODE", "on").lower()
    COMPANION_MODE_ENABLED = COMPANION_MODE != "off"
    _cards_path = os.getenv("COMPANION_CARDS_PATH")
    COMPANION_CARDS_PATH = Path(_cards_path) if _cards_path else DATA_DIR / "companion_cards.json"

    # Semantic memory (LanceDB)
    SEMANTIC_MEMORY_ENABLED = os.getenv("SEMANTIC_MEMORY", "on").lower() != "off"
    SEMANTIC_MEMORY_PATH = DATA_DIR / "semantic_memory"
    
    def __init__(self):
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.USERS_DIR.mkdir(exist_ok=True)
        self.LOGS_DIR.mkdir(exist_ok=True)
        
        # Validate required settings
        if not self.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
    
    def validate(self):
        """Validate configuration - for backward compatibility"""
        pass


def get_config():
    """Get configuration instance - for backward compatibility"""
    return Config()
