import logging

import anthropic

from config import Config

logger = logging.getLogger(__name__)

SIMPLE_MESSAGES = {
    "hi", "hey", "hello", "yo", "sup", "hola",
    "thanks", "thank you", "thx", "ty",
    "ok", "okay", "k", "kk",
    "cool", "nice", "great", "awesome", "good", "perfect", "sweet",
    "bye", "goodbye", "later", "cya", "gn", "night",
    "yes", "no", "yep", "nope", "yeah", "nah", "sure", "yea",
    "got it", "understood", "noted", "right",
    "lol", "haha", "hehe", "lmao",
    "wow", "oh", "ah", "hmm",
}

SIMPLE_EMOJI = {"👍", "🙏", "❤️", "👌", "😂", "🤣", "💪", "🔥", "✅", "👋", "😊", "🙌"}

QUICK_SYSTEM = (
    "You are a warm, helpful personal assistant on Telegram. "
    "Keep responses brief and natural. 1-2 sentences max."
)


class Router:
    def __init__(self, config: Config):
        self.client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        self.haiku = config.HAIKU_MODEL
        self.sonnet = config.SONNET_MODEL

    def _is_simple(self, message: str) -> bool:
        normalized = message.strip().lower().rstrip("!?.,")
        if normalized in SIMPLE_MESSAGES:
            return True
        if message.strip() in SIMPLE_EMOJI:
            return True
        words = normalized.split()
        if len(words) <= 2 and "?" not in message:
            return True
        return False

    async def classify(self, message: str) -> str:
        if self._is_simple(message):
            return "simple"

        try:
            response = await self.client.messages.create(
                model=self.haiku,
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": (
                        "Classify this user message as SIMPLE or COMPLEX.\n"
                        "SIMPLE = greeting, acknowledgment, short reply, casual chat, "
                        "anything that needs no tools or memory lookup.\n"
                        "COMPLEX = needs personal memory, web search, scheduling, "
                        "analysis, or a detailed response.\n\n"
                        f"Message: {message}\n\nOne word:"
                    ),
                }],
            )
            result = response.content[0].text.strip().upper()
            return "simple" if "SIMPLE" in result else "complex"
        except Exception as e:
            logger.warning(f"Classification failed, defaulting to complex: {e}")
            return "complex"

    async def quick_reply(self, message: str, recent_context: str = "") -> str:
        try:
            content = message
            if recent_context:
                content = f"[Recent conversation]\n{recent_context}\n\n[Message]\n{message}"

            response = await self.client.messages.create(
                model=self.haiku,
                max_tokens=200,
                system=QUICK_SYSTEM,
                messages=[{"role": "user", "content": content}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Quick reply failed: {e}")
            return "Hey! 👋"
