import logging

import anthropic

from config import Config

logger = logging.getLogger(__name__)

# Only exact-match trivial messages skip the full agent.
# These genuinely need no memory, no brain, no context.
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


class Router:
    def __init__(self, config: Config):
        self.client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        self.haiku = config.HAIKU_MODEL

    def _is_simple(self, message: str) -> bool:
        """Only exact-match greetings/acks. No word-count heuristic."""
        normalized = message.strip().lower().rstrip("!?.,")
        if normalized in SIMPLE_MESSAGES:
            return True
        if message.strip() in SIMPLE_EMOJI:
            return True
        return False

    async def classify(self, message: str) -> str:
        if self._is_simple(message):
            return "simple"
        # Everything else goes through the full agent.
        # The Haiku classifier was mis-routing too many real messages.
        return "complex"

    async def quick_reply(self, message: str, recent_context: str = "") -> str:
        """Quick reply for trivial messages only."""
        try:
            content = message
            if recent_context:
                content = f"[Recent conversation]\n{recent_context}\n\n[Message]\n{message}"

            response = await self.client.messages.create(
                model=self.haiku,
                max_tokens=200,
                system=(
                    "You are a warm, helpful personal assistant on Telegram. "
                    "Keep responses brief and natural. 1-2 sentences max."
                ),
                messages=[{"role": "user", "content": content}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Quick reply failed: {e}")
            return "Hey! 👋"
