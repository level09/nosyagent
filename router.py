import logging

import anthropic

from config import Config

logger = logging.getLogger(__name__)

# Only exact-match trivial messages skip the full agent.
# These genuinely need no memory, no brain, no context.
SIMPLE_MESSAGES = {
    "hi",
    "hey",
    "hello",
    "yo",
    "sup",
    "hola",
    "thanks",
    "thank you",
    "thx",
    "ty",
    "ok",
    "okay",
    "k",
    "kk",
    "cool",
    "nice",
    "great",
    "awesome",
    "good",
    "perfect",
    "sweet",
    "bye",
    "goodbye",
    "later",
    "cya",
    "gn",
    "night",
    "yes",
    "no",
    "yep",
    "nope",
    "yeah",
    "nah",
    "sure",
    "yea",
    "got it",
    "understood",
    "noted",
    "right",
    "lol",
    "haha",
    "hehe",
    "lmao",
    "wow",
    "oh",
    "ah",
    "hmm",
}

SIMPLE_EMOJI = {"👍", "🙏", "❤️", "👌", "😂", "🤣", "💪", "🔥", "✅", "👋", "😊", "🙌"}

COMPLEX_KEYWORDS = {
    "ocr",
    "image",
    "photo",
    "picture",
    "screenshot",
    "attached",
    "attachment",
    "read this",
    "read the",
    "transcribe",
    "extract text",
    "web",
    "search",
    "research",
    "fetch",
    "http://",
    "https://",
}


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

    async def classify(self, message: str, warm: bool = False) -> str:
        normalized = message.strip().lower()
        if any(keyword in normalized for keyword in COMPLEX_KEYWORDS):
            return "complex"
        if self._is_simple(message):
            return "simple"
        # Mid-conversation short messages are follow-ups that need full context
        # ("Will be 400g min", "No i meant ribeye") — never send them to the
        # context-poor quick path.
        if warm:
            return "complex"
        # Short casual messages without questions don't need tools/memory
        words = message.strip().split()
        if len(words) <= 6 and "?" not in message:
            return await self._haiku_classify(message)
        return "complex"

    async def _haiku_classify(self, message: str) -> str:
        """Use Haiku to classify ambiguous short messages. ~10 tokens, <$0.001."""
        try:
            response = await self.client.messages.create(
                model=self.haiku,
                max_tokens=10,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Is this message a casual/social remark (SIMPLE) or does it need "
                            "personal memory, tools, or detailed advice (COMPLEX)? "
                            f'Reply with one word.\n\nMessage: "{message}"'
                        ),
                    }
                ],
            )
            result = response.content[0].text.strip().lower()
            classification = "simple" if "simple" in result else "complex"
            logger.info(f"route: '{message[:40]}' -> {classification}")
            return classification
        except Exception:
            return "complex"

    async def quick_reply(self, message: str, recent_context: str = "") -> str:
        """Quick reply for trivial messages only."""
        try:
            content = message
            if recent_context:
                content = (
                    f"[Recent conversation]\n{recent_context}\n\n[Message]\n{message}"
                )

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
