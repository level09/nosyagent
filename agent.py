import asyncio
import base64
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import anthropic
import dateparser

from companion import CompanionService
from config import Config
from reminder_scheduler import schedule_reminder_task
from storage import Message, Storage

# Optional semantic memory
try:
    from semantic_memory import SemanticMemory

    SEMANTIC_MEMORY_AVAILABLE = True
except ImportError:
    SEMANTIC_MEMORY_AVAILABLE = False
    SemanticMemory = None

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


class AIAgent:
    def __init__(
        self,
        config: Config,
        storage: Storage,
        companion_service: Optional[CompanionService] = None,
        semantic_memory_path: Optional[Path] = None,
    ):
        self.config = config
        self.client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        self.storage = storage
        self.companion = companion_service

        self.semantic_memory = None
        if SEMANTIC_MEMORY_AVAILABLE and semantic_memory_path:
            try:
                self.semantic_memory = SemanticMemory(semantic_memory_path)
                logger.info(f"Semantic memory enabled at {semantic_memory_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic memory: {e}")

    def _get_claude_tools(self) -> List[Dict[str, Any]]:
        return [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
            {
                "name": "update_brain_file",
                "description": "Update the user's personal brain file. Use for DURABLE, LONG-TERM facts only.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "New content (markdown). Be concise."},
                        "reason": {"type": "string", "description": "Why this is worth remembering."},
                    },
                    "required": ["content", "reason"],
                },
            },
            {
                "name": "read_brain_file",
                "description": "Read the user's brain file.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "query_entities",
                "description": (
                    "Search structured facts about the user. Returns specific triples "
                    "like medications, relationships, preferences, work info. "
                    "Faster and more precise than reading the full brain file."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to look up (e.g. 'medication', 'spouse', 'allergy', 'work')",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "schedule_message",
                "description": "Schedule a reminder. Understands natural language time.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "The message to send."},
                        "when": {"type": "string", "description": "Time expression (e.g. 'in 5 mins', 'friday at noon')."},
                    },
                    "required": ["message", "when"],
                },
            },
        ]

    # === Main entry points ===

    async def process_message(self, chat_id: str, user_message: str) -> tuple[str, Optional[dict]]:
        try:
            recent_messages = await self.storage.get_recent_conversations(chat_id, limit=50)
            user_context = await self.storage.read_user_context(chat_id)

            semantic_context = []
            if self.semantic_memory:
                try:
                    semantic_context = self.semantic_memory.search(user_message, chat_id, limit=3)
                except Exception as e:
                    logger.warning(f"Semantic search failed: {e}")

            system_prompt = self._build_system_prompt()
            user_msg = self._build_adaptive_message(
                user_message, user_context, recent_messages, semantic_context
            )

            response, tool_results = await self._call_claude_with_tools(
                system_prompt, user_msg, chat_id
            )

            final_response = response
            if self.companion:
                try:
                    final_response = await self.companion.wrap_response(
                        chat_id, user_message, response, recent_messages
                    )
                except Exception as exc:
                    logger.warning(f"Companion wrapper failed: {exc}")

            await self.storage.store_conversation(chat_id, user_message, final_response)
            return final_response, None

        except Exception as e:
            logger.error(f"Error processing message for {chat_id}: {e}")
            return "I'm having trouble right now. Could you try asking again?", None

    async def process_message_with_image(
        self, chat_id: str, user_message: str, image_bytes: bytes
    ) -> tuple[str, Optional[dict]]:
        try:
            recent_messages = await self.storage.get_recent_conversations(chat_id, limit=50)
            user_context = await self.storage.read_user_context(chat_id)

            semantic_context = []
            if self.semantic_memory:
                try:
                    semantic_context = self.semantic_memory.search(user_message, chat_id, limit=3)
                except Exception as e:
                    logger.warning(f"Semantic search failed: {e}")

            system_prompt = self._build_system_prompt()
            user_msg = self._build_adaptive_message(
                user_message, user_context, recent_messages, semantic_context
            )

            response, tool_results = await self._call_claude_with_image(
                system_prompt, user_msg, image_bytes, chat_id
            )

            final_response = response
            if self.companion:
                try:
                    final_response = await self.companion.wrap_response(
                        chat_id, user_message, response, recent_messages
                    )
                except Exception as exc:
                    logger.warning(f"Companion wrapper failed (image): {exc}")

            await self.storage.store_conversation(chat_id, user_message, final_response)
            return final_response, None

        except Exception as e:
            logger.error(f"Error processing image for {chat_id}: {e}")
            return "I had trouble processing your image. Could you try again?", None

    # === Prompt building ===

    def _build_system_prompt(self) -> str:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"You are a proactive life optimization AI assistant. Current time: {current_time}\n\n"
            "FORMAT (Telegram):\n"
            "- Keep responses readable on mobile, 1-3 short paragraphs.\n"
            "- Skip preambles. Be conversational but efficient.\n"
            "- Use bullet points when listing multiple items.\n\n"
            "Core capabilities: Productivity, Health, Relationships, Finance, Goals.\n\n"
            "Tools:\n"
            "- Reminders: schedule_message, confirm briefly\n"
            "- Memory: update_brain_file for lasting insights only\n"
            "- Facts: query_entities for quick structured lookups\n"
            "- Search: web_search, cite sources briefly\n\n"
            "Be helpful and warm, but don't over-explain."
        )

    def _build_adaptive_message(
        self,
        user_message: str,
        user_context: str,
        recent_messages: List[Message],
        semantic_context: List = None,
    ) -> str:
        """Token-aware context builder. Fills budget greedily: brain > semantic > history."""
        budget = self.config.CONTEXT_TOKEN_BUDGET
        layers = []
        used = 0

        # Reserve space for current message + formatting
        msg_tokens = estimate_tokens(user_message) + 50
        used += msg_tokens

        # Layer 1: Brain (always include)
        if user_context:
            brain_text = f"[Personal Context]\n{user_context}"
        else:
            brain_text = (
                "[Personal Context]\nNo personal context yet. "
                "Share your goals, constraints, or routines so I can help better."
            )
        brain_tokens = estimate_tokens(brain_text)
        layers.append(brain_text)
        used += brain_tokens

        # Layer 2: Semantic search results (cap at 30% of budget)
        if semantic_context:
            semantic_text = self._format_semantic_context(semantic_context)
            semantic_tokens = estimate_tokens(semantic_text)
            if used + semantic_tokens < budget * 0.7:
                layers.append(semantic_text)
                used += semantic_tokens

        # Layer 3: Conversation history, newest first, fill remaining budget
        remaining = budget - used
        if recent_messages and remaining > 100:
            conversation_lines = []
            for msg in reversed(recent_messages):
                # skip reminder noise
                if "REMINDER:" in msg.agent_response or "🔔" in msg.agent_response:
                    continue
                line = f"User: {msg.user_message}\nAssistant: {msg.agent_response}"
                line_tokens = estimate_tokens(line)
                if line_tokens > remaining:
                    break
                conversation_lines.insert(0, line)
                remaining -= line_tokens

            if conversation_lines:
                conv_text = "[Recent Conversation]\n" + "\n\n".join(conversation_lines)
                layers.append(conv_text)

        full_message = "\n\n".join(layers)
        return f"{full_message}\n\n[Current Message]\n{user_message}"

    def _format_semantic_context(self, semantic_context: List) -> str:
        now = datetime.utcnow()
        lines = ["[Relevant Past Context]"]
        for chunk in semantic_context:
            age_days = (now - chunk.timestamp).days
            if age_days == 0:
                age_str = "today"
            elif age_days == 1:
                age_str = "yesterday"
            elif age_days < 7:
                age_str = f"{age_days}d ago"
            else:
                age_str = f"{age_days // 7}w ago"
            lines.append(f"- ({age_str}) {chunk.content}")
        return "\n".join(lines)

    # === Tool handling ===

    async def _handle_tool_call(self, tool_call, chat_id: str) -> Dict[str, Any]:
        tool_name = tool_call.name
        tool_input = tool_call.input

        try:
            if tool_name == "update_brain_file":
                content = tool_input.get("content", "")
                reason = tool_input.get("reason", "")

                await self.storage.update_user_context(chat_id, content, reason)
                logger.info(f"Updated brain for {chat_id}: {reason}")

                if self.semantic_memory:
                    try:
                        self.semantic_memory.reindex_brain(chat_id, content)
                    except Exception as e:
                        logger.warning(f"Failed to reindex brain: {e}")

                # Extract entity triples in background
                asyncio.create_task(self._extract_entities(chat_id, content))

                return {"tool_name": tool_name, "content": f"Brain updated: {reason}", "success": True}

            elif tool_name == "read_brain_file":
                content = await self.storage.read_user_context(chat_id)
                return {"tool_name": tool_name, "content": content or "No personal context.", "success": True}

            elif tool_name == "query_entities":
                query = tool_input.get("query", "")
                entities = await self.storage.search_entities(chat_id, query)
                if entities:
                    result = "\n".join(
                        f"- {e['subject']} {e['predicate']} {e['object']}" for e in entities
                    )
                else:
                    result = "No matching facts found."
                return {"tool_name": tool_name, "content": result, "success": True}

            elif tool_name == "schedule_message":
                message = tool_input.get("message", "")
                when_text = tool_input.get("when", "")

                scheduled_time = self._parse_when(when_text)
                if not scheduled_time:
                    return {"tool_name": tool_name, "content": f"Could not parse time: {when_text}", "success": False}

                success = await schedule_reminder_task(chat_id, message, scheduled_time)
                time_str = scheduled_time.strftime("%I:%M %p on %b %d")
                return {
                    "tool_name": tool_name,
                    "content": f"Reminder scheduled for {time_str}" if success else "Failed to schedule",
                    "success": success,
                }

            else:
                return {"tool_name": tool_name, "content": f"Unknown tool: {tool_name}", "success": False}

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"tool_name": tool_name, "content": "Action failed. Please try again.", "success": False}

    async def _extract_entities(self, chat_id: str, brain_content: str):
        """Extract entity triples from brain content using Haiku."""
        if not brain_content.strip():
            return
        try:
            response = await self.client.messages.create(
                model=self.config.HAIKU_MODEL,
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": (
                        "Extract factual triples from this personal context. "
                        "Return a JSON array of {\"subject\", \"predicate\", \"object\"} objects.\n\n"
                        "Rules:\n"
                        "- Subject is 'user' for facts about the person\n"
                        "- Normalize predicates: takes, works_at, lives_in, spouse, child, "
                        "goal, preference, allergy, hobby, tracks, age, etc.\n"
                        "- Only extract concrete, durable facts\n"
                        "- Skip opinions, transient states, conversation artifacts\n\n"
                        f"Context:\n{brain_content}\n\n"
                        "Return ONLY a JSON array, nothing else."
                    ),
                }],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            triples = json.loads(text)
            if isinstance(triples, list):
                await self.storage.replace_entities(chat_id, triples)
                logger.info(f"Extracted {len(triples)} entities for {chat_id}")
        except Exception as e:
            logger.warning(f"Entity extraction failed for {chat_id}: {e}")

    # === Claude API calls ===

    async def _call_claude_with_tools(
        self, system_prompt: str, user_message: str, chat_id: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        max_retries = self.config.CLAUDE_MAX_RETRIES
        base_delay = self.config.CLAUDE_BASE_DELAY
        tool_results = []

        for attempt in range(max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.config.SONNET_MODEL,
                    max_tokens=self.config.CLAUDE_MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                    tools=self._get_claude_tools(),
                )

                messages = [{"role": "user", "content": user_message}]

                while response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})

                    tool_results_turn = []
                    for block in response.content:
                        if hasattr(block, "type") and block.type == "tool_use":
                            result = await self._handle_tool_call(block, chat_id)
                            tool_results.append(result)
                            tool_results_turn.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result["content"],
                            })

                    messages.append({"role": "user", "content": tool_results_turn})
                    response = await self.client.messages.create(
                        model=self.config.SONNET_MODEL,
                        max_tokens=self.config.CLAUDE_MAX_TOKENS,
                        system=system_prompt,
                        messages=messages,
                        tools=self._get_claude_tools(),
                    )

                full_text = ""
                for block in response.content:
                    if block.type == "text":
                        full_text += block.text

                if response.stop_reason == "max_tokens" and full_text.strip():
                    full_text += "... [continued]"

                if not full_text.strip():
                    if tool_results:
                        return "", tool_results
                    if response.stop_reason == "max_tokens":
                        return "I need to continue this response...", tool_results

                return full_text, tool_results

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt)
                logger.warning(f"Claude API failed (attempt {attempt + 1}), retry in {delay}s: {e}")
                await asyncio.sleep(delay)

        return "", tool_results

    async def _call_claude_with_image(
        self, system_prompt: str, user_message: str, image_bytes: bytes, chat_id: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        max_retries = self.config.CLAUDE_MAX_RETRIES
        base_delay = self.config.CLAUDE_BASE_DELAY
        tool_results = []

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_format = "image/jpeg"
        if image_bytes.startswith(b"\x89PNG"):
            image_format = "image/png"
        elif image_bytes.startswith(b"GIF"):
            image_format = "image/gif"
        elif image_bytes.startswith(b"WEBP", 8):
            image_format = "image/webp"

        message_content = [
            {"type": "image", "source": {"type": "base64", "media_type": image_format, "data": image_base64}},
        ]
        if user_message.strip():
            message_content.append({"type": "text", "text": user_message})

        for attempt in range(max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.config.SONNET_MODEL,
                    max_tokens=self.config.CLAUDE_MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": message_content}],
                    tools=self._get_claude_tools(),
                )

                messages = [{"role": "user", "content": message_content}]

                while response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})

                    tool_results_turn = []
                    for block in response.content:
                        if hasattr(block, "type") and block.type == "tool_use":
                            result = await self._handle_tool_call(block, chat_id)
                            tool_results.append(result)
                            tool_results_turn.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result["content"],
                            })

                    messages.append({"role": "user", "content": tool_results_turn})
                    response = await self.client.messages.create(
                        model=self.config.SONNET_MODEL,
                        max_tokens=self.config.CLAUDE_MAX_TOKENS,
                        system=system_prompt,
                        messages=messages,
                        tools=self._get_claude_tools(),
                    )

                full_text = ""
                for block in response.content:
                    if block.type == "text":
                        full_text += block.text

                if response.stop_reason == "max_tokens" and full_text.strip():
                    full_text += "... [continued]"

                if not full_text.strip():
                    if tool_results:
                        return "", tool_results
                    if response.stop_reason == "max_tokens":
                        return "I need to continue analyzing this image...", tool_results

                return full_text, tool_results

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt)
                logger.warning(f"Claude image API failed (attempt {attempt + 1}), retry in {delay}s: {e}")
                await asyncio.sleep(delay)

        return "", tool_results

    def _parse_when(self, when_text: str) -> Optional[datetime]:
        settings = {
            "PREFER_DATES_FROM": "future",
            "PREFER_DAY_OF_MONTH": "first",
            "RETURN_AS_TIMEZONE_AWARE": False,
        }
        cleaned = when_text.lower().strip()
        dt = dateparser.parse(cleaned, settings=settings)
        if dt and dt < datetime.now() and "ago" not in cleaned:
            dt = dt + timedelta(days=1)
        return dt

    # === Streaming interface (backward compat) ===

    async def stream_chat(self, message: str, chat_id: int, context: str = "") -> AsyncGenerator[str, None]:
        try:
            yield "🤔 Thinking..."
            response, _ = await self.process_message(str(chat_id), message)
            if not response or not response.strip():
                response = "✓ Done"
            yield response
        except Exception as e:
            logger.error(f"Error in stream_chat: {e}")
            yield "🤔 Thinking..."
            yield f"Sorry, I encountered an error: {e}"

    async def stream_chat_with_image(
        self, message: str, chat_id: int, image_bytes: bytes, context: str = ""
    ) -> AsyncGenerator[str, None]:
        try:
            yield "🤔 Analyzing image..."
            response, _ = await self.process_message_with_image(str(chat_id), message, image_bytes)
            if not response or not response.strip():
                response = "✓ Done"
            yield response
        except Exception as e:
            logger.error(f"Error in stream_chat_with_image: {e}")
            yield "🤔 Analyzing image..."
            yield f"Sorry, error analyzing the image: {e}"
