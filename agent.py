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
from notion_tools import NotionService
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

        self.notion = None
        if config.NOTION_TOKEN:
            self.notion = NotionService(config.NOTION_TOKEN)
            logger.info("Notion integration enabled")

        self.semantic_memory = None
        if SEMANTIC_MEMORY_AVAILABLE and semantic_memory_path:
            try:
                self.semantic_memory = SemanticMemory(semantic_memory_path)
                logger.info(f"Semantic memory enabled at {semantic_memory_path}")
            except Exception as e:
                logger.warning(f"Semantic memory unavailable: {e}")

    def _get_claude_tools(self) -> List[Dict[str, Any]]:
        tools = [
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

        # Notion tools (only if token configured)
        if self.notion:
            tools.extend([
                {
                    "name": "notion_search",
                    "description": "Search the user's Notion workspace for pages, notes, docs, or databases.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query (e.g. 'meeting notes', 'project plan')"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "notion_read",
                    "description": "Read the full content of a Notion page by its ID. Use after notion_search to read a specific result.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "page_id": {"type": "string", "description": "The Notion page ID to read"},
                        },
                        "required": ["page_id"],
                    },
                },
                {
                    "name": "notion_create",
                    "description": "Create a new page in the user's Notion. Use for saving notes, plans, summaries, or any structured content.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Page title"},
                            "content": {"type": "string", "description": "Page content in markdown format"},
                            "parent_page_id": {"type": "string", "description": "Optional: parent page ID to nest under"},
                        },
                        "required": ["title", "content"],
                    },
                },
                {
                    "name": "notion_append",
                    "description": "Append content to an existing Notion page. Use to add notes, updates, or entries to an existing page.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "page_id": {"type": "string", "description": "The Notion page ID to append to"},
                            "content": {"type": "string", "description": "Content to append in markdown format"},
                        },
                        "required": ["page_id", "content"],
                    },
                },
            ])

        return tools

    # === Single code path: everything goes through _prepare_context + _run_tool_loop ===

    async def _prepare_context(self, chat_id: str, user_message: str):
        """Build system prompt and user message with full context. Used by all paths."""
        recent_messages = await self.storage.get_recent_conversations(chat_id, limit=50)
        user_context = await self.storage.read_user_context(chat_id)

        semantic_context = []
        if self.semantic_memory:
            try:
                semantic_context = self.semantic_memory.search(user_message, chat_id, limit=3)
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        system_prompt = self._build_system_prompt()
        built_message = self._build_adaptive_message(
            user_message, user_context, recent_messages, semantic_context
        )

        brain_len = len(user_context) if user_context else 0
        logger.info(
            f"context: chat={chat_id} brain={brain_len} history={len(recent_messages)} "
            f"semantic={len(semantic_context)} budget={self.config.CONTEXT_TOKEN_BUDGET}"
        )

        return system_prompt, built_message, recent_messages

    async def _run_tool_loop(self, messages: list, api_kwargs: dict, chat_id: str, on_tool=None):
        """Run non-streaming tool turns. Calls on_tool(name) for status updates."""
        for turn in range(5):
            response = await self.client.messages.create(messages=messages, **api_kwargs)
            if response.stop_reason != "tool_use":
                return messages, response
            messages.append({"role": "assistant", "content": response.content})
            tool_results_turn = []
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    if on_tool:
                        await on_tool(block.name)
                    result = await self._handle_tool_call(block, chat_id)
                    tool_results_turn.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result["content"],
                    })
                    logger.info(f"tool: {block.name} success={result['success']}")
            messages.append({"role": "user", "content": tool_results_turn})
        return messages, response

    async def _finalize(self, chat_id: str, user_message: str, response_text: str, recent_messages):
        """Companion wrap + store. Used by all paths."""
        final = response_text
        if self.companion and final.strip():
            try:
                final = await self.companion.wrap_response(
                    chat_id, user_message, response_text, recent_messages
                )
            except Exception as exc:
                logger.warning(f"Companion wrapper failed: {exc}")
        if not final.strip():
            final = "Done"
        await self.storage.store_conversation(chat_id, user_message, final)
        return final

    # === Public API ===

    async def process_message(self, chat_id: str, user_message: str) -> tuple[str, Optional[dict]]:
        """Non-streaming response. Used by CLI and as fallback."""
        try:
            system_prompt, built_message, recent_messages = await self._prepare_context(chat_id, user_message)
            messages = [{"role": "user", "content": built_message}]
            api_kwargs = dict(
                model=self.config.SONNET_MODEL,
                max_tokens=self.config.CLAUDE_MAX_TOKENS,
                system=system_prompt,
                tools=self._get_claude_tools(),
            )

            messages, response = await self._run_tool_loop(messages, api_kwargs, chat_id)

            # Extract text from final response
            full_text = ""
            for block in response.content:
                if block.type == "text":
                    full_text += block.text

            final = await self._finalize(chat_id, user_message, full_text, recent_messages)
            return final, None

        except Exception as e:
            logger.error(f"process_message error for {chat_id}: {e}")
            return "I'm having trouble right now. Could you try asking again?", None

    async def process_message_with_image(
        self, chat_id: str, user_message: str, image_bytes: bytes
    ) -> tuple[str, Optional[dict]]:
        try:
            system_prompt, built_message, recent_messages = await self._prepare_context(chat_id, user_message)

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
                {"type": "text", "text": built_message},
            ]

            messages = [{"role": "user", "content": message_content}]
            api_kwargs = dict(
                model=self.config.SONNET_MODEL,
                max_tokens=self.config.CLAUDE_MAX_TOKENS,
                system=system_prompt,
                tools=self._get_claude_tools(),
            )

            messages, response = await self._run_tool_loop(messages, api_kwargs, chat_id)

            full_text = ""
            for block in response.content:
                if block.type == "text":
                    full_text += block.text

            final = await self._finalize(chat_id, user_message, full_text, recent_messages)
            return final, None

        except Exception as e:
            logger.error(f"process_image error for {chat_id}: {e}")
            return "I had trouble processing your image. Could you try again?", None

    TOOL_LABELS = {
        "web_search": "searching the web",
        "notion_search": "searching Notion",
        "notion_read": "reading from Notion",
        "notion_create": "writing to Notion",
        "notion_append": "updating Notion",
        "update_brain_file": "updating memory",
        "read_brain_file": "checking memory",
        "query_entities": "looking up facts",
        "schedule_message": "setting reminder",
    }

    async def stream_response(self, chat_id: str, user_message: str, on_status=None) -> AsyncGenerator[str, None]:
        """Stream text deltas. Calls on_status(label) during tool turns for UX feedback."""
        async def on_tool(name):
            label = self.TOOL_LABELS.get(name, name)
            if on_status:
                await on_status(label)

        try:
            if on_status:
                await on_status("thinking")

            system_prompt, built_message, recent_messages = await self._prepare_context(chat_id, user_message)
            messages = [{"role": "user", "content": built_message}]
            api_kwargs = dict(
                model=self.config.SONNET_MODEL,
                max_tokens=self.config.CLAUDE_MAX_TOKENS,
                system=system_prompt,
                tools=self._get_claude_tools(),
            )

            messages, _ = await self._run_tool_loop(messages, api_kwargs, chat_id, on_tool=on_tool)

            # Final turn: stream text
            full_response = ""
            async with self.client.messages.stream(messages=messages, **api_kwargs) as stream:
                async for text in stream.text_stream:
                    full_response += text
                    yield text

            # Companion may append extra text
            final = await self._finalize(chat_id, user_message, full_response, recent_messages)
            if final != full_response:
                extra = final[len(full_response):]
                if extra:
                    yield extra

        except Exception as e:
            logger.error(f"stream error for {chat_id}: {e}")
            yield f"Sorry, I encountered an error: {e}"

    # Backward compat for CLI
    async def stream_chat(self, message: str, chat_id: int, context: str = "") -> AsyncGenerator[str, None]:
        try:
            yield "Thinking..."
            response, _ = await self.process_message(str(chat_id), message)
            yield response
        except Exception as e:
            logger.error(f"stream_chat error: {e}")
            yield f"Sorry, I encountered an error: {e}"

    async def stream_chat_with_image(
        self, message: str, chat_id: int, image_bytes: bytes, context: str = ""
    ) -> AsyncGenerator[str, None]:
        try:
            yield "Analyzing image..."
            response, _ = await self.process_message_with_image(str(chat_id), message, image_bytes)
            yield response
        except Exception as e:
            logger.error(f"stream_chat_with_image error: {e}")
            yield f"Sorry, error analyzing the image: {e}"

    # === Prompt building ===

    def _build_system_prompt(self) -> str:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tools = (
            "- Reminders: schedule_message, confirm briefly\n"
            "- Memory: update_brain_file for lasting insights only\n"
            "- Facts: query_entities for quick structured lookups\n"
            "- Search: web_search, cite sources briefly"
        )
        if self.notion:
            tools += "\n- Notion: search, read, create, and append to the user's Notion workspace"
        return (
            f"You are a proactive life optimization AI assistant. Current time: {current_time}\n\n"
            "FORMAT (Telegram):\n"
            "- Keep responses readable on mobile, 1-3 short paragraphs.\n"
            "- Skip preambles. Be conversational but efficient.\n"
            "- Use bullet points when listing multiple items.\n\n"
            "Core capabilities: Productivity, Health, Relationships, Finance, Goals.\n\n"
            f"Tools:\n{tools}\n\n"
            "Be helpful and warm, but don't over-explain."
        )

    def _build_adaptive_message(
        self,
        user_message: str,
        user_context: str,
        recent_messages: List[Message],
        semantic_context: List = None,
    ) -> str:
        budget = self.config.CONTEXT_TOKEN_BUDGET
        layers = []
        used = 0

        msg_tokens = estimate_tokens(user_message) + 50
        used += msg_tokens

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

        if semantic_context:
            semantic_text = self._format_semantic_context(semantic_context)
            semantic_tokens = estimate_tokens(semantic_text)
            if used + semantic_tokens < budget * 0.7:
                layers.append(semantic_text)
                used += semantic_tokens

        remaining = budget - used
        if recent_messages and remaining > 100:
            conversation_lines = []
            for msg in reversed(recent_messages):
                if "REMINDER:" in msg.agent_response:
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
                logger.info(f"brain updated for {chat_id}: {reason}")
                if self.semantic_memory:
                    try:
                        self.semantic_memory.reindex_brain(chat_id, content)
                    except Exception as e:
                        logger.warning(f"Reindex failed: {e}")
                asyncio.create_task(self._extract_entities(chat_id, content))
                return {"tool_name": tool_name, "content": f"Brain updated: {reason}", "success": True}

            elif tool_name == "read_brain_file":
                content = await self.storage.read_user_context(chat_id)
                return {"tool_name": tool_name, "content": content or "No personal context.", "success": True}

            elif tool_name == "query_entities":
                query = tool_input.get("query", "")
                entities = await self.storage.search_entities(chat_id, query)
                if entities:
                    result = "\n".join(f"- {e['subject']} {e['predicate']} {e['object']}" for e in entities)
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

            elif tool_name == "notion_search":
                query = tool_input.get("query", "")
                results = await self.notion.search(query, limit=5)
                if results:
                    lines = []
                    for r in results:
                        lines.append(f"- [{r['title']}] id={r['id']} ({r['last_edited'][:10]})")
                    return {"tool_name": tool_name, "content": "\n".join(lines), "success": True}
                return {"tool_name": tool_name, "content": "No results found.", "success": True}

            elif tool_name == "notion_read":
                page_id = tool_input.get("page_id", "")
                page = await self.notion.read_page(page_id)
                # Truncate if too long for tool result
                content = page["content"]
                if len(content) > 4000:
                    content = content[:4000] + "\n\n... (truncated)"
                return {"tool_name": tool_name, "content": f"# {page['title']}\n\n{content}", "success": True}

            elif tool_name == "notion_create":
                title = tool_input.get("title", "")
                content = tool_input.get("content", "")
                parent = tool_input.get("parent_page_id")
                result = await self.notion.create_page(title, content, parent)
                return {"tool_name": tool_name, "content": f"Created: {result['title']} ({result['url']})", "success": True}

            elif tool_name == "notion_append":
                page_id = tool_input.get("page_id", "")
                content = tool_input.get("content", "")
                await self.notion.append_to_page(page_id, content)
                return {"tool_name": tool_name, "content": "Content appended.", "success": True}

            else:
                return {"tool_name": tool_name, "content": f"Unknown tool: {tool_name}", "success": False}

        except Exception as e:
            logger.error(f"tool {tool_name} failed: {e}")
            return {"tool_name": tool_name, "content": "Action failed. Please try again.", "success": False}

    async def _extract_entities(self, chat_id: str, brain_content: str):
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
                        'Return a JSON array of {"subject", "predicate", "object"} objects.\n\n'
                        "Rules:\n"
                        "- Subject is 'user' for facts about the person\n"
                        "- Normalize predicates: takes, works_at, lives_in, spouse, child, "
                        "goal, preference, allergy, hobby, tracks, age, etc.\n"
                        "- Only extract concrete, durable facts\n\n"
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
                logger.info(f"entities: extracted {len(triples)} for {chat_id}")
        except Exception as e:
            logger.warning(f"Entity extraction failed for {chat_id}: {e}")

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
