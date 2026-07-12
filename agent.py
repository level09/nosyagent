import asyncio
import base64
import ipaddress
import json
import logging
import re
import secrets
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import anthropic
import dateparser
import httpx

from config import Config
from notion_tools import NotionService
from reminder_scheduler import schedule_reminder_task
from storage import Message, Storage

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


class AIAgent:
    def __init__(
        self,
        config: Config,
        storage: Storage,
    ):
        self.config = config
        self.client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        self.storage = storage

        self.notion = None
        if config.NOTION_TOKEN:
            self.notion = NotionService(config.NOTION_TOKEN)
            logger.info("Notion integration enabled")

    def _get_claude_tools(self) -> List[Dict[str, Any]]:
        tools = [
            {"type": "web_search_20260209", "name": "web_search", "max_uses": 8},
            {
                "name": "update_brain_file",
                "description": "Update the user's personal brain file. Use for DURABLE, LONG-TERM facts only.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "New content (markdown). Be concise.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why this is worth remembering.",
                        },
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
                        "message": {
                            "type": "string",
                            "description": "The message to send.",
                        },
                        "when": {
                            "type": "string",
                            "description": "Time expression (e.g. 'in 5 mins', 'friday at noon').",
                        },
                    },
                    "required": ["message", "when"],
                },
            },
            {
                "name": "web_fetch",
                "description": "Fetch and read a specific HTTP/HTTPS URL. Use when the user gives a link to inspect.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "HTTP or HTTPS URL to fetch.",
                        },
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "share_file",
                "description": (
                    "Publish a document to a public web link and share the URL with the user. "
                    "Use for research reports, summaries, plans, or data too long for chat. "
                    "Use .html with inline CSS for formatted documents (renders in the browser), "
                    ".md/.txt for plain notes, .csv/.json for data. Returns the public URL - "
                    "include it in your reply."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Base filename with extension, e.g. 'sleep-research.html'",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full file content.",
                        },
                    },
                    "required": ["filename", "content"],
                },
            },
        ]

        # Notion tools (only if token configured)
        if self.notion:
            tools.extend(
                [
                    {
                        "name": "notion_search",
                        "description": "Search the user's Notion workspace for pages, notes, docs, or databases.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query (e.g. 'meeting notes', 'project plan')",
                                },
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
                                "page_id": {
                                    "type": "string",
                                    "description": "The Notion page ID to read",
                                },
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
                                "title": {
                                    "type": "string",
                                    "description": "Page title",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Page content in markdown format",
                                },
                                "parent_page_id": {
                                    "type": "string",
                                    "description": "Optional: parent page ID to nest under",
                                },
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
                                "page_id": {
                                    "type": "string",
                                    "description": "The Notion page ID to append to",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Content to append in markdown format",
                                },
                            },
                            "required": ["page_id", "content"],
                        },
                    },
                ]
            )

        # Cache tool definitions (identical across requests)
        if tools:
            tools[-1]["cache_control"] = {"type": "ephemeral"}
        return tools

    def _mark_cache_breakpoint(self, messages: list):
        """Mark last user message for prompt caching. Clears previous message breakpoints."""
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)
        for msg in reversed(messages):
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, str):
                    msg["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif isinstance(content, list) and content:
                    last = content[-1]
                    if isinstance(last, dict):
                        last["cache_control"] = {"type": "ephemeral"}
                break

    # === Single code path: everything goes through _prepare_context + _run_tool_loop ===

    async def _prepare_context(self, chat_id: str, user_message: str):
        """Build system prompt and user message with full context. Used by all paths."""
        recent_messages = await self.storage.get_recent_conversations(chat_id, limit=50)
        user_context = await self.storage.read_user_context(chat_id)

        structured_memory = []
        try:
            structured_memory = await self.storage.search_memory_items(
                chat_id, user_message, limit=5
            )
        except Exception as e:
            logger.warning(f"Structured memory search failed: {e}")

        system_prompt = self._build_system_prompt()
        built_message = self._build_adaptive_message(
            user_message, user_context, recent_messages, structured_memory
        )

        brain_len = len(user_context) if user_context else 0
        logger.info(
            f"context: chat={chat_id} brain={brain_len} history={len(recent_messages)} "
            f"structured={len(structured_memory)} "
            f"budget={self.config.CONTEXT_TOKEN_BUDGET}"
        )

        return system_prompt, built_message, recent_messages

    def _log_usage(self, response, label="api"):
        """Log token usage including cache stats."""
        u = getattr(response, "usage", None)
        if not u:
            return
        parts = [f"{label}: in={u.input_tokens} out={u.output_tokens}"]
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(u, "cache_creation_input_tokens", 0) or 0
        if cache_read or cache_create:
            parts.append(f"cache_read={cache_read} cache_write={cache_create}")
        logger.info(" ".join(parts))

    async def _run_tool_loop(
        self, messages: list, api_kwargs: dict, chat_id: str, on_tool=None
    ):
        """Run non-streaming tool turns. Calls on_tool(name) for status updates."""
        for turn in range(8):
            self._mark_cache_breakpoint(messages)
            response = await self.client.messages.create(
                messages=messages, **api_kwargs
            )
            self._log_usage(response, f"tool_loop[{turn}]")
            if response.stop_reason == "pause_turn":
                # Server-side tool (web search) paused mid-turn; re-send to resume
                messages.append({"role": "assistant", "content": response.content})
                continue
            if response.stop_reason != "tool_use":
                return messages, response
            messages.append({"role": "assistant", "content": response.content})
            tool_results_turn = []
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    if on_tool:
                        await on_tool(block.name)
                    result = await self._handle_tool_call(block, chat_id)
                    tool_results_turn.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result["content"],
                        }
                    )
                    logger.info(f"tool: {block.name} success={result['success']}")
            messages.append({"role": "user", "content": tool_results_turn})
        return messages, response

    async def _finalize(self, chat_id: str, user_message: str, response_text: str):
        """Store the exchange. Used by all paths."""
        final = response_text if response_text.strip() else "Done"
        await self.storage.store_conversation(chat_id, user_message, final)
        return final

    # === Public API ===

    async def process_message(
        self, chat_id: str, user_message: str
    ) -> tuple[str, Optional[dict]]:
        """Non-streaming response. Used by CLI and as fallback."""
        try:
            system_prompt, built_message, recent_messages = await self._prepare_context(
                chat_id, user_message
            )
            messages = [{"role": "user", "content": built_message}]
            api_kwargs = dict(
                model=self.config.SONNET_MODEL,
                max_tokens=self.config.CLAUDE_MAX_TOKENS,
                system=system_prompt,
                tools=self._get_claude_tools(),
            )

            messages, response = await self._run_tool_loop(
                messages, api_kwargs, chat_id
            )

            # Extract text from final response
            full_text = ""
            for block in response.content:
                if block.type == "text":
                    full_text += block.text

            final = await self._finalize(chat_id, user_message, full_text)
            return final, None

        except Exception as e:
            logger.error(f"process_message error for {chat_id}: {e}")
            return "I'm having trouble right now. Could you try asking again?", None

    async def process_message_with_image(
        self, chat_id: str, user_message: str, image_bytes: bytes
    ) -> tuple[str, Optional[dict]]:
        try:
            system_prompt, built_message, recent_messages = await self._prepare_context(
                chat_id, user_message
            )

            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            image_format = "image/jpeg"
            if image_bytes.startswith(b"\x89PNG"):
                image_format = "image/png"
            elif image_bytes.startswith(b"GIF"):
                image_format = "image/gif"
            elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
                image_format = "image/webp"

            image_task = (
                "For this attached image: first OCR/transcribe any visible text exactly. "
                "Then answer the user's request. If external facts are needed, use tools."
            )

            message_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_format,
                        "data": image_base64,
                    },
                },
                {"type": "text", "text": f"{image_task}\n\n{built_message}"},
            ]

            messages = [{"role": "user", "content": message_content}]
            api_kwargs = dict(
                model=self.config.SONNET_MODEL,
                max_tokens=self.config.CLAUDE_MAX_TOKENS,
                system=system_prompt,
                tools=self._get_claude_tools(),
            )

            messages, response = await self._run_tool_loop(
                messages, api_kwargs, chat_id
            )

            full_text = ""
            for block in response.content:
                if block.type == "text":
                    full_text += block.text

            final = await self._finalize(chat_id, user_message, full_text)
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
        "web_fetch": "fetching a page",
        "share_file": "publishing a document",
    }

    async def stream_response(
        self, chat_id: str, user_message: str, on_status=None
    ) -> AsyncGenerator[str, None]:
        """Stream text deltas. Calls on_status(label) during tool turns for UX feedback."""

        async def on_tool(name):
            label = self.TOOL_LABELS.get(name, name)
            if on_status:
                await on_status(label)

        try:
            if on_status:
                await on_status("thinking")

            system_prompt, built_message, recent_messages = await self._prepare_context(
                chat_id, user_message
            )
            messages = [{"role": "user", "content": built_message}]
            api_kwargs = dict(
                model=self.config.SONNET_MODEL,
                max_tokens=self.config.CLAUDE_MAX_TOKENS,
                system=system_prompt,
                tools=self._get_claude_tools(),
            )

            # Stream directly if no tool use needed, otherwise run tool loop first
            self._mark_cache_breakpoint(messages)
            first_response = await self.client.messages.create(
                messages=messages, **api_kwargs
            )
            self._log_usage(first_response, "stream_first")

            used_tools = first_response.stop_reason in ("tool_use", "pause_turn")
            if used_tools:
                # Process tool turns non-streaming
                messages.append(
                    {"role": "assistant", "content": first_response.content}
                )
                tool_results_turn = []
                for block in first_response.content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        if on_tool:
                            await on_tool(block.name)
                        result = await self._handle_tool_call(block, chat_id)
                        tool_results_turn.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result["content"],
                            }
                        )
                        logger.info(f"tool: {block.name} success={result['success']}")
                if tool_results_turn:
                    messages.append({"role": "user", "content": tool_results_turn})
                # Continue tool loop if more tools needed (or resume a paused turn)
                messages, final_response = await self._run_tool_loop(
                    messages, api_kwargs, chat_id, on_tool=on_tool
                )
            else:
                final_response = first_response

            # Use the final response we already have (a second streamed call
            # would regenerate the same answer and double cost + latency)
            full_response = ""
            for block in final_response.content:
                if block.type == "text":
                    full_response += block.text
            yield full_response

            # Companion may append extra text
            final = await self._finalize(chat_id, user_message, full_response)
            if final != full_response:
                extra = final[len(full_response) :]
                if extra:
                    yield extra

        except Exception as e:
            logger.error(f"stream error for {chat_id}: {e}")
            yield f"Sorry, I encountered an error: {e}"

    # Backward compat for CLI
    async def stream_chat(
        self, message: str, chat_id: int, context: str = ""
    ) -> AsyncGenerator[str, None]:
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
            full_message = message
            if context:
                full_message = f"{context}\n\n{message}"
            response, _ = await self.process_message_with_image(
                str(chat_id), full_message, image_bytes
            )
            yield response
        except Exception as e:
            logger.error(f"stream_chat_with_image error: {e}")
            yield f"Sorry, error analyzing the image: {e}"

    # === Prompt building ===

    def _build_system_prompt(self) -> list:
        """Return system prompt as cacheable content blocks (no timestamp, that goes in user message)."""
        tools = (
            "- Reminders: schedule_message, confirm briefly\n"
            "- Memory: update_brain_file for lasting insights only\n"
            "- Facts: query_entities for quick structured lookups\n"
            "- Search: web_search, cite sources briefly. For research requests, "
            "run several searches from different angles before answering, and "
            "cross-check claims across sources\n"
            "- Fetch: web_fetch for specific URLs\n"
            "- Share: share_file publishes a document to a public link. Use it "
            "for research reports and anything too long for chat: write a "
            "clean standalone .html, then send the link with a 2-3 line "
            "summary in the message\n"
            "- Vision/OCR: read attached images directly and transcribe visible text"
        )
        if self.notion:
            tools += "\n- Notion: search, read, create, and append to the user's Notion workspace"
        return [
            {
                "type": "text",
                "text": (
                    "You are a personal assistant and coach. Direct, grounded, "
                    "occasionally funny. Not a hype man.\n\n"
                    "FORMAT (Telegram, mobile):\n"
                    "- 1-3 short paragraphs. Skip preambles.\n"
                    "- Bullets only for genuine lists. Bold at most one phrase "
                    "per message. No emoji sign-offs.\n\n"
                    "ENDINGS:\n"
                    "- End when the answer is complete. A statement is a valid "
                    "ending; most messages should not end with a question.\n"
                    "- Ask a question only if you need information to act, or "
                    "the user asked to be held accountable on something specific. "
                    "At most one question per message, and it must reference "
                    "something concrete from this conversation.\n"
                    "- Never use generic check-ins ('How are things going?', "
                    "'What would make today feel successful?', 'What's the move "
                    "tomorrow?').\n"
                    "- When the user shares something, acknowledging it well is "
                    "a complete response. Don't turn every message into a "
                    "coaching moment.\n\n"
                    "TONE:\n"
                    "- No hype interjections ('Real talk:', 'that's elite', "
                    "'monster', 'locked in'). Say the substance plainly.\n"
                    "- Disagree when the user is wrong; don't open with "
                    "'Fair point—you're right' reflexively.\n\n"
                    "Core capabilities: Productivity, Health, Relationships, Finance, Goals.\n\n"
                    f"Tools:\n{tools}"
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _build_adaptive_message(
        self,
        user_message: str,
        user_context: str,
        recent_messages: List[Message],
        structured_memory: List = None,
    ) -> str:
        budget = self.config.CONTEXT_TOKEN_BUDGET
        layers = []
        used = 0

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        layers.append(f"Current time: {current_time}")
        used += 10

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

        if structured_memory:
            memory_text = self._format_structured_memory_context(structured_memory)
            memory_tokens = estimate_tokens(memory_text)
            if used + memory_tokens < budget * 0.75:
                layers.append(memory_text)
                used += memory_tokens

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

    def _format_structured_memory_context(self, memories: List) -> str:
        lines = ["[Structured Memory]"]
        for memory in memories:
            observed_at = memory.created_at.strftime("%Y-%m-%d")
            prefix = f"[{memory.confidence}] Previously observed on {observed_at}:"
            if memory.confidence == "inferred":
                prefix = f"[inferred] Possible pattern from {observed_at}:"
            elif memory.confidence == "stale":
                prefix = f"[stale] Older observation from {observed_at}:"
            lines.append(f"- {prefix} {memory.content}")
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
                try:
                    await self.storage.store_memory_item(
                        chat_id,
                        content,
                        layer="semantic",
                        tags=["brain"],
                        confidence="verified",
                        half_life_days=30.0,
                    )
                except Exception as e:
                    logger.warning(f"Structured memory write failed: {e}")
                logger.info(f"brain updated for {chat_id}: {reason}")
                asyncio.create_task(self._extract_entities(chat_id, content))
                return {
                    "tool_name": tool_name,
                    "content": f"Brain updated: {reason}",
                    "success": True,
                }

            elif tool_name == "read_brain_file":
                content = await self.storage.read_user_context(chat_id)
                return {
                    "tool_name": tool_name,
                    "content": content or "No personal context.",
                    "success": True,
                }

            elif tool_name == "query_entities":
                query = tool_input.get("query", "")
                entities = await self.storage.search_entities(chat_id, query)
                if entities:
                    result = "\n".join(
                        f"- {e['subject']} {e['predicate']} {e['object']}"
                        for e in entities
                    )
                else:
                    result = "No matching facts found."
                return {"tool_name": tool_name, "content": result, "success": True}

            elif tool_name == "schedule_message":
                message = tool_input.get("message", "")
                when_text = tool_input.get("when", "")
                scheduled_time = self._parse_when(when_text)
                if not scheduled_time:
                    return {
                        "tool_name": tool_name,
                        "content": f"Could not parse time: {when_text}",
                        "success": False,
                    }
                success = await schedule_reminder_task(chat_id, message, scheduled_time)
                time_str = scheduled_time.strftime("%I:%M %p on %b %d")
                return {
                    "tool_name": tool_name,
                    "content": f"Reminder scheduled for {time_str}"
                    if success
                    else "Failed to schedule",
                    "success": success,
                }

            elif tool_name == "web_fetch":
                url = tool_input.get("url", "")
                content = await self._fetch_url(url)
                return {"tool_name": tool_name, "content": content, "success": True}

            elif tool_name == "share_file":
                filename = tool_input.get("filename", "")
                content = tool_input.get("content", "")
                try:
                    url = self._share_file(filename, content)
                except ValueError as exc:
                    return {
                        "tool_name": tool_name,
                        "content": str(exc),
                        "success": False,
                    }
                logger.info(f"shared file for {chat_id}: {url}")
                return {
                    "tool_name": tool_name,
                    "content": f"Published: {url}",
                    "success": True,
                }

            elif tool_name == "notion_search":
                query = tool_input.get("query", "")
                results = await self.notion.search(query, limit=5)
                if results:
                    lines = []
                    for r in results:
                        lines.append(
                            f"- [{r['title']}] id={r['id']} ({r['last_edited'][:10]})"
                        )
                    return {
                        "tool_name": tool_name,
                        "content": "\n".join(lines),
                        "success": True,
                    }
                return {
                    "tool_name": tool_name,
                    "content": "No results found.",
                    "success": True,
                }

            elif tool_name == "notion_read":
                page_id = tool_input.get("page_id", "")
                page = await self.notion.read_page(page_id)
                # Truncate if too long for tool result
                content = page["content"]
                if len(content) > 4000:
                    content = content[:4000] + "\n\n... (truncated)"
                return {
                    "tool_name": tool_name,
                    "content": f"# {page['title']}\n\n{content}",
                    "success": True,
                }

            elif tool_name == "notion_create":
                title = tool_input.get("title", "")
                content = tool_input.get("content", "")
                parent = tool_input.get("parent_page_id")
                result = await self.notion.create_page(title, content, parent)
                return {
                    "tool_name": tool_name,
                    "content": f"Created: {result['title']} ({result['url']})",
                    "success": True,
                }

            elif tool_name == "notion_append":
                page_id = tool_input.get("page_id", "")
                content = tool_input.get("content", "")
                await self.notion.append_to_page(page_id, content)
                return {
                    "tool_name": tool_name,
                    "content": "Content appended.",
                    "success": True,
                }

            else:
                return {
                    "tool_name": tool_name,
                    "content": f"Unknown tool: {tool_name}",
                    "success": False,
                }

        except Exception as e:
            logger.error(f"tool {tool_name} failed: {e}")
            return {
                "tool_name": tool_name,
                "content": "Action failed. Please try again.",
                "success": False,
            }

    SHARE_EXTENSIONS = {".md", ".html", ".txt", ".csv", ".json"}

    def _share_file(self, filename: str, content: str) -> str:
        """Write content to the public share dir, return its URL."""
        name = Path(filename).name
        stem, ext = name.rsplit(".", 1) if "." in name else (name, "")
        ext = f".{ext.lower()}"
        if ext not in self.SHARE_EXTENSIONS:
            raise ValueError(
                f"Extension '{ext}' not allowed. Use one of: "
                + ", ".join(sorted(self.SHARE_EXTENSIONS))
            )
        stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-") or "file"
        slug = f"{stem}-{secrets.token_hex(3)}{ext}"
        share_dir = self.config.SHARE_DIR
        share_dir.mkdir(parents=True, exist_ok=True)
        (share_dir / slug).write_text(content, encoding="utf-8")
        return f"{self.config.SHARE_URL_BASE}/{slug}"

    async def _fetch_url(self, url: str) -> str:
        current_url = url.strip()
        parsed = urlparse(current_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Invalid URL. Only http:// and https:// links are supported."
        if await self._is_blocked_fetch_host(parsed.hostname):
            return "Blocked private or local URL."

        async with httpx.AsyncClient(follow_redirects=False, timeout=12.0) as client:
            response = None
            for _ in range(5):
                response = await client.get(
                    current_url,
                    headers={"User-Agent": "NosyAgent/1.0 (+https://nosyagent.com)"},
                )
                if response.status_code not in {301, 302, 303, 307, 308}:
                    break
                location = response.headers.get("location")
                if not location:
                    break
                next_url = urljoin(str(response.url), location)
                parsed_next = urlparse(next_url)
                if (
                    parsed_next.scheme not in {"http", "https"}
                    or not parsed_next.netloc
                ):
                    return "Blocked redirect to an unsupported URL."
                if await self._is_blocked_fetch_host(parsed_next.hostname):
                    return "Blocked redirect to a private or local URL."
                current_url = next_url
            else:
                return "Too many redirects."

        if response is None:
            return "Fetch failed."
        content_type = response.headers.get("content-type", "unknown")
        text = response.text
        if "html" in content_type:
            text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 6000:
            text = text[:6000] + "\n\n... (truncated)"
        return (
            f"URL: {response.url}\n"
            f"Status: {response.status_code}\n"
            f"Content-Type: {content_type}\n\n"
            f"{text}"
        )

    async def _is_blocked_fetch_host(self, hostname: Optional[str]) -> bool:
        if not hostname:
            return True

        host = hostname.rstrip(".").lower()
        if host == "localhost" or host.endswith(".localhost"):
            return True

        def is_blocked_ip(ip_text: str) -> bool:
            try:
                ip = ipaddress.ip_address(ip_text)
            except ValueError:
                return False
            return (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            )

        if is_blocked_ip(host):
            return True

        try:
            infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
        except socket.gaierror:
            return False

        return any(is_blocked_ip(info[4][0]) for info in infos)

    async def _extract_entities(self, chat_id: str, brain_content: str):
        if not brain_content.strip():
            return
        try:
            response = await self.client.messages.create(
                model=self.config.HAIKU_MODEL,
                max_tokens=1500,
                messages=[
                    {
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
                    }
                ],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            triples = json.loads(text)
            if isinstance(triples, list):
                await self.storage.replace_entities(chat_id, triples)
                for triple in triples:
                    subject = triple.get("subject", "").strip()
                    predicate = triple.get("predicate", "").strip()
                    obj = triple.get("object", "").strip()
                    if subject and predicate and obj:
                        await self.storage.store_memory_item(
                            chat_id,
                            f"{subject} {predicate} {obj}",
                            layer="semantic",
                            tags=["entity", predicate.lower()],
                            confidence="verified",
                            half_life_days=30.0,
                        )
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
