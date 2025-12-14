import asyncio
import logging
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, AsyncGenerator, Any, Dict
import re
import dateparser

import anthropic
from storage import Storage, Message
from config import Config
from reminder_scheduler import schedule_reminder_task
from companion import CompanionService

# Optional semantic memory - gracefully degrade if not available
try:
    from semantic_memory import SemanticMemory
    SEMANTIC_MEMORY_AVAILABLE = True
except ImportError:
    SEMANTIC_MEMORY_AVAILABLE = False
    SemanticMemory = None

logger = logging.getLogger(__name__)

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

        # Initialize semantic memory if available and path provided
        self.semantic_memory = None
        if SEMANTIC_MEMORY_AVAILABLE and semantic_memory_path:
            try:
                self.semantic_memory = SemanticMemory(semantic_memory_path)
                logger.info(f"Semantic memory enabled at {semantic_memory_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic memory: {e}")
    
    def _get_claude_tools(self) -> List[Dict[str, Any]]:
        """Get Claude API tool definitions"""
        return [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5
            },
            {
                "name": "update_brain_file",
                "description": "Update the user's personal brain file. Use this for DURABLE, LONG-TERM facts or preferences only. Do not use for temporary context.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The new content to write (markdown). Be concise."
                        },
                        "reason": {
                            "type": "string", 
                            "description": "Why this is worth remembering long-term."
                        }
                    },
                    "required": ["content", "reason"]
                }
            },
            {
                "name": "read_brain_file",
                "description": "Read the user's brain file.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "schedule_message",
                "description": "Schedule a reminder/message. You understand natural language time (e.g., 'tomorrow at 9am', 'in 2 hours').",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send."
                        },
                        "when": {
                            "type": "string", 
                            "description": "Time expression (e.g. 'in 5 mins', 'friday at noon')."
                        }
                    },
                    "required": ["message", "when"]
                }
            }
        ]
    
    async def process_message(self, chat_id: str, user_message: str) -> tuple[str, Optional[dict]]:
        """
        Process user message and return (response, reminder_data)
        reminder_data is dict with 'message' and 'when' keys if reminder requested
        """
        try:
            # Get conversation context
            recent_messages = await self.storage.get_recent_conversations(chat_id)
            user_context = await self.storage.read_user_context(chat_id)

            # Semantic search for relevant past context
            semantic_context = []
            if self.semantic_memory:
                try:
                    semantic_context = self.semantic_memory.search(user_message, chat_id, limit=3)
                except Exception as e:
                    logger.warning(f"Semantic search failed: {e}")

            # Build layered prompts
            system_prompt = self._build_system_prompt()
            user_message_with_context = self._build_layered_user_message(
                user_message, user_context, recent_messages, semantic_context
            )
            
            # Call Claude API with tools
            response, tool_results = await self._call_claude_with_tools(
                system_prompt, user_message_with_context, chat_id
            )

            final_response = response
            if self.companion:
                try:
                    final_response = await self.companion.wrap_response(
                        chat_id, user_message, response, recent_messages
                    )
                except Exception as exc:
                    logger.warning(f"Companion wrapper failed for {chat_id}: {exc}")
                    final_response = response

            # Store conversation
            await self.storage.store_conversation(chat_id, user_message, final_response)

            return final_response, None
            
        except Exception as e:
            logger.error(f"❌ Error processing message for {chat_id}: {e}")
            return "I'm having trouble right now. Could you try asking again?", None
    
    async def process_message_with_image(self, chat_id: str, user_message: str, image_bytes: bytes) -> tuple[str, Optional[dict]]:
        """
        Process user message with image and return (response, reminder_data)
        """
        try:
            # Get conversation context
            recent_messages = await self.storage.get_recent_conversations(chat_id)
            user_context = await self.storage.read_user_context(chat_id)

            # Semantic search for relevant past context
            semantic_context = []
            if self.semantic_memory:
                try:
                    semantic_context = self.semantic_memory.search(user_message, chat_id, limit=3)
                except Exception as e:
                    logger.warning(f"Semantic search failed: {e}")

            # Build layered prompts
            system_prompt = self._build_system_prompt()
            user_message_with_context = self._build_layered_user_message(
                user_message, user_context, recent_messages, semantic_context
            )

            # Call Claude API with image
            response, tool_results = await self._call_claude_with_image(
                system_prompt, user_message_with_context, image_bytes, chat_id
            )

            final_response = response
            if self.companion:
                try:
                    final_response = await self.companion.wrap_response(
                        chat_id, user_message, response, recent_messages
                    )
                except Exception as exc:
                    logger.warning(f"Companion wrapper failed (image) for {chat_id}: {exc}")
                    final_response = response

            # Store conversation
            await self.storage.store_conversation(chat_id, user_message, final_response)

            return final_response, None
            
        except Exception as e:
            logger.error(f"❌ Error processing image for {chat_id}: {e}")
            return "I had trouble processing your image. Could you try sending it again?", None
    
    def _build_system_prompt(self) -> str:
        """Build core system prompt - Layer 1: Identity and behavior"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""You are a proactive life optimization AI assistant. Current time: {current_time}

FORMAT (Telegram):
- Keep responses readable on mobile - aim for 1-3 short paragraphs.
- Skip preambles like "Great question!" or "Here's what I think..."
- Use bullet points when listing multiple items.
- Be conversational but efficient.

Core capabilities: Productivity, Health, Relationships, Finance, Goals.

Tools:
- Reminders: schedule_message → confirm briefly "✓ Scheduled for [time]"
- Memory: update_brain_file for lasting insights only
- Search: cite sources briefly

Be helpful and warm, but don't over-explain. Get to the point, then stop."""
    
    def _build_layered_user_message(
        self, user_message: str, user_context: str, recent_messages: List[Message], semantic_context: List = None
    ) -> str:
        """Build layered user message: Layer 2 (brain) + Layer 3 (conversation) + semantic + current message"""

        layers = []

        # Layer 2: Personal context (brain file)
        if user_context:
            layers.append(f"[Personal Context]\n{user_context}")
        else:
            layers.append(
                "[Personal Context]\nNo personal context yet — share your current goals, "
                "constraints, or routines so I can tailor the next steps."
            )

        # Layer 3: Recent conversation context
        if recent_messages:
            clean_messages = [
                msg for msg in recent_messages[-3:]
                if "REMINDER:" not in msg.agent_response and "🔔" not in msg.agent_response
            ]

            if clean_messages:
                context_layer = "[Recent Conversation]\n"
                for msg in clean_messages[-2:]:  # Max 2 recent exchanges
                    context_layer += f"User: {msg.user_message}\nAssistant: {msg.agent_response}\n\n"
                layers.append(context_layer.strip())

        # Layer 4: Semantic memory - relevant past context
        if semantic_context:
            semantic_layer = "[Relevant Past Context]\n"
            for chunk in semantic_context:
                semantic_layer += f"- {chunk.content}\n"
            layers.append(semantic_layer.strip())

        # Combine layers with current user message
        full_message = "\n\n".join(layers)
        if full_message:
            return f"{full_message}\n\n[Current Message]\n{user_message}"
        else:
            return user_message
    
    async def _handle_tool_call(self, tool_call, chat_id: str) -> Dict[str, Any]:
        """Handle individual tool calls"""
        tool_name = tool_call.name
        tool_input = tool_call.input
        
        try:
            if tool_name == "update_brain_file":
                content = tool_input.get("content", "")
                reason = tool_input.get("reason", "")

                await self.storage.update_user_context(chat_id, content, reason)
                logger.info(f"✅ Updated brain file for {chat_id}: {reason}")

                # Re-index brain content for semantic search
                if self.semantic_memory:
                    try:
                        self.semantic_memory.reindex_brain(chat_id, content)
                    except Exception as e:
                        logger.warning(f"Failed to reindex brain: {e}")

                return {
                    "tool_name": tool_name,
                    "content": f"Brain file updated: {reason}",
                    "success": True
                }
            
            elif tool_name == "read_brain_file":
                content = await self.storage.read_user_context(chat_id)
                return {
                    "tool_name": tool_name,
                    "content": content or "No personal context available.",
                    "success": True
                }
            
            elif tool_name == "schedule_message":
                message = tool_input.get("message", "")
                when_text = tool_input.get("when", "")
                
                # Parse the "when" into a datetime
                scheduled_time = self._parse_when(when_text)
                if not scheduled_time:
                    return {
                        "tool_name": tool_name,
                        "content": f"Could not parse time: {when_text}",
                        "success": False
                    }
                
                # Schedule the message
                from reminder_scheduler import schedule_reminder_task
                success = await schedule_reminder_task(chat_id, message, scheduled_time)
                
                return {
                    "tool_name": tool_name,
                    "content": f"Reminder scheduled for {scheduled_time.strftime('%I:%M %p on %b %d')}" if success else "Failed to schedule reminder",
                    "success": success
                }
            
            else:
                return {
                    "tool_name": tool_name,
                    "content": f"Unknown tool: {tool_name}",
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"❌ Tool {tool_name} failed: {e}")
            return {
                "tool_name": tool_name,
                "content": f"I couldn't complete that action. Please try again.",
                "success": False
            }
    
    async def _call_claude_with_tools(self, system_prompt: str, user_message: str, chat_id: str) -> tuple[str, List[Dict[str, Any]]]:
        """Call Claude API with tools and handle tool calls"""
        max_retries = self.config.CLAUDE_MAX_RETRIES
        base_delay = self.config.CLAUDE_BASE_DELAY
        tool_results = []
        
        for attempt in range(max_retries):
            try:
                response = await self.client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=self.config.CLAUDE_MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                    tools=self._get_claude_tools()
                )
                
                # Handle tool calls
                messages = [{"role": "user", "content": user_message}]
                
                while response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})
                    
                    tool_results_for_this_turn = []
                    for content_block in response.content:
                        if hasattr(content_block, 'type') and content_block.type == "tool_use":
                            tool_result = await self._handle_tool_call(content_block, chat_id)
                            tool_results.append(tool_result)
                            tool_results_for_this_turn.append({
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": tool_result["content"]
                            })
                    
                    messages.append({"role": "user", "content": tool_results_for_this_turn})
                    
                    # Continue conversation with tool results
                    response = await self.client.messages.create(
                        model="claude-sonnet-4-5",
                        max_tokens=self.config.CLAUDE_MAX_TOKENS,
                        system=system_prompt,
                        messages=messages,
                        tools=self._get_claude_tools()
                    )
                
                # Extract final text response
                full_text = ""
                logger.debug(f"Processing {len(response.content)} content blocks, stop_reason: {response.stop_reason}")
                
                for i, content_block in enumerate(response.content):
                    logger.debug(f"Block {i}: type={content_block.type}")
                    if content_block.type == 'text':
                        full_text += content_block.text
                        logger.debug(f"Added text block: {len(content_block.text)} chars")
                
                logger.debug(f"Final response length: {len(full_text)}")
                
                # Handle special stop_reason cases
                if response.stop_reason == "max_tokens":
                    logger.warning("Claude hit max_tokens limit - response may be incomplete")
                    if full_text.strip():
                        # Add indication of truncation
                        full_text += "... [continued]"
                
                # Handle empty responses based on stop_reason and context  
                if not full_text.strip():
                    if response.stop_reason == "end_turn":
                        if tool_results:
                            # Claude naturally completed after tool use - this is normal
                            logger.debug("Claude completed tool use without additional text (normal behavior)")
                            return "", tool_results  # Return empty string, let caller handle
                        else:
                            logger.warning("Claude returned empty response with end_turn but no tools used")
                    elif response.stop_reason == "max_tokens":
                        logger.warning("Claude hit max_tokens limit with empty response")
                        return "I need to continue this response...", tool_results
                    else:
                        logger.warning(f"Empty response with stop_reason: {response.stop_reason}")
                
                return full_text, tool_results
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Claude API call failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
        
        return "", tool_results
    
    async def _call_claude_with_image(self, system_prompt: str, user_message: str, image_bytes: bytes, chat_id: str) -> tuple[str, List[Dict[str, Any]]]:
        """Call Claude API with image and handle tool calls"""
        max_retries = self.config.CLAUDE_MAX_RETRIES
        base_delay = self.config.CLAUDE_BASE_DELAY
        tool_results = []
        
        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Detect image format (basic detection)
        image_format = "image/jpeg"  # default
        if image_bytes.startswith(b'\x89PNG'):
            image_format = "image/png"
        elif image_bytes.startswith(b'GIF'):
            image_format = "image/gif"
        elif image_bytes.startswith(b'\xff\xd8\xff'):
            image_format = "image/jpeg"
        elif image_bytes.startswith(b'WEBP', 8):
            image_format = "image/webp"
        
        logger.debug(f"Detected image format: {image_format}, size: {len(image_bytes)} bytes")
        
        for attempt in range(max_retries):
            try:
                # Create message with image
                message_content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_format,
                            "data": image_base64
                        }
                    }
                ]
                
                # Add text if provided
                if user_message.strip():
                    message_content.append({
                        "type": "text",
                        "text": user_message
                    })
                
                response = await self.client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=self.config.CLAUDE_MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": message_content}],
                    tools=self._get_claude_tools()
                )
                
                # Handle tool calls
                messages = [{"role": "user", "content": message_content}]
                
                while response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})
                    
                    tool_results_for_this_turn = []
                    for content_block in response.content:
                        if hasattr(content_block, 'type') and content_block.type == "tool_use":
                            tool_result = await self._handle_tool_call(content_block, chat_id)
                            tool_results.append(tool_result)
                            tool_results_for_this_turn.append({
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": tool_result["content"]
                            })
                    
                    messages.append({"role": "user", "content": tool_results_for_this_turn})
                    
                    # Continue conversation with tool results
                    response = await self.client.messages.create(
                        model="claude-sonnet-4-5",
                        max_tokens=self.config.CLAUDE_MAX_TOKENS,
                        system=system_prompt,
                        messages=messages,
                        tools=self._get_claude_tools()
                    )
                
                # Extract final text response
                full_text = ""
                logger.debug(f"Processing {len(response.content)} content blocks for image, stop_reason: {response.stop_reason}")
                
                for i, content_block in enumerate(response.content):
                    logger.debug(f"Image Block {i}: type={content_block.type}")
                    if content_block.type == 'text':
                        full_text += content_block.text
                        logger.debug(f"Added image text block: {len(content_block.text)} chars")
                
                logger.debug(f"Final image response length: {len(full_text)}")
                
                # Handle special stop_reason cases
                if response.stop_reason == "max_tokens":
                    logger.warning("Claude hit max_tokens limit for image - response may be incomplete")
                    if full_text.strip():
                        # Add indication of truncation
                        full_text += "... [continued]"
                
                # Handle empty responses based on stop_reason and context
                if not full_text.strip():
                    if response.stop_reason == "end_turn":
                        if tool_results:
                            # Claude naturally completed after tool use - this is normal
                            logger.debug("Claude completed image tool use without additional text (normal behavior)")
                            return "", tool_results  # Return empty string, let caller handle
                        else:
                            logger.warning("Claude returned empty response with end_turn but no tools used for image")
                    elif response.stop_reason == "max_tokens":
                        logger.warning("Claude hit max_tokens limit for image with empty response")
                        return "I need to continue analyzing this image...", tool_results
                    else:
                        logger.warning(f"Empty image response with stop_reason: {response.stop_reason}")
                
                return full_text, tool_results
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Claude API call with image failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
        
        return "", tool_results
    
    
    def _parse_when(self, when_text: str) -> Optional[datetime]:
        """Parse natural language time into datetime using dateparser"""
        # Settings for dateparser to prefer future dates
        settings = {
            'PREFER_DATES_FROM': 'future',
            'PREFER_DAY_OF_MONTH': 'first',
            'RETURN_AS_TIMEZONE_AWARE': False  # We use naive datetimes in DB for now
        }
        
        # Clean up input
        cleaned_text = when_text.lower().strip()
        if cleaned_text.startswith("in "):
            # "in 5 mins" is handled well by dateparser, but let's ensure it's clean
            pass
            
        dt = dateparser.parse(cleaned_text, settings=settings)
        
        # If dateparser fails or returns past time (and user didn't specify "past"), try to fix
        if dt and dt < datetime.now() and "ago" not in cleaned_text:
            # If it's a time like "at 9am" and it's currently 10am, dateparser might give today 9am (past).
            # We want tomorrow 9am.
            dt = dt + timedelta(days=1)
            
        return dt
    
    async def stream_chat(self, message: str, chat_id: int, context: str = "") -> AsyncGenerator[str, None]:
        """Stream chat method for compatibility with nosy_bot.py"""
        try:
            # Phase 1: Show thinking indicator (expected by nosy_bot.py)
            yield "🤔 Thinking..."
            
            # Phase 2: Get complete response and yield it
            response, _ = await self.process_message(str(chat_id), message)
            
            # Handle empty responses based on context
            if not response or not response.strip():
                # Check if tools were used (common tool use case)
                logger.warning(f"Empty response from process_message for chat {chat_id}")
                response = "✓ Done"  # Simple acknowledgment
            
            yield response
        except Exception as e:
            logger.error(f"Error in stream_chat: {e}")
            yield "🤔 Thinking..."
            yield f"Sorry, I encountered an error: {e}"
    
    async def stream_chat_with_image(self, message: str, chat_id: int, image_bytes: bytes, context: str = "") -> AsyncGenerator[str, None]:
        """Stream chat method with image support for compatibility with nosy_bot.py"""
        try:
            # Phase 1: Show thinking indicator (expected by nosy_bot.py)
            yield "🤔 Analyzing image..."
            
            # Phase 2: Get complete response with image and yield it
            response, _ = await self.process_message_with_image(str(chat_id), message, image_bytes)
            
            # Handle empty responses based on context
            if not response or not response.strip():
                # Check if tools were used (common tool use case)
                logger.warning(f"Empty response from process_message_with_image for chat {chat_id}")
                response = "✓ Done"  # Simple acknowledgment
            
            yield response
        except Exception as e:
            logger.error(f"Error in stream_chat_with_image: {e}")
            yield "🤔 Analyzing image..."
            yield f"Sorry, I encountered an error analyzing the image: {e}"
