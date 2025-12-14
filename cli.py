#!/usr/bin/env python3
"""
Simple CLI for NosyAgent - Local Mac Interaction

Usage:
    python cli.py                    # Interactive chat mode
    python cli.py "Your message"     # Single message mode
"""

import sys
import asyncio
import os
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent import AIAgent
from storage import Storage
from config import get_config
from companion import CompanionService

# Import test MCP agent for testing
try:
    from test_mcp_agent import TestMCPAgent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# CLI-specific chat ID for local usage
LOCAL_CHAT_ID = "cli_local"

class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(text, color):
    """Print colored text to terminal"""
    print(f"{color}{text}{Colors.END}")

async def single_message(agent: AIAgent, message: str):
    """Process a single message and return response"""
    print_colored(f"🤖 Processing: {message}", Colors.BLUE)
    
    try:
        response, reminder_data = await agent.process_message(LOCAL_CHAT_ID, message)
        
        print_colored("✨ Response:", Colors.GREEN)
        print(response)
        
        if reminder_data:
            print_colored(f"⏰ Reminder set: {reminder_data['message']} at {reminder_data['when']}", Colors.YELLOW)
            
    except Exception as e:
        print_colored(f"❌ Error: {e}", Colors.RED)

async def interactive_chat(agent: AIAgent, companion_service: Optional[CompanionService] = None):
    """Start interactive chat session"""
    agent_type = "MCP-Enhanced" if isinstance(agent, TestMCPAgent) else "Standard"
    print_colored(f"🚀 NosyAgent CLI - Interactive Mode ({agent_type})", Colors.BOLD)
    print_colored("Type 'exit', 'quit', or 'bye' to end the session", Colors.BLUE)
    print_colored("Type 'clear' to clear your conversation history", Colors.BLUE)
    if isinstance(agent, TestMCPAgent):
        print_colored("Type 'memory' to check MCP knowledge graph status", Colors.BLUE)
    print("")
    
    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.YELLOW}You: {Colors.END}").strip()
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print_colored("👋 Goodbye!", Colors.GREEN)
                break
                
            if user_input.lower() == 'clear':
                # Clear conversation history by using a new chat ID
                global LOCAL_CHAT_ID
                LOCAL_CHAT_ID = f"cli_local_{asyncio.get_event_loop().time()}"
                print_colored("🧹 Conversation history cleared", Colors.GREEN)
                continue

            if user_input.startswith('/mode') and not companion_service:
                print_colored("Companion mode is disabled in this environment", Colors.YELLOW)
                continue

            if user_input.startswith('/mode') and companion_service:
                parts = user_input.split()
                if len(parts) == 1:
                    settings = await companion_service.storage.get_user_settings(LOCAL_CHAT_ID)
                    print_colored(
                        f"ℹ️ Companion mode is {settings.companion_level} (use /mode off|light|standard)",
                        Colors.BLUE,
                    )
                else:
                    try:
                        settings = await companion_service.set_companion_level(LOCAL_CHAT_ID, parts[1])
                    except ValueError:
                        print_colored("Usage: /mode off|light|standard", Colors.YELLOW)
                        continue
                    print_colored(f"✅ Companion mode set to {settings.companion_level}", Colors.GREEN)
                    if settings.companion_level != 'off':
                        scheduled = await companion_service.schedule_next_nudge(LOCAL_CHAT_ID)
                        if scheduled:
                            print_colored(
                                f"📬 Next spark queued for {scheduled.strftime('%a %H:%M UTC')}",
                                Colors.BLUE,
                            )
                continue

            if user_input.startswith('/quiet') and not companion_service:
                print_colored("Companion mode is disabled in this environment", Colors.YELLOW)
                continue

            if user_input.startswith('/quiet') and companion_service:
                parts = user_input.split()
                if len(parts) != 3:
                    print_colored("Usage: /quiet HH:MM HH:MM", Colors.YELLOW)
                    continue
                try:
                    settings = await companion_service.set_quiet_hours(LOCAL_CHAT_ID, parts[1], parts[2])
                except ValueError:
                    print_colored("Usage: /quiet HH:MM HH:MM", Colors.YELLOW)
                    continue
                print_colored(
                    f"😴 Quiet hours set to {settings.quiet_hours_start}–{settings.quiet_hours_end}",
                    Colors.BLUE,
                )
                continue

            if user_input.startswith('/nudge') and not companion_service:
                print_colored("Companion mode is disabled in this environment", Colors.YELLOW)
                continue

            if user_input.startswith('/nudge') and companion_service:
                parts = user_input.split()
                if len(parts) == 1:
                    settings = await companion_service.storage.get_user_settings(LOCAL_CHAT_ID)
                    print_colored(
                        f"ℹ️ Nudges are {settings.nudge_frequency} (off|weekly|standard)",
                        Colors.BLUE,
                    )
                    continue
                choice = parts[1].lower()
                if choice == 'on':
                    choice = 'weekly'
                try:
                    settings = await companion_service.set_nudge_frequency(LOCAL_CHAT_ID, choice)
                except ValueError:
                    print_colored("Usage: /nudge off|weekly|standard", Colors.YELLOW)
                    continue
                print_colored(f"✅ Nudges set to {settings.nudge_frequency}", Colors.GREEN)
                if settings.nudge_frequency != 'off':
                    scheduled = await companion_service.schedule_next_nudge(LOCAL_CHAT_ID)
                    if scheduled:
                        print_colored(
                            f"📬 Next spark queued for {scheduled.strftime('%a %H:%M UTC')}",
                            Colors.BLUE,
                        )
                continue

            if user_input.lower() == 'memory' and isinstance(agent, TestMCPAgent):
                # Check MCP knowledge graph status
                jsonl_path = Path.home() / ".aim" / "memory.jsonl"
                if jsonl_path.exists():
                    with open(jsonl_path, 'r') as f:
                        lines = f.readlines()
                    print_colored(f"🧠 MCP Knowledge Graph: {len(lines)} entries stored", Colors.GREEN)
                    if lines:
                        print_colored("Recent entries:", Colors.YELLOW)
                        for line in lines[-3:]:
                            print(f"  📄 {line.strip()}")
                else:
                    print_colored("🧠 MCP Knowledge Graph: No memory file found yet", Colors.RED)
                continue
                
            if not user_input:
                continue
                
            # Process message
            print_colored("🤔 Thinking...", Colors.BLUE)
            
            response, reminder_data = await agent.process_message(LOCAL_CHAT_ID, user_input)
            
            print_colored("🤖 NosyAgent:", Colors.GREEN)
            print(response)
            
            if reminder_data:
                print_colored(f"⏰ Reminder set: {reminder_data['message']} at {reminder_data['when']}", Colors.YELLOW)
            
            print()  # Add spacing
            
        except KeyboardInterrupt:
            print_colored("\n👋 Goodbye!", Colors.GREEN)
            break
        except EOFError:
            print_colored("\n👋 Goodbye!", Colors.GREEN)
            break
        except Exception as e:
            print_colored(f"❌ Error: {e}", Colors.RED)

async def report_companion(storage: Storage, chat_id: str):
    metrics = await storage.get_recent_companion_metrics(chat_id, limit=20)
    if not metrics:
        print_colored("ℹ️ No companion reflections logged yet", Colors.BLUE)
        return
    print_colored(f"🧾 Last {len(metrics)} reflections:", Colors.GREEN)
    for metric in metrics:
        timestamp = metric.shown_at.strftime('%Y-%m-%d %H:%M')
        template = metric.template_id or "?"
        print(f"  • {timestamp} | template={template} | lines={metric.line_count}")


async def main():
    """Main CLI function"""
    try:
        # Initialize config and validate
        config = get_config()
        
        # Check for required API key
        if not config.ANTHROPIC_API_KEY:
            print_colored("❌ Error: ANTHROPIC_API_KEY not found in environment", Colors.RED)
            print_colored("Please set it in your .env file or environment variables", Colors.YELLOW)
            sys.exit(1)
        
        # Initialize storage and agent
        storage = Storage(config.DB_PATH)
        companion_service = None
        if config.COMPANION_MODE_ENABLED:
            companion_service = CompanionService(
                storage,
                config.COMPANION_CARDS_PATH,
                enabled=config.COMPANION_MODE_ENABLED,
            )

        # Check if MCP testing is requested
        use_mcp = '--mcp' in sys.argv
        if use_mcp:
            sys.argv.remove('--mcp')  # Remove flag from args

        if '--report-companion' in sys.argv:
            sys.argv.remove('--report-companion')
            await report_companion(storage, LOCAL_CHAT_ID)
            return

        if use_mcp and MCP_AVAILABLE:
            print_colored("🧠 Using MCP-Enhanced Agent for testing", Colors.YELLOW)
            agent = TestMCPAgent(config.ANTHROPIC_API_KEY, storage)
        else:
            if use_mcp and not MCP_AVAILABLE:
                print_colored("⚠️ MCP agent not available, using standard agent", Colors.YELLOW)
            semantic_memory_path = config.SEMANTIC_MEMORY_PATH if config.SEMANTIC_MEMORY_ENABLED else None
            agent = AIAgent(config, storage, companion_service, semantic_memory_path=semantic_memory_path)
        
        # Check command line arguments
        if len(sys.argv) > 1:
            # Single message mode
            message = " ".join(sys.argv[1:])
            await single_message(agent, message)
        else:
            # Interactive mode
            await interactive_chat(agent, companion_service)
            
    except Exception as e:
        print_colored(f"❌ Fatal error: {e}", Colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
