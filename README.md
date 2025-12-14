# NosyAgent

**An AI that actually knows you.**

## The Problem

I wanted to make better decisions using AI. But I can't feed ChatGPT context about everything - my health markers, financial goals, what I'm working on, what I said last month. Every conversation starts from zero.

So I built an AI with its own engineered memory. One that improves over time. One that knows my biomarkers, diet goals, finance targets, work context - everything. Like building a second brain, or a superhuman advisor that's always there.

## The Idea

Feed it everything about you. Let it build a persistent "brain" that evolves. Over time, it becomes something like another you - one that remembers everything, connects dots you miss, and helps you make better decisions.

This is a rough concept, but I think it has interesting potential.

## Key Features

### Persistent Brain
Not chat history. A living document about you - health markers, diet goals, finance targets, work context, personal preferences. Updated automatically as you talk. The more you use it, the smarter it gets about *you*.

### Time Awareness
Most AI exists in a timeless bubble. NosyAgent understands temporal context - how much time passed since you last talked, what day it is, when you mentioned something. While traveling? It knows you've been offline for days before suggesting anything.

### Semantic Memory
Ask "what did we discuss about my startup?" and it finds relevant context from weeks ago. Vector search across your conversation history.

### Proactive Reflection
Background thinking about recent conversations. Suggests things you might be missing. Not just reactive - actually reaches out with useful nudges.

### Natural Reminders
"Remind me to check my glucose after dinner" - just works. No syntax, no apps. LLM understands you, delivers on time.

### Telegram Native
This isn't a web app you forget to check. It lives in Telegram - super convenient, always in your pocket. Send it messages, images, voice notes. Chat with your AI like you'd chat with a friend.

## Tech Stack

- Python + Claude API (Anthropic)
- SQLite for storage + automatic brain versioning
- LanceDB for semantic memory
- Telegram webhook bot
- ARQ + Redis for scheduled reminders

## Quick Start

```bash
# Install
git clone https://github.com/level09/nosyagent.git
cd nosyagent
uv sync

# Configure
cp .env.sample .env
# Add your ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, etc.

# Run locally
uv run python cli.py

# Run Telegram bot
uv run uvicorn nosy_bot:app --host 0.0.0.0 --port 8000
uv run arq worker.WorkerSettings  # for reminders
```

## Future Ideas

- Complex file handling (PDFs, documents)
- Voice message processing
- Health data integrations
- Smarter proactive suggestions
- Multi-modal inputs

## Why "Nosy"?

Because a good AI assistant should be nosy. It should remember your sister's birthday is coming up. Notice you've been stressed. Connect dots you're too busy to connect.

Most AI is politely amnesiac. This one pays attention.

---

MIT License. Do whatever you want with it.
