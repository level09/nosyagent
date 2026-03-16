#!/bin/bash
# Local smoke test: sends a real message through the agent and prints the result.
# Requires ANTHROPIC_API_KEY in .env
set -e

cd "$(dirname "$0")/.."

echo "=== smoke test ==="

uv run python -c "
import asyncio, os
from dotenv import load_dotenv
load_dotenv()

from config import Config
from storage import Storage
from agent import AIAgent

async def test():
    config = Config()
    storage = Storage(config.DB_PATH)
    agent = AIAgent(config, storage)

    # Test 1: non-streaming
    print('--- process_message ---')
    resp, _ = await agent.process_message('smoke_test', 'What time is it?')
    print(f'Response: {resp[:200]}')
    print()

    # Test 2: streaming
    print('--- stream_response ---')
    chunks = []
    async for chunk in agent.stream_response('smoke_test', 'Say hello in 3 words'):
        chunks.append(chunk)
    full = ''.join(chunks)
    print(f'Streamed ({len(chunks)} chunks): {full[:200]}')
    print()

    # Test 3: entity query
    print('--- query_entities ---')
    entities = await storage.search_entities('smoke_test', 'test')
    print(f'Entities: {len(entities)}')
    print()

    print('All smoke tests passed.')

asyncio.run(test())
"
