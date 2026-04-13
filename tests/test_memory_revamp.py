import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AIAgent
from cli import handle_memory_command
from router import Router
from storage import Storage

WEAK_STRENGTH = 0.2


class MockConfig:
    ANTHROPIC_API_KEY = "fake_key"
    CLAUDE_MAX_TOKENS = 100
    SONNET_MODEL = "claude-sonnet-4-6"
    HAIKU_MODEL = "claude-haiku-4-5-20251001"
    CONTEXT_TOKEN_BUDGET = 12000
    NOTION_TOKEN = ""


class MockStorage:
    pass


def test_memory_recall_strengthens_item(tmp_path):
    async def run():
        storage = Storage(tmp_path / "memory.db")
        memory_id = await storage.store_memory_item(
            "chat1",
            "User prefers concise Telegram replies",
            layer="semantic",
            tags=["preference"],
            confidence="verified",
        )

        before = (await storage.get_memory_items_by_ids([memory_id]))[0]
        results = await storage.search_memory_items("chat1", "concise replies")
        after = (await storage.get_memory_items_by_ids([memory_id]))[0]

        assert results[0].id == memory_id
        assert after.retrieval_count == before.retrieval_count + 1
        assert after.strength > before.strength
        assert after.half_life_days > before.half_life_days
        assert after.last_retrieved_at is not None

    asyncio.run(run())


def test_memory_sleep_marks_old_weak_memory_stale(tmp_path):
    async def run():
        storage = Storage(tmp_path / "sleep.db")
        memory_id = await storage.store_memory_item(
            "chat1",
            "User was testing an old deployment flow",
            strength=WEAK_STRENGTH,
            half_life_days=7.0,
        )

        old_time = (
            (datetime.utcnow() - timedelta(days=45))
            .replace(microsecond=0)
            .isoformat()
        )
        async with aiosqlite.connect(storage.db_path) as db:
            await db.execute(
                "UPDATE memory_items SET updated_at = ?, created_at = ? WHERE id = ?",
                (old_time, old_time, memory_id),
            )
            await db.commit()

        result = await storage.run_memory_sleep("chat1")
        item = (await storage.get_memory_items_by_ids([memory_id]))[0]

        assert result["staled"] == 1
        assert item.confidence == "stale"
        assert item.strength < WEAK_STRENGTH

    asyncio.run(run())


def test_memory_sleep_promotes_repeated_episodic_memory(tmp_path):
    async def run():
        storage = Storage(tmp_path / "promote.db")
        for _ in range(3):
            await storage.store_memory_item(
                "chat1",
                "User hit cache refresh issues",
                layer="episodic",
                tags=["error"],
            )

        result = await storage.run_memory_sleep("chat1")
        memories = await storage.list_memory_items("chat1", include_stale=True)

        assert result["promotions"] == 1
        assert any(
            memory.layer == "semantic"
            and memory.content == "Repeated pattern: User hit cache refresh issues"
            for memory in memories
        )

    asyncio.run(run())


def test_memory_sleep_detects_obvious_conflict(tmp_path):
    async def run():
        storage = Storage(tmp_path / "conflict.db")
        await storage.store_memory_item(
            "chat1",
            "User lives in Berlin",
            layer="semantic",
        )
        await storage.store_memory_item(
            "chat1",
            "User lives in Warsaw",
            layer="semantic",
        )

        result = await storage.run_memory_sleep("chat1")
        conflicts = await storage.get_memory_conflicts("chat1")

        assert result["conflicts"] == 1
        assert conflicts
        assert "conflicting lives_in" in conflicts[0]["reason"]

    asyncio.run(run())


def test_structured_memory_prompt_framing():
    agent = AIAgent(MockConfig(), MockStorage())
    created = datetime(2026, 4, 13)
    memory = type(
        "Memory",
        (),
        {
            "confidence": "verified",
            "created_at": created,
            "content": "User prefers concise answers",
        },
    )()

    text = agent._format_structured_memory_context([memory])  # noqa: SLF001

    assert "[Structured Memory]" in text
    assert "[verified] Previously observed on 2026-04-13" in text
    assert "User prefers concise answers" in text


def test_cli_memory_command_status(tmp_path, capsys):
    async def run():
        storage = Storage(tmp_path / "command.db")
        await storage.store_memory_item(
            "cli_local",
            "User prefers memory inspection commands",
            layer="semantic",
            confidence="observed",
        )

        await handle_memory_command(storage, "cli_local", "/memory")

    asyncio.run(run())

    captured = capsys.readouterr()
    assert "Memory items: 1" in captured.out
    assert "semantic/observed" in captured.out


def test_router_forces_ocr_and_research_to_complex():
    router = Router(MockConfig())

    assert asyncio.run(router.classify("Ocr the image")) == "complex"
    result = asyncio.run(router.classify("research this https://example.com"))
    assert result == "complex"


def test_web_fetch_blocks_local_and_private_urls():
    async def run():
        agent = AIAgent(MockConfig(), MockStorage())

        assert "Blocked" in await agent._fetch_url("http://localhost:8000")  # noqa: SLF001
        assert "Blocked" in await agent._fetch_url("http://169.254.169.254")  # noqa: SLF001
        assert "Invalid URL" in await agent._fetch_url("file:///etc/passwd")  # noqa: SLF001

    asyncio.run(run())
