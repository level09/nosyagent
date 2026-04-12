# NosyAgent Memory Revamp

## Goal

NosyAgent should borrow the safest ideas from Hippo Memory without adopting Hippo as a dependency or changing the app into a separate agent platform. The v1 revamp keeps SQLite, LanceDB, Telegram, and the existing brain file, then adds a structured memory lifecycle around them.

The principle is simple: good memory is not saving everything forever. Useful memories should survive because they are recalled, confirmed, or consolidated. Weak, stale, or contradicted memories should lose authority.

## Non-Negotiable Easy Wins

1. **Decay by default**: every structured memory has strength and half-life.
2. **Retrieval strengthens memory**: recalled memories gain strength and live longer.
3. **Confidence tiers**: memory is labeled as `verified`, `observed`, `inferred`, or `stale`.
4. **Episodic vs semantic layers**: event-like observations and durable facts are separate.
5. **Sleep consolidation**: a daily worker job decays, merges, promotes, and scans memories.
6. **Superseding old facts**: newer memory can mark older memory as stale.
7. **Observation framing**: memories enter prompts as dated context, not instructions.

## V1 Architecture

SQLite remains the source of truth. The new `memory_items` table stores structured memories with layer, confidence, strength, half-life, retrieval count, timestamps, tags, and supersession links. The existing `brain` table remains a human-readable summary and compatibility layer.

LanceDB remains optional semantic search. It helps retrieve older context, but correctness does not depend on embeddings. If LanceDB is unavailable, structured memory still works through SQLite keyword search.

Daily consolidation runs from the existing ARQ worker. It uses deterministic heuristics in v1:

- decay strength based on half-life and inactivity
- mark weak or old unrecalled memories as `stale`
- weaken duplicate memories while keeping the stronger copy
- promote repeated episodic memories into semantic memories
- flag obvious conflicts such as changed residence, employer, medication, or preference

The v1 consolidation pass does not delete existing conversations, brain records, entities, or brain files. Weak memories are marked `stale` and duplicates are weakened rather than removed.

Before deploying or running risky maintenance, create a local backup:

```bash
./scripts/backup_data.sh
```

Backups are written under `data/backups/<timestamp>/` and include the SQLite database, semantic memory directory when present, `brain/`, instruction files, and `.env`.

## User-Facing Commands

Telegram and CLI expose the same inspection surface:

- `/memory` or `/memory status`: show memory counts by layer and confidence
- `/memory search <query>`: recall matching structured memories and strengthen them
- `/memory conflicts`: show unresolved contradictions
- `/memory sleep`: preview consolidation
- `/memory sleep --run`: apply consolidation

## Prompt Safety

Structured memory is injected as observational context:

```text
[verified] Previously observed on 2026-04-13: User prefers short direct replies.
[observed] Previously observed on 2026-04-13: User is evaluating biological memory ideas.
[inferred] Possible pattern from 2026-04-13: User prefers low-debug features.
[stale] Older observation from 2026-03-01: User was using an older deployment flow.
```

This prevents stale memory from acting like a command and makes uncertainty visible to the model.

## Deferred

These Hippo ideas are intentionally out of scope for v1:

- adding the `hippo-memory` Node package
- cross-tool hooks for Claude Code, Cursor, or Codex
- MCP server integration
- Redis Stack or a new memory service
- LLM-powered sleep consolidation
- automatic command failure capture
- cross-project shared memory
- dashboard/UI work

Those are useful later, but they increase operational surface area. V1 focuses on memory lifecycle quality inside the current NosyAgent architecture.
