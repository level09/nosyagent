"""
Notion integration for NosyAgent.
Thin async wrapper around notion-client with markdown conversion.
"""

import logging
import re
from typing import Optional

from notion_client import AsyncClient

logger = logging.getLogger(__name__)


class NotionService:
    def __init__(self, token: str):
        self.client = AsyncClient(auth=token)

    # === Search ===

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search Notion pages. Returns simplified results."""
        r = await self.client.search(query=query, page_size=limit)
        results = []
        for page in r.get("results", []):
            title = _extract_title(page)
            results.append({
                "id": page["id"],
                "title": title,
                "url": page.get("url", ""),
                "type": page["object"],
                "last_edited": page.get("last_edited_time", ""),
            })
        return results

    # === Read ===

    async def read_page(self, page_id: str) -> dict:
        """Read a page's metadata and content as markdown."""
        page = await self.client.pages.retrieve(page_id=page_id)
        title = _extract_title(page)

        # Fetch all blocks recursively
        markdown = await self._blocks_to_markdown(page_id)

        return {"id": page_id, "title": title, "url": page.get("url", ""), "content": markdown}

    async def _blocks_to_markdown(self, block_id: str, depth: int = 0) -> str:
        """Convert Notion blocks to markdown, recursively."""
        lines = []
        cursor = None

        while True:
            kwargs = {"block_id": block_id, "page_size": 100}
            if cursor:
                kwargs["start_cursor"] = cursor

            r = await self.client.blocks.children.list(**kwargs)

            for block in r["results"]:
                md = _block_to_md(block, depth)
                if md is not None:
                    lines.append(md)

                # Recurse into children
                if block.get("has_children"):
                    child_md = await self._blocks_to_markdown(block["id"], depth + 1)
                    if child_md:
                        lines.append(child_md)

            if not r.get("has_more"):
                break
            cursor = r.get("next_cursor")

        return "\n".join(lines)

    # === Create ===

    async def create_page(
        self, title: str, content_markdown: str, parent_page_id: Optional[str] = None
    ) -> dict:
        """Create a new page with markdown content. If no parent, creates in first available page."""
        blocks = _markdown_to_blocks(content_markdown)

        parent = {"page_id": parent_page_id} if parent_page_id else {"page_id": await self._default_parent()}

        page = await self.client.pages.create(
            parent=parent,
            properties={"title": [{"text": {"content": title}}]},
            children=blocks,
        )
        return {"id": page["id"], "url": page.get("url", ""), "title": title}

    # === Append ===

    async def append_to_page(self, page_id: str, content_markdown: str) -> bool:
        """Append markdown content to an existing page."""
        blocks = _markdown_to_blocks(content_markdown)
        await self.client.blocks.children.append(block_id=page_id, children=blocks)
        return True

    async def _default_parent(self) -> str:
        """Find a default parent page to create under."""
        r = await self.client.search(query="", page_size=1)
        if r["results"]:
            return r["results"][0]["id"]
        raise ValueError("No accessible Notion pages to create under")


# === Block <-> Markdown conversion ===

def _extract_title(page: dict) -> str:
    props = page.get("properties", {})
    for v in props.values():
        if v.get("type") == "title":
            return "".join(t.get("plain_text", "") for t in v.get("title", []))
    return ""


def _rich_text_to_md(rich_text: list) -> str:
    """Convert Notion rich_text array to markdown."""
    parts = []
    for t in rich_text:
        text = t.get("plain_text", "")
        ann = t.get("annotations", {})
        if ann.get("bold"):
            text = f"**{text}**"
        if ann.get("italic"):
            text = f"*{text}*"
        if ann.get("code"):
            text = f"`{text}`"
        if ann.get("strikethrough"):
            text = f"~~{text}~~"
        href = t.get("href")
        if href:
            text = f"[{text}]({href})"
        parts.append(text)
    return "".join(parts)


def _block_to_md(block: dict, depth: int = 0) -> Optional[str]:
    """Convert a single Notion block to markdown."""
    btype = block["type"]
    data = block.get(btype, {})
    indent = "  " * depth

    rt = data.get("rich_text", [])
    text = _rich_text_to_md(rt)

    if btype == "paragraph":
        return f"{indent}{text}" if text else ""
    elif btype == "heading_1":
        return f"# {text}"
    elif btype == "heading_2":
        return f"## {text}"
    elif btype == "heading_3":
        return f"### {text}"
    elif btype == "bulleted_list_item":
        return f"{indent}- {text}"
    elif btype == "numbered_list_item":
        return f"{indent}1. {text}"
    elif btype == "to_do":
        checked = "x" if data.get("checked") else " "
        return f"{indent}- [{checked}] {text}"
    elif btype == "toggle":
        return f"{indent}<details><summary>{text}</summary>"
    elif btype == "quote":
        return f"{indent}> {text}"
    elif btype == "callout":
        icon = data.get("icon", {}).get("emoji", "")
        return f"{indent}> {icon} {text}"
    elif btype == "code":
        lang = data.get("language", "")
        return f"```{lang}\n{text}\n```"
    elif btype == "divider":
        return "---"
    elif btype == "image":
        url = data.get("external", {}).get("url") or data.get("file", {}).get("url", "")
        caption = _rich_text_to_md(data.get("caption", []))
        return f"![{caption}]({url})" if url else None
    elif btype == "bookmark":
        url = data.get("url", "")
        return f"[{url}]({url})" if url else None
    elif btype == "child_page":
        return f"[{data.get('title', 'Subpage')}]"
    elif btype in ("table", "column_list", "synced_block", "template", "link_preview"):
        return None  # skip complex blocks
    else:
        return f"{indent}{text}" if text else None


def _markdown_to_blocks(markdown: str) -> list:
    """Convert markdown to Notion blocks. Handles common patterns."""
    blocks = []
    lines = markdown.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.startswith("```"):
            lang = line[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            blocks.append({
                "type": "code",
                "code": {"rich_text": [{"text": {"content": "\n".join(code_lines)}}], "language": lang or "plain text"},
            })
            i += 1
            continue

        # Headings
        if line.startswith("### "):
            blocks.append(_text_block("heading_3", line[4:]))
        elif line.startswith("## "):
            blocks.append(_text_block("heading_2", line[3:]))
        elif line.startswith("# "):
            blocks.append(_text_block("heading_1", line[2:]))
        # Divider
        elif line.strip() in ("---", "***", "___"):
            blocks.append({"type": "divider", "divider": {}})
        # Bullet list
        elif line.startswith("- ") or line.startswith("* "):
            blocks.append(_text_block("bulleted_list_item", line[2:]))
        # Numbered list
        elif re.match(r"^\d+\. ", line):
            blocks.append(_text_block("numbered_list_item", re.sub(r"^\d+\. ", "", line)))
        # Quote
        elif line.startswith("> "):
            blocks.append(_text_block("quote", line[2:]))
        # Todo
        elif line.startswith("- [ ] "):
            blocks.append({"type": "to_do", "to_do": {"rich_text": [{"text": {"content": line[6:]}}], "checked": False}})
        elif line.startswith("- [x] "):
            blocks.append({"type": "to_do", "to_do": {"rich_text": [{"text": {"content": line[6:]}}], "checked": True}})
        # Paragraph (skip empty lines)
        elif line.strip():
            blocks.append(_text_block("paragraph", line))

        i += 1

    return blocks


def _text_block(block_type: str, text: str) -> dict:
    """Create a simple text block."""
    return {
        "type": block_type,
        block_type: {"rich_text": _md_text_to_rich_text(text)},
    }


def _md_text_to_rich_text(text: str) -> list:
    """Convert inline markdown (bold, italic, code, links) to Notion rich_text."""
    # Simple approach: parse bold, italic, code, links
    segments = []
    remaining = text

    pattern = re.compile(
        r"(?P<bold>\*\*(.+?)\*\*)"
        r"|(?P<italic>\*(.+?)\*)"
        r"|(?P<code>`(.+?)`)"
        r"|(?P<link>\[(.+?)\]\((.+?)\))"
    )

    last_end = 0
    for m in pattern.finditer(remaining):
        # Plain text before match
        if m.start() > last_end:
            segments.append({"text": {"content": remaining[last_end : m.start()]}})

        if m.group("bold"):
            segments.append({
                "text": {"content": m.group(2)},
                "annotations": {"bold": True},
            })
        elif m.group("italic"):
            segments.append({
                "text": {"content": m.group(4)},
                "annotations": {"italic": True},
            })
        elif m.group("code"):
            segments.append({
                "text": {"content": m.group(6)},
                "annotations": {"code": True},
            })
        elif m.group("link"):
            segments.append({
                "text": {"content": m.group(8), "link": {"url": m.group(9)}},
            })

        last_end = m.end()

    # Remaining text after last match
    if last_end < len(remaining):
        segments.append({"text": {"content": remaining[last_end:]}})

    return segments if segments else [{"text": {"content": text}}]
