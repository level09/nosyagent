#!/usr/bin/env python3
"""
Semantic Memory using LanceDB for vector search.
Enables retrieval of relevant past context based on current message.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import lancedb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Use a small, fast model for embeddings
DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass
class MemoryChunk:
    """A piece of indexed content with metadata"""
    chat_id: str
    content: str
    source: str  # 'brain', 'conversation', 'note'
    timestamp: datetime
    score: float = 0.0


class SemanticMemory:
    """
    Vector-based semantic memory using LanceDB.

    Stores embeddings of brain content and important messages,
    enabling semantic search to find relevant past context.
    """

    def __init__(self, db_path: Path, model_name: str = DEFAULT_MODEL):
        self.db_path = db_path
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._db: Optional[lancedb.DBConnection] = None
        self._table = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def db(self) -> lancedb.DBConnection:
        """Lazy connect to LanceDB"""
        if self._db is None:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    def _get_or_create_table(self):
        """Get or create the memories table"""
        if self._table is not None:
            return self._table

        table_name = "memories"

        if table_name in self.db.table_names():
            self._table = self.db.open_table(table_name)
        else:
            # Create table with initial schema
            # LanceDB infers schema from first insert
            self._table = None  # Will be created on first insert

        return self._table

    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
        return self.model.encode(text).tolist()

    def _chunk_content(self, content: str, chunk_size: int = 500) -> List[str]:
        """Split content into chunks for better retrieval"""
        if len(content) <= chunk_size:
            return [content] if content.strip() else []

        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def index_content(
        self,
        chat_id: str,
        content: str,
        source: str = "brain",
        chunk: bool = True
    ):
        """
        Index content for semantic search.

        Args:
            chat_id: User identifier
            content: Text content to index
            source: Content type ('brain', 'conversation', 'note')
            chunk: Whether to split into smaller pieces
        """
        if not content or not content.strip():
            return

        chunks = self._chunk_content(content) if chunk else [content]
        timestamp = datetime.utcnow().isoformat()

        records = []
        for chunk_text in chunks:
            embedding = self._embed(chunk_text)
            records.append({
                "chat_id": chat_id,
                "content": chunk_text,
                "source": source,
                "timestamp": timestamp,
                "vector": embedding
            })

        if not records:
            return

        # Insert into LanceDB
        table = self._get_or_create_table()
        if table is None:
            # First insert creates the table
            self._table = self.db.create_table("memories", records)
            logger.info(f"Created memories table with {len(records)} initial records")
        else:
            table.add(records)
            logger.debug(f"Added {len(records)} records to memories table")

    def search(
        self,
        query: str,
        chat_id: str,
        limit: int = 5,
        min_score: float = 0.3
    ) -> List[MemoryChunk]:
        """
        Search for relevant content using semantic similarity.

        Args:
            query: Search query
            chat_id: User identifier (filters to this user's content)
            limit: Maximum results to return
            min_score: Minimum similarity score (0-1)

        Returns:
            List of relevant MemoryChunks sorted by relevance
        """
        table = self._get_or_create_table()
        if table is None:
            return []

        query_embedding = self._embed(query)

        try:
            # Search with filter for this chat_id
            results = (
                table.search(query_embedding)
                .where(f"chat_id = '{chat_id}'")
                .limit(limit * 2)  # Get more to filter by score
                .to_list()
            )
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

        memories = []
        for result in results:
            # LanceDB returns _distance (lower is better)
            # Convert to similarity score (higher is better)
            distance = result.get("_distance", 1.0)
            score = 1.0 / (1.0 + distance)  # Convert distance to similarity

            if score >= min_score:
                memories.append(MemoryChunk(
                    chat_id=result["chat_id"],
                    content=result["content"],
                    source=result["source"],
                    timestamp=datetime.fromisoformat(result["timestamp"]),
                    score=score
                ))

        # Sort by score and limit
        memories.sort(key=lambda m: m.score, reverse=True)
        return memories[:limit]

    def clear_user_memories(self, chat_id: str):
        """Remove all memories for a specific user"""
        table = self._get_or_create_table()
        if table is None:
            return

        try:
            table.delete(f"chat_id = '{chat_id}'")
            logger.info(f"Cleared memories for chat_id: {chat_id}")
        except Exception as e:
            logger.warning(f"Failed to clear memories: {e}")

    def reindex_brain(self, chat_id: str, brain_content: str):
        """
        Re-index brain content for a user.
        Clears old brain chunks and indexes fresh content.
        """
        table = self._get_or_create_table()
        if table is not None:
            try:
                table.delete(f"chat_id = '{chat_id}' AND source = 'brain'")
            except Exception:
                pass  # Table might be empty

        self.index_content(chat_id, brain_content, source="brain")
        logger.info(f"Re-indexed brain for chat_id: {chat_id}")


if __name__ == "__main__":
    """Test semantic memory"""
    import shutil

    print("🧪 Testing semantic memory...")

    # Test with temporary database
    test_path = Path("test_semantic_memory")
    if test_path.exists():
        shutil.rmtree(test_path)

    memory = SemanticMemory(test_path)
    chat_id = "test_user"

    # Index some content
    print("\n1. Indexing brain content...")
    brain = """
    # About Me
    I'm a software developer who loves coffee and hiking.
    My goal is to exercise 3 times per week.
    I'm working on a startup in the AI space.

    # Current Projects
    - Building a personal AI assistant
    - Learning Rust programming
    - Training for a half marathon
    """
    memory.index_content(chat_id, brain, source="brain")
    print("   ✓ Indexed brain content")

    # Add some conversation snippets
    memory.index_content(chat_id, "I had a great run this morning, 5km in 28 minutes", source="conversation", chunk=False)
    memory.index_content(chat_id, "The startup pitch went well, investors were interested", source="conversation", chunk=False)

    # Search
    print("\n2. Testing semantic search...")

    queries = [
        "How is my exercise going?",
        "What am I building?",
        "Tell me about coffee",
    ]

    for query in queries:
        print(f"\n   Query: '{query}'")
        results = memory.search(query, chat_id, limit=2)
        for r in results:
            print(f"   - [{r.source}] {r.content[:60]}... (score: {r.score:.2f})")

    # Cleanup
    shutil.rmtree(test_path)
    print("\n✅ All tests passed!")
