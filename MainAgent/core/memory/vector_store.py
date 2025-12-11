# ============================================================================
# FILE: vector_store.py
# Semantic search using LanceDB
# ============================================================================

from typing import List, Optional, Tuple, Dict
from datetime import datetime
import json
from pathlib import Path

from .memory_types import MemoryEntry, MemoryPriority

# Optional sentence_transformers import
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import lancedb
    HAS_LANCEDB = True
except ImportError:
    lancedb = None
    HAS_LANCEDB = False

# Optional numpy import for vector operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


class VectorStore:
    """
    Semantic search using LanceDB for:
    - Finding relevant code examples
    - Retrieving similar bug fixes
    - Discovering related documentation
    - Context-aware code suggestions
    """

    def __init__(self, embedding_dim: int = 384, storage_path: Optional[str] = None):
        if not HAS_LANCEDB:
            raise ImportError("lancedb is required. Install with: pip install lancedb")

        self.embedding_dim = embedding_dim
        self.storage_path = Path(storage_path) if storage_path else Path("./.agent_memory/vectors")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model (optional - falls back to simple embedding if unavailable)
        self.model = None
        if HAS_SENTENCE_TRANSFORMERS and SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.model = None

        # Initialize LanceDB
        self.db = lancedb.connect(str(self.storage_path))
        self.table_name = "memory_entries"
        self.table = None
        self._init_table()

    def _init_table(self):
        """Initialize or connect to LanceDB table"""
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            # Table doesn't exist yet, will be created on first add
            self.table = None

    def add_entry(self, entry: MemoryEntry, embedding: Optional[List[float]] = None):
        """Add entry with embedding to LanceDB"""
        if embedding is None:
            embedding = self._simple_embedding(entry.content)

        # Prepare record for LanceDB
        record = {
            "id": len(self) if self.table else 0,
            "content": entry.content,
            "embedding": embedding,
            "entry_type": "memory",
            "priority": entry.priority.value if hasattr(entry.priority, 'value') else str(entry.priority),
            "timestamp": entry.timestamp.isoformat() if isinstance(entry.timestamp, datetime) else str(entry.timestamp),
            "agent_id": entry.agent_id,
            "related_files": json.dumps(entry.related_files if entry.related_files else []),
            "tags": json.dumps(entry.tags if entry.tags else []),
            "metadata": "{}",
        }
        
        if self.table is None:
            # Create table with first record
            self.table = self.db.create_table(self.table_name, data=[record], mode="overwrite")
        else:
            # Add to existing table
            self.table.add([record])
    
    def _simple_embedding(self, text: str) -> List[float]:
        """
        Generate a simple word-frequency based embedding.
        Used as fallback when sentence-transformers is unavailable.
        """
        words = text.lower().split()
        word_freq: Dict[str, int] = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        embedding = [0.0] * self.embedding_dim
        for word, freq in word_freq.items():
            idx = hash(word) % self.embedding_dim
            embedding[idx] += freq

        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        Falls back to simple embedding if sentence_transformers is unavailable.
        """
        if self.model is not None:
            try:
                # SentenceTransformer.encode expects a string or list of strings
                embedding = self.model.encode(text)
                # Convert numpy array to list if needed
                if hasattr(embedding, 'tolist'):
                    return embedding.tolist()
                return list(embedding)
            except Exception:
                pass
        # Fallback to simple embedding
        return self._simple_embedding(text)

    def semantic_search(self, query: str, k: int = 5,
                        filters: Optional[Dict[str, List[str]]] = None) -> List[Tuple[MemoryEntry, float]]:
        """
        Find k most semantically similar entries using LanceDB vector search.
        filters: {"tags": ["bug_fix"], "files": ["main.py"], "agents": ["agent_1"]}
        """
        if self.table is None:
            return []

        query_embedding = self.generate_embedding(query)
        
        # Build where clause from filters
        where_clause = self._build_where_clause(filters)
        
        try:
            # Perform vector search
            results = self.table.search(query_embedding).limit(k)
            
            if where_clause:
                results = results.where(where_clause)
            
            search_results = results.to_list()
            
            # Convert results to (MemoryEntry, similarity) tuples
            output = []
            for result in search_results:
                entry = self._dict_to_entry(result)
                # LanceDB returns distance; convert to similarity
                similarity = 1 / (1 + result.get("_distance", 0))
                output.append((entry, similarity))
            
            return output
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def search_by_file(self, filename: str, k: int = 5) -> List[MemoryEntry]:
        """Find all entries related to a specific file"""
        if self.table is None:
            return []

        try:
            results = self.table.search().where(
                f"related_files LIKE '%{filename}%'"
            ).limit(k).to_list()
            
            entries = [self._dict_to_entry(r) for r in results]
            return sorted(entries, key=lambda x: x.timestamp, reverse=True)
        except Exception as e:
            print(f"File search error: {e}")
            return []

    def search_by_agent(self, agent_id: str, k: int = 10) -> List[MemoryEntry]:
        """Get recent work by specific agent"""
        if self.table is None:
            return []

        try:
            results = self.table.search().where(
                f"agent_id = '{agent_id}'"
            ).limit(k).to_list()
            
            entries = [self._dict_to_entry(r) for r in results]
            return sorted(entries, key=lambda x: x.timestamp, reverse=True)
        except Exception as e:
            print(f"Agent search error: {e}")
            return []

    def search_by_tags(self, tags: List[str], k: int = 10) -> List[MemoryEntry]:
        """Get entries matching any of the tags"""
        if self.table is None:
            return []

        try:
            # Build OR condition for tags
            tag_conditions = [f"tags LIKE '%{tag}%'" for tag in tags]
            where_clause = " OR ".join(tag_conditions)
            
            results = self.table.search().where(where_clause).limit(k).to_list()
            
            entries = [self._dict_to_entry(r) for r in results]
            return sorted(entries, key=lambda x: x.timestamp, reverse=True)
        except Exception as e:
            print(f"Tag search error: {e}")
            return []

    def keyword_search(self, query: str, k: int = 5,
                       filters: Optional[Dict[str, List[str]]] = None) -> List[Tuple[MemoryEntry, float]]:
        """Fallback text-based search using LanceDB"""
        if self.table is None:
            return []

        try:
            where_clause = self._build_where_clause(filters)
            query_lower = query.lower()
            
            # Search with text filter
            results = self.table.search().where(
                f"content LIKE '%{query_lower}%'"
            )
            
            if where_clause:
                results = results.where(where_clause)
            
            all_results = results.limit(k * 2).to_list()
            
            # Score by relevance
            scored = []
            for result in all_results:
                entry = self._dict_to_entry(result)
                content_lower = entry.content.lower()
                word_matches = sum(1 for word in query_lower.split() if word in content_lower)
                exact_match = query_lower in content_lower
                score = word_matches + (10 if exact_match else 0)
                
                if score > 0:
                    scored.append((entry, float(score)))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:k]
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []

    def _build_where_clause(self, filters: Optional[Dict[str, List[str]]]) -> Optional[str]:
        """Build LanceDB where clause from filters"""
        if not filters:
            return None

        conditions = []

        if "tags" in filters and filters["tags"]:
            tag_conditions = [f"tags LIKE '%{tag}%'" for tag in filters["tags"]]
            conditions.append(f"({' OR '.join(tag_conditions)})")

        if "files" in filters and filters["files"]:
            file_conditions = [f"related_files LIKE '%{file}%'" for file in filters["files"]]
            conditions.append(f"({' OR '.join(file_conditions)})")

        if "agents" in filters and filters["agents"]:
            agent_conditions = [f"agent_id = '{agent}'" for agent in filters["agents"]]
            conditions.append(f"({' OR '.join(agent_conditions)})")

        if conditions:
            return " AND ".join(conditions)
        
        return None

    def _dict_to_entry(self, record: Dict) -> MemoryEntry:
        """Convert LanceDB record back to MemoryEntry"""
        # Parse priority from stored value
        priority_str = record.get("priority", "2")
        try:
            if isinstance(priority_str, int):
                priority = MemoryPriority(priority_str)
            else:
                priority = MemoryPriority(int(priority_str))
        except (ValueError, TypeError):
            priority = MemoryPriority.MEDIUM

        return MemoryEntry(
            content=record.get("content", ""),
            timestamp=datetime.fromisoformat(record.get("timestamp", datetime.now().isoformat())),
            priority=priority,
            agent_id=record.get("agent_id", ""),
            related_files=json.loads(record.get("related_files", "[]")),
            tags=json.loads(record.get("tags", "[]")),
        )

    def save_to_disk(self, filename: str = "vector_store.json"):
        """Persist vector store metadata to disk (LanceDB handles data persistence)"""
        if self.table is None:
            return

        metadata = {
            "embedding_dim": self.embedding_dim,
            "table_name": self.table_name,
            "storage_path": str(self.storage_path),
            "saved_at": datetime.now().isoformat(),
            "total_entries": len(self)
        }

        filepath = self.storage_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def load_from_disk(self, filename: str = "vector_store.json") -> bool:
        """Load vector store (LanceDB handles data loading)"""
        filepath = self.storage_path / filename

        if not filepath.exists():
            # Metadata file missing but LanceDB table may exist
            self._init_table()
            return self.table is not None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.embedding_dim = data.get("embedding_dim", 384)
            self._init_table()
            return True
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load vector store: {e}")
            return False

    def clear(self):
        """Clear all entries"""
        if self.table is not None:
            try:
                self.db.drop_table(self.table_name)
                self.table = None
            except Exception as e:
                print(f"Error clearing table: {e}")

    def get_stats(self) -> Dict[str, int]:
        """Get vector store statistics"""
        if self.table is None:
            return {
                "total_entries": 0,
                "unique_files": 0,
                "unique_agents": 0,
                "unique_tags": 0
            }

        try:
            all_records = self.table.search().limit(10000).to_list()
            
            unique_files = set()
            unique_agents = set()
            unique_tags = set()
            
            for record in all_records:
                unique_files.update(json.loads(record.get("related_files", "[]")))
                unique_agents.add(record.get("agent_id", ""))
                unique_tags.update(json.loads(record.get("tags", "[]")))
            
            return {
                "total_entries": len(all_records),
                "unique_files": len(unique_files),
                "unique_agents": len(unique_agents),
                "unique_tags": len(unique_tags)
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_entries": 0}

    def __len__(self) -> int:
        if self.table is None:
            return 0
        try:
            return self.table.search().limit(1).to_pandas().shape[0] or len(self.table.search().to_list())
        except Exception:
            return 0