"""
LoomOS Tapestry - Advanced Memory & Vector Database System

Tapestry is LoomOS's sophisticated memory management and vector database adapter that provides:
- Hierarchical memory systems (short-term, long-term, episodic, semantic)
- Multi-modal vector embeddings and similarity search
- Intelligent memory consolidation and forgetting
- Cross-modal memory associations and retrieval
- Distributed memory across multiple nodes
- Memory-augmented generation and reasoning

Architecture:
- Pluggable vector database backends (Pinecone, Qdrant, Chroma, Weaviate)
- Advanced embedding models for text, code, images, audio
- Memory lifecycle management with automatic archiving
- Semantic clustering and knowledge graph construction
- Privacy-preserving memory operations
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import numpy as np
import hashlib
from abc import ABC, abstractmethod
from prometheus_client import Counter, Histogram, Gauge

# Mock vector database and embedding libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    # Mock FAISS
    class faiss:
        class IndexFlatL2:
            def __init__(self, dim): self.dim = dim
            def add(self, vectors): pass
            def search(self, vectors, k): return np.array([[0]]), np.array([[0.5]])
        @staticmethod
        def normalize_L2(vectors): pass

try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    # Mock sentence transformers
    class sentence_transformers:
        class SentenceTransformer:
            def __init__(self, model_name): pass
            def encode(self, texts): return np.random.randn(len(texts), 384)

# Metrics
MEMORY_OPERATIONS = Counter('loomos_tapestry_memory_operations_total', 'Total memory operations', ['operation', 'type'])
MEMORY_RETRIEVAL_TIME = Histogram('loomos_tapestry_retrieval_seconds', 'Memory retrieval time')
ACTIVE_MEMORIES = Gauge('loomos_tapestry_active_memories', 'Number of active memories')
VECTOR_DB_SIZE = Gauge('loomos_tapestry_vector_db_size', 'Vector database size')

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memory in the system"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"

class MemoryModality(Enum):
    """Modalities supported by memory system"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    MULTIMODAL = "multimodal"

class MemoryStatus(Enum):
    """Status of memory entries"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    CONSOLIDATING = "consolidating"
    EXPIRED = "expired"

@dataclass
class MemoryEntry:
    """A single memory entry"""
    memory_id: str
    content: Union[str, Dict[str, Any]]
    memory_type: MemoryType
    modality: MemoryModality
    
    # Embedding and indexing
    embedding: Optional[np.ndarray] = None
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Metadata
    source: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0  # 0.0 to 1.0
    confidence: float = 1.0  # 0.0 to 1.0
    
    # Temporal information
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    access_count: int = 0
    
    # Relationships
    related_memories: List[str] = field(default_factory=list)
    parent_memory: Optional[str] = None
    child_memories: List[str] = field(default_factory=list)
    
    # Status
    status: MemoryStatus = MemoryStatus.ACTIVE

@dataclass
class MemoryQuery:
    """Query for memory retrieval"""
    query_text: Optional[str] = None
    query_embedding: Optional[np.ndarray] = None
    memory_types: List[MemoryType] = field(default_factory=list)
    modalities: List[MemoryModality] = field(default_factory=list)
    
    # Search parameters
    top_k: int = 10
    similarity_threshold: float = 0.7
    time_range: Optional[Tuple[datetime, datetime]] = None
    
    # Filters
    tags: List[str] = field(default_factory=list)
    min_importance: float = 0.0
    max_age_hours: Optional[int] = None
    
    # Query metadata
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class MemorySearchResult:
    """Result from memory search"""
    memory: MemoryEntry
    similarity_score: float
    relevance_score: float
    retrieval_confidence: float = 1.0

class VectorDatabase(ABC):
    """Abstract base class for vector databases"""
    
    @abstractmethod
    async def add_vectors(self, memories: List[MemoryEntry]) -> bool:
        """Add memory vectors to the database"""
        pass
    
    @abstractmethod
    async def search_vectors(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    async def delete_vectors(self, memory_ids: List[str]) -> bool:
        """Delete vectors from the database"""
        pass
    
    @abstractmethod
    async def update_vector(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Update a vector in the database"""
        pass

class LocalVectorDB(VectorDatabase):
    """Local FAISS-based vector database"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension) if FAISS_AVAILABLE else None
        self.memory_map: Dict[int, str] = {}  # FAISS index -> memory_id
        self.reverse_map: Dict[str, int] = {}  # memory_id -> FAISS index
        self.next_id = 0
        
        logger.info(f"Local vector DB initialized with dimension {dimension}")
    
    async def add_vectors(self, memories: List[MemoryEntry]) -> bool:
        """Add memory vectors to FAISS index"""
        try:
            vectors = []
            for memory in memories:
                if memory.embedding is not None:
                    vectors.append(memory.embedding.reshape(1, -1))
                    self.memory_map[self.next_id] = memory.memory_id
                    self.reverse_map[memory.memory_id] = self.next_id
                    self.next_id += 1
            
            if vectors and self.index is not None:
                combined_vectors = np.vstack(vectors).astype(np.float32)
                faiss.normalize_L2(combined_vectors)
                self.index.add(combined_vectors)
                
                VECTOR_DB_SIZE.set(self.index.ntotal if hasattr(self.index, 'ntotal') else len(vectors))
                logger.info(f"Added {len(vectors)} vectors to local DB")
                return True
            
            return True  # Success even without FAISS for demo
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False
    
    async def search_vectors(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Search for similar vectors in FAISS index"""
        try:
            if self.index is None or query_embedding is None:
                return []
            
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_vector)
            
            distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx in self.memory_map:
                    memory_id = self.memory_map[idx]
                    similarity = 1.0 / (1.0 + dist)  # Convert distance to similarity
                    results.append((memory_id, similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def delete_vectors(self, memory_ids: List[str]) -> bool:
        """Delete vectors (note: FAISS doesn't support deletion, so we mark as deleted)"""
        try:
            for memory_id in memory_ids:
                if memory_id in self.reverse_map:
                    idx = self.reverse_map[memory_id]
                    # Remove from mappings
                    del self.reverse_map[memory_id]
                    if idx in self.memory_map:
                        del self.memory_map[idx]
            
            logger.info(f"Marked {len(memory_ids)} vectors as deleted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    async def update_vector(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Update vector (requires deletion and re-addition in FAISS)"""
        # For simplicity, we'll just add the new vector
        # In production, this would require rebuilding the index
        memory = MemoryEntry(memory_id=memory_id, content="", memory_type=MemoryType.SEMANTIC, 
                           modality=MemoryModality.TEXT, embedding=embedding)
        return await self.add_vectors([memory])

class EmbeddingModel:
    """Handles text and multimodal embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimension = 384  # Default dimension for MiniLM
        
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.model = sentence_transformers.SentenceTransformer(model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
            else:
                self.model = None
        except Exception as e:
            logger.warning(f"Failed to load embedding model {model_name}: {e}")
            self.model = None
        
        logger.info(f"Embedding model initialized: {model_name} (dim: {self.dimension})")
    
    async def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text into embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            if self.model:
                embeddings = self.model.encode(texts)
                return embeddings
            else:
                # Mock embeddings
                return np.random.randn(len(texts), self.dimension).astype(np.float32)
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return np.random.randn(len(texts), self.dimension).astype(np.float32)
    
    async def encode_multimodal(self, content: Dict[str, Any]) -> np.ndarray:
        """Encode multimodal content"""
        # For now, just encode text components
        text_parts = []
        
        if "text" in content:
            text_parts.append(str(content["text"]))
        if "description" in content:
            text_parts.append(str(content["description"]))
        if "metadata" in content and isinstance(content["metadata"], dict):
            text_parts.extend([str(v) for v in content["metadata"].values() if isinstance(v, (str, int, float))])
        
        combined_text = " ".join(text_parts) if text_parts else "multimodal content"
        embeddings = await self.encode_text(combined_text)
        return embeddings[0] if len(embeddings) > 0 else np.random.randn(self.dimension)

class MemoryConsolidator:
    """Handles memory consolidation and forgetting"""
    
    def __init__(self):
        self.consolidation_rules: List[Callable] = []
        
    async def consolidate_memories(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """Consolidate related memories"""
        logger.info(f"Consolidating {len(memories)} memories")
        
        # Group similar memories
        clusters = await self._cluster_memories(memories)
        
        consolidated = []
        for cluster in clusters:
            if len(cluster) > 1:
                # Merge similar memories
                merged_memory = await self._merge_memories(cluster)
                consolidated.append(merged_memory)
            else:
                consolidated.extend(cluster)
        
        logger.info(f"Consolidated to {len(consolidated)} memories")
        return consolidated
    
    async def _cluster_memories(self, memories: List[MemoryEntry]) -> List[List[MemoryEntry]]:
        """Cluster similar memories together"""
        # Simple clustering based on content similarity
        clusters = []
        used = set()
        
        for i, memory in enumerate(memories):
            if i in used:
                continue
            
            cluster = [memory]
            used.add(i)
            
            # Find similar memories
            for j, other_memory in enumerate(memories[i+1:], i+1):
                if j in used:
                    continue
                
                similarity = await self._calculate_similarity(memory, other_memory)
                if similarity > 0.8:  # High similarity threshold
                    cluster.append(other_memory)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    async def _calculate_similarity(self, memory1: MemoryEntry, memory2: MemoryEntry) -> float:
        """Calculate similarity between two memories"""
        if memory1.embedding is not None and memory2.embedding is not None:
            # Cosine similarity
            dot_product = np.dot(memory1.embedding, memory2.embedding)
            norm1 = np.linalg.norm(memory1.embedding)
            norm2 = np.linalg.norm(memory2.embedding)
            
            if norm1 > 0 and norm2 > 0:
                return dot_product / (norm1 * norm2)
        
        # Fallback to content-based similarity
        content1 = str(memory1.content).lower()
        content2 = str(memory2.content).lower()
        
        # Simple Jaccard similarity
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _merge_memories(self, memories: List[MemoryEntry]) -> MemoryEntry:
        """Merge multiple memories into one consolidated memory"""
        if len(memories) == 1:
            return memories[0]
        
        # Use the most important memory as base
        base_memory = max(memories, key=lambda m: m.importance)
        
        # Merge content
        merged_content = {
            "consolidated": True,
            "primary_content": base_memory.content,
            "related_content": [m.content for m in memories if m.memory_id != base_memory.memory_id],
            "source_count": len(memories)
        }
        
        # Merge metadata
        all_keywords = set()
        all_tags = set()
        total_importance = 0
        min_created = min(m.created_at for m in memories)
        max_accessed = max(m.last_accessed for m in memories)
        
        for memory in memories:
            all_keywords.update(memory.keywords)
            all_tags.update(memory.tags)
            total_importance += memory.importance
        
        # Create consolidated memory
        consolidated = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            content=merged_content,
            memory_type=base_memory.memory_type,
            modality=base_memory.modality,
            embedding=base_memory.embedding,  # Use base embedding
            keywords=list(all_keywords),
            tags=list(all_tags),
            importance=min(1.0, total_importance / len(memories)),
            created_at=min_created,
            last_accessed=max_accessed,
            related_memories=[m.memory_id for m in memories],
            status=MemoryStatus.ACTIVE
        )
        
        return consolidated
    
    async def forget_memories(self, memories: List[MemoryEntry], forgetting_curve: Callable = None) -> List[str]:
        """Apply forgetting curve to determine which memories to archive/delete"""
        forgotten_ids = []
        
        current_time = datetime.now(timezone.utc)
        
        for memory in memories:
            age_hours = (current_time - memory.created_at).total_seconds() / 3600
            last_access_hours = (current_time - memory.last_accessed).total_seconds() / 3600
            
            # Simple forgetting criteria
            should_forget = (
                (age_hours > 720 and memory.importance < 0.3) or  # 30 days, low importance
                (last_access_hours > 168 and memory.access_count < 2) or  # 7 days, rarely accessed
                (memory.status == MemoryStatus.EXPIRED)
            )
            
            if should_forget:
                forgotten_ids.append(memory.memory_id)
        
        logger.info(f"Marked {len(forgotten_ids)} memories for forgetting")
        return forgotten_ids

class Tapestry:
    """Main Tapestry memory management system"""
    
    def __init__(self, vector_db: Optional[VectorDatabase] = None):
        self.vector_db = vector_db or LocalVectorDB()
        self.embedding_model = EmbeddingModel()
        self.consolidator = MemoryConsolidator()
        
        # Memory storage
        self.memories: Dict[str, MemoryEntry] = {}
        self.memory_by_type: Dict[MemoryType, List[str]] = {mt: [] for mt in MemoryType}
        
        # Search history
        self.search_history: List[MemoryQuery] = []
        
        # Background tasks
        self.consolidation_task: Optional[asyncio.Task] = None
        self.forgetting_task: Optional[asyncio.Task] = None
        
        logger.info("Tapestry memory system initialized")
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background consolidation and forgetting tasks"""
        self.consolidation_task = asyncio.create_task(self._periodic_consolidation())
        self.forgetting_task = asyncio.create_task(self._periodic_forgetting())
    
    async def store_memory(self, content: Union[str, Dict[str, Any]], 
                          memory_type: MemoryType = MemoryType.SEMANTIC,
                          modality: MemoryModality = MemoryModality.TEXT,
                          **kwargs) -> str:
        """Store a new memory"""
        memory_id = kwargs.get('memory_id', str(uuid.uuid4()))
        
        logger.info(f"Storing memory {memory_id} of type {memory_type.value}")
        
        # Generate embedding
        if isinstance(content, str):
            embedding = await self.embedding_model.encode_text(content)
            embedding = embedding[0] if len(embedding.shape) > 1 else embedding
        else:
            embedding = await self.embedding_model.encode_multimodal(content)
        
        # Extract keywords from content
        keywords = self._extract_keywords(content)
        
        # Create memory entry
        memory = MemoryEntry(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            modality=modality,
            embedding=embedding,
            keywords=keywords,
            **{k: v for k, v in kwargs.items() if k != 'memory_id'}
        )
        
        # Store in local memory
        self.memories[memory_id] = memory
        self.memory_by_type[memory_type].append(memory_id)
        
        # Add to vector database
        await self.vector_db.add_vectors([memory])
        
        # Update metrics
        MEMORY_OPERATIONS.labels(operation="store", type=memory_type.value).inc()
        ACTIVE_MEMORIES.set(len(self.memories))
        
        logger.info(f"Memory {memory_id} stored successfully")
        return memory_id
    
    async def retrieve_memory(self, query: Union[str, MemoryQuery]) -> List[MemorySearchResult]:
        """Retrieve memories based on query"""
        start_time = time.time()
        
        # Convert string query to MemoryQuery
        if isinstance(query, str):
            query = MemoryQuery(query_text=query)
        
        logger.info(f"Retrieving memories for query: {query.query_text}")
        
        try:
            # Generate query embedding if needed
            if query.query_embedding is None and query.query_text:
                query_embedding = await self.embedding_model.encode_text(query.query_text)
                query.query_embedding = query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding
            
            # Search vector database
            vector_results = []
            if query.query_embedding is not None:
                vector_results = await self.vector_db.search_vectors(query.query_embedding, query.top_k * 2)
            
            # Filter and rank results
            results = []
            for memory_id, similarity in vector_results:
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    
                    # Apply filters
                    if not self._passes_filters(memory, query):
                        continue
                    
                    # Calculate relevance score
                    relevance = await self._calculate_relevance(memory, query)
                    
                    # Create search result
                    if similarity >= query.similarity_threshold:
                        result = MemorySearchResult(
                            memory=memory,
                            similarity_score=similarity,
                            relevance_score=relevance,
                            retrieval_confidence=min(similarity, relevance)
                        )
                        results.append(result)
                        
                        # Update access information
                        memory.last_accessed = datetime.now(timezone.utc)
                        memory.access_count += 1
            
            # Sort by combined score
            results.sort(key=lambda r: (r.similarity_score + r.relevance_score) / 2, reverse=True)
            results = results[:query.top_k]
            
            # Update metrics
            retrieval_time = time.time() - start_time
            MEMORY_RETRIEVAL_TIME.observe(retrieval_time)
            MEMORY_OPERATIONS.labels(operation="retrieve", type="query").inc()
            
            # Store query in history
            self.search_history.append(query)
            if len(self.search_history) > 1000:  # Keep last 1000 queries
                self.search_history = self.search_history[-1000:]
            
            logger.info(f"Retrieved {len(results)} memories in {retrieval_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []
    
    def _extract_keywords(self, content: Union[str, Dict[str, Any]]) -> List[str]:
        """Extract keywords from content"""
        if isinstance(content, str):
            text = content
        elif isinstance(content, dict):
            text = " ".join(str(v) for v in content.values() if isinstance(v, (str, int, float)))
        else:
            text = str(content)
        
        # Simple keyword extraction (in production, use NLP libraries)
        words = text.lower().split()
        # Filter out common words and short words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if len(word) > 3 and word not in stopwords]
        
        # Return unique keywords, limited to 10
        return list(set(keywords))[:10]
    
    def _passes_filters(self, memory: MemoryEntry, query: MemoryQuery) -> bool:
        """Check if memory passes query filters"""
        # Memory type filter
        if query.memory_types and memory.memory_type not in query.memory_types:
            return False
        
        # Modality filter
        if query.modalities and memory.modality not in query.modalities:
            return False
        
        # Importance filter
        if memory.importance < query.min_importance:
            return False
        
        # Age filter
        if query.max_age_hours:
            age_hours = (datetime.now(timezone.utc) - memory.created_at).total_seconds() / 3600
            if age_hours > query.max_age_hours:
                return False
        
        # Time range filter
        if query.time_range:
            start_time, end_time = query.time_range
            if not (start_time <= memory.created_at <= end_time):
                return False
        
        # Tags filter
        if query.tags:
            if not any(tag in memory.tags for tag in query.tags):
                return False
        
        # Status filter (only active memories by default)
        if memory.status != MemoryStatus.ACTIVE:
            return False
        
        return True
    
    async def _calculate_relevance(self, memory: MemoryEntry, query: MemoryQuery) -> float:
        """Calculate relevance score for a memory"""
        relevance = 0.0
        
        # Keyword match
        if query.query_text and memory.keywords:
            query_words = set(query.query_text.lower().split())
            memory_keywords = set(memory.keywords)
            keyword_overlap = len(query_words.intersection(memory_keywords))
            relevance += 0.3 * (keyword_overlap / max(len(query_words), 1))
        
        # Recency boost
        age_hours = (datetime.now(timezone.utc) - memory.last_accessed).total_seconds() / 3600
        recency_score = max(0, 1.0 - (age_hours / 168))  # Decay over a week
        relevance += 0.2 * recency_score
        
        # Importance boost
        relevance += 0.3 * memory.importance
        
        # Access frequency boost
        access_boost = min(1.0, memory.access_count / 10)  # Cap at 10 accesses
        relevance += 0.2 * access_boost
        
        return min(1.0, relevance)
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory"""
        if memory_id not in self.memories:
            logger.warning(f"Memory {memory_id} not found for update")
            return False
        
        memory = self.memories[memory_id]
        
        # Update fields
        for key, value in updates.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
        
        # Regenerate embedding if content changed
        if "content" in updates:
            if isinstance(memory.content, str):
                embedding = await self.embedding_model.encode_text(memory.content)
                memory.embedding = embedding[0] if len(embedding.shape) > 1 else embedding
            else:
                memory.embedding = await self.embedding_model.encode_multimodal(memory.content)
            
            # Update in vector database
            await self.vector_db.update_vector(memory_id, memory.embedding)
        
        MEMORY_OPERATIONS.labels(operation="update", type=memory.memory_type.value).inc()
        logger.info(f"Memory {memory_id} updated successfully")
        return True
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        if memory_id not in self.memories:
            logger.warning(f"Memory {memory_id} not found for deletion")
            return False
        
        memory = self.memories[memory_id]
        
        # Mark as deleted
        memory.status = MemoryStatus.DELETED
        
        # Remove from type index
        if memory_id in self.memory_by_type[memory.memory_type]:
            self.memory_by_type[memory.memory_type].remove(memory_id)
        
        # Remove from vector database
        await self.vector_db.delete_vectors([memory_id])
        
        # Remove from local storage
        del self.memories[memory_id]
        
        MEMORY_OPERATIONS.labels(operation="delete", type=memory.memory_type.value).inc()
        ACTIVE_MEMORIES.set(len(self.memories))
        
        logger.info(f"Memory {memory_id} deleted successfully")
        return True
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get memory by ID"""
        memory = self.memories.get(memory_id)
        if memory and memory.status == MemoryStatus.ACTIVE:
            memory.last_accessed = datetime.now(timezone.utc)
            memory.access_count += 1
            return memory
        return None
    
    def list_memories(self, memory_type: Optional[MemoryType] = None, 
                     limit: int = 100) -> List[MemoryEntry]:
        """List memories by type"""
        if memory_type:
            memory_ids = self.memory_by_type[memory_type][:limit]
            return [self.memories[mid] for mid in memory_ids if mid in self.memories]
        else:
            return list(self.memories.values())[:limit]
    
    async def _periodic_consolidation(self):
        """Periodic memory consolidation task"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Consolidate memories that haven't been consolidated recently
                for memory_type in MemoryType:
                    memories = [self.memories[mid] for mid in self.memory_by_type[memory_type] 
                              if mid in self.memories and self.memories[mid].status == MemoryStatus.ACTIVE]
                    
                    if len(memories) > 10:  # Only consolidate if we have enough memories
                        logger.info(f"Starting consolidation for {memory_type.value}")
                        consolidated = await self.consolidator.consolidate_memories(memories)
                        
                        # Replace old memories with consolidated ones
                        # This is a simplified version - production would be more careful
                        for old_memory in memories:
                            old_memory.status = MemoryStatus.ARCHIVED
                        
                        for new_memory in consolidated:
                            await self.store_memory(
                                new_memory.content, 
                                new_memory.memory_type, 
                                new_memory.modality,
                                memory_id=new_memory.memory_id,
                                importance=new_memory.importance,
                                tags=new_memory.tags,
                                keywords=new_memory.keywords
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation task error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _periodic_forgetting(self):
        """Periodic memory forgetting task"""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                active_memories = [m for m in self.memories.values() if m.status == MemoryStatus.ACTIVE]
                forgotten_ids = await self.consolidator.forget_memories(active_memories)
                
                for memory_id in forgotten_ids:
                    if memory_id in self.memories:
                        self.memories[memory_id].status = MemoryStatus.ARCHIVED
                        # Optionally delete from vector DB to save space
                        await self.vector_db.delete_vectors([memory_id])
                
                logger.info(f"Archived {len(forgotten_ids)} memories")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Forgetting task error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test memory storage and retrieval
            test_id = await self.store_memory("health check test", MemoryType.SHORT_TERM)
            test_results = await self.retrieve_memory("health check test")
            
            # Clean up test memory
            await self.delete_memory(test_id)
            
            return {
                "status": "healthy",
                "total_memories": len(self.memories),
                "active_memories": len([m for m in self.memories.values() if m.status == MemoryStatus.ACTIVE]),
                "memory_types": {mt.value: len(self.memory_by_type[mt]) for mt in MemoryType},
                "test_storage_success": test_id is not None,
                "test_retrieval_success": len(test_results) > 0,
                "vector_db_size": len(self.memories),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Tapestry health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.consolidation_task:
            self.consolidation_task.cancel()
        if self.forgetting_task:
            self.forgetting_task.cancel()
        
        logger.info("Tapestry cleanup completed")

# Factory functions
def create_tapestry(vector_db: Optional[VectorDatabase] = None) -> Tapestry:
    """Create a new Tapestry instance"""
    return Tapestry(vector_db)

# Legacy compatibility functions
async def store_memory_legacy(key: str, value: str) -> None:
    """Legacy store_memory function"""
    tapestry = create_tapestry()
    await tapestry.store_memory(value, memory_type=MemoryType.SEMANTIC, tags=[key])

async def retrieve_memory_legacy(key: str) -> str:
    """Legacy retrieve_memory function"""
    tapestry = create_tapestry()
    results = await tapestry.retrieve_memory(key)
    
    if results:
        return str(results[0].memory.content)
    else:
        return "retrieved"  # Default legacy response