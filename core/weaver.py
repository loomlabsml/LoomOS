"""
Weaver: Chain-of-Thought Traces Manager

Advanced trace management system for LoomOS providing:
- Hierarchical trace storage and compression
- Real-time trace analysis and pattern detection
- Trace-based learning and optimization
- Multi-modal trace support (text, code, reasoning)
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib
from pathlib import Path
import sqlite3
import aiosqlite

logger = logging.getLogger(__name__)

class TraceType(Enum):
    """Types of traces supported by Weaver"""
    REASONING = "reasoning"
    EXECUTION = "execution"
    INTERACTION = "interaction"
    ERROR = "error"
    OPTIMIZATION = "optimization"

class TraceStatus(Enum):
    """Trace processing status"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    FAILED = "failed"

@dataclass
class TraceStep:
    """Individual step in a trace"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    step_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_step_id: Optional[str] = None
    duration_ms: float = 0.0

@dataclass
class Trace:
    """Complete trace object with all steps and metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: TraceType = TraceType.REASONING
    status: TraceStatus = TraceStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    request_id: str = ""
    user_id: str = ""
    session_id: str = ""
    steps: List[TraceStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    compressed_data: Optional[bytes] = None
    compression_ratio: float = 0.0

class TraceCompressor:
    """Advanced trace compression for storage optimization"""
    
    @staticmethod
    def compress_trace(trace: Trace) -> bytes:
        """Compress trace data using advanced algorithms"""
        import gzip
        import pickle
        
        # Convert trace to serializable format
        trace_data = {
            "id": trace.id,
            "type": trace.type.value,
            "status": trace.status.value,
            "created_at": trace.created_at.isoformat(),
            "steps": [
                {
                    "id": step.id,
                    "timestamp": step.timestamp.isoformat(),
                    "step_type": step.step_type,
                    "content": step.content,
                    "metadata": step.metadata,
                    "parent_step_id": step.parent_step_id,
                    "duration_ms": step.duration_ms
                }
                for step in trace.steps
            ],
            "metadata": trace.metadata,
            "tags": trace.tags
        }
        
        # Serialize and compress
        serialized = pickle.dumps(trace_data)
        compressed = gzip.compress(serialized, compresslevel=9)
        
        return compressed
    
    @staticmethod
    def decompress_trace(compressed_data: bytes) -> Trace:
        """Decompress trace data"""
        import gzip
        import pickle
        
        decompressed = gzip.decompress(compressed_data)
        trace_data = pickle.loads(decompressed)
        
        # Reconstruct trace object
        trace = Trace(
            id=trace_data["id"],
            type=TraceType(trace_data["type"]),
            status=TraceStatus(trace_data["status"]),
            created_at=datetime.fromisoformat(trace_data["created_at"]),
            metadata=trace_data["metadata"],
            tags=trace_data["tags"]
        )
        
        # Reconstruct steps
        for step_data in trace_data["steps"]:
            step = TraceStep(
                id=step_data["id"],
                timestamp=datetime.fromisoformat(step_data["timestamp"]),
                step_type=step_data["step_type"],
                content=step_data["content"],
                metadata=step_data["metadata"],
                parent_step_id=step_data["parent_step_id"],
                duration_ms=step_data["duration_ms"]
            )
            trace.steps.append(step)
        
        return trace

class TraceAnalyzer:
    """Advanced trace analysis and pattern detection"""
    
    def __init__(self):
        self.patterns = {}
        self.performance_baselines = {}
    
    async def analyze_trace(self, trace: Trace) -> Dict[str, Any]:
        """Analyze trace for patterns, performance, and anomalies"""
        analysis = {
            "trace_id": trace.id,
            "performance_metrics": self._analyze_performance(trace),
            "pattern_matches": await self._detect_patterns(trace),
            "anomalies": self._detect_anomalies(trace),
            "optimization_suggestions": self._suggest_optimizations(trace),
            "quality_score": self._calculate_quality_score(trace)
        }
        
        return analysis
    
    def _analyze_performance(self, trace: Trace) -> Dict[str, float]:
        """Analyze trace performance metrics"""
        if not trace.steps:
            return {}
        
        total_duration = sum(step.duration_ms for step in trace.steps)
        step_count = len(trace.steps)
        
        return {
            "total_duration_ms": total_duration,
            "avg_step_duration_ms": total_duration / step_count if step_count > 0 else 0,
            "step_count": step_count,
            "reasoning_depth": self._calculate_reasoning_depth(trace),
            "efficiency_score": self._calculate_efficiency_score(trace)
        }
    
    async def _detect_patterns(self, trace: Trace) -> List[Dict[str, Any]]:
        """Detect common patterns in trace execution"""
        patterns = []
        
        # Pattern: Repeated reasoning loops
        reasoning_steps = [s for s in trace.steps if s.step_type == "reasoning"]
        if len(reasoning_steps) > 3:
            patterns.append({
                "type": "reasoning_loop",
                "confidence": 0.8,
                "description": "Multiple reasoning steps detected",
                "step_count": len(reasoning_steps)
            })
        
        # Pattern: Error recovery
        error_steps = [s for s in trace.steps if s.step_type == "error"]
        recovery_steps = [s for s in trace.steps if s.step_type == "recovery"]
        if error_steps and recovery_steps:
            patterns.append({
                "type": "error_recovery",
                "confidence": 0.9,
                "description": "Error recovery pattern detected",
                "error_count": len(error_steps)
            })
        
        return patterns
    
    def _detect_anomalies(self, trace: Trace) -> List[Dict[str, Any]]:
        """Detect anomalies in trace execution"""
        anomalies = []
        
        # Detect unusually long steps
        if trace.steps:
            avg_duration = sum(s.duration_ms for s in trace.steps) / len(trace.steps)
            for step in trace.steps:
                if step.duration_ms > avg_duration * 3:
                    anomalies.append({
                        "type": "slow_step",
                        "step_id": step.id,
                        "duration_ms": step.duration_ms,
                        "avg_duration_ms": avg_duration,
                        "severity": "medium"
                    })
        
        return anomalies
    
    def _suggest_optimizations(self, trace: Trace) -> List[Dict[str, Any]]:
        """Suggest optimizations based on trace analysis"""
        suggestions = []
        
        # Suggest parallel execution for independent steps
        independent_steps = self._find_independent_steps(trace)
        if len(independent_steps) > 2:
            suggestions.append({
                "type": "parallel_execution",
                "description": "Multiple independent steps can be parallelized",
                "potential_speedup": "20-40%",
                "step_ids": independent_steps
            })
        
        return suggestions
    
    def _calculate_reasoning_depth(self, trace: Trace) -> int:
        """Calculate the depth of reasoning in the trace"""
        max_depth = 0
        
        def get_depth(step_id: str, current_depth: int = 0) -> int:
            children = [s for s in trace.steps if s.parent_step_id == step_id]
            if not children:
                return current_depth
            return max(get_depth(child.id, current_depth + 1) for child in children)
        
        root_steps = [s for s in trace.steps if s.parent_step_id is None]
        for root_step in root_steps:
            depth = get_depth(root_step.id)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_efficiency_score(self, trace: Trace) -> float:
        """Calculate efficiency score (0-1) based on trace characteristics"""
        if not trace.steps:
            return 0.0
        
        # Factors: step count, duration, error rate, reasoning depth
        step_count_score = min(1.0, 10 / len(trace.steps))  # Fewer steps = higher score
        
        error_steps = [s for s in trace.steps if s.step_type == "error"]
        error_rate = len(error_steps) / len(trace.steps)
        error_score = 1.0 - error_rate
        
        # Combine scores
        efficiency = (step_count_score * 0.4 + error_score * 0.6)
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_quality_score(self, trace: Trace) -> float:
        """Calculate overall quality score for the trace"""
        if not trace.steps:
            return 0.0
        
        # Factors: completeness, consistency, performance
        completeness_score = 1.0 if trace.status == TraceStatus.COMPLETED else 0.5
        
        # Check for consistent step types
        step_types = set(s.step_type for s in trace.steps)
        consistency_score = min(1.0, len(step_types) / 10)  # Variety is good
        
        efficiency_score = self._calculate_efficiency_score(trace)
        
        quality = (completeness_score * 0.3 + consistency_score * 0.3 + efficiency_score * 0.4)
        return max(0.0, min(1.0, quality))
    
    def _find_independent_steps(self, trace: Trace) -> List[str]:
        """Find steps that can be executed independently"""
        independent = []
        
        for step in trace.steps:
            # Check if step has dependencies
            has_dependencies = any(
                s.parent_step_id == step.id for s in trace.steps
            )
            if not has_dependencies and not step.parent_step_id:
                independent.append(step.id)
        
        return independent

class Weaver:
    """
    Advanced Chain-of-Thought Traces Manager
    
    Provides comprehensive trace management including:
    - Real-time trace creation and updates
    - Advanced compression and storage
    - Pattern detection and analysis
    - Performance optimization suggestions
    - Multi-modal trace support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Weaver with configuration"""
        self.config = config or {}
        self.db_path = self.config.get("db_path", "traces.db")
        self.max_trace_size = self.config.get("max_trace_size", 10000)
        self.compression_threshold = self.config.get("compression_threshold", 1000)
        
        self.active_traces: Dict[str, Trace] = {}
        self.analyzer = TraceAnalyzer()
        self.compressor = TraceCompressor()
        
        # Initialize database
        asyncio.create_task(self._init_database())
        
        logger.info("Weaver initialized with config: %s", self.config)
    
    async def _init_database(self):
        """Initialize SQLite database for trace storage"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    request_id TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    metadata TEXT,
                    tags TEXT,
                    compressed_data BLOB,
                    compression_ratio REAL,
                    analysis_results TEXT
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_request_id ON traces(request_id)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_user_id ON traces(user_id)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces(created_at)
            """)
            
            await db.commit()
    
    async def create_trace(self, trace_data: Dict[str, Any]) -> str:
        """Create a new trace"""
        trace = Trace(
            request_id=trace_data.get("request_id", ""),
            user_id=trace_data.get("user_id", ""),
            session_id=trace_data.get("session_id", ""),
            type=TraceType(trace_data.get("type", "reasoning")),
            metadata=trace_data.get("metadata", {}),
            tags=trace_data.get("tags", [])
        )
        
        self.active_traces[trace.id] = trace
        
        logger.info("Created trace: %s (type: %s)", trace.id, trace.type.value)
        return trace.id
    
    async def add_step(self, trace_id: str, step_data: Dict[str, Any]) -> str:
        """Add a step to an existing trace"""
        if trace_id not in self.active_traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        trace = self.active_traces[trace_id]
        
        step = TraceStep(
            step_type=step_data.get("step_type", ""),
            content=step_data.get("content", {}),
            metadata=step_data.get("metadata", {}),
            parent_step_id=step_data.get("parent_step_id"),
            duration_ms=step_data.get("duration_ms", 0.0)
        )
        
        trace.steps.append(step)
        trace.updated_at = datetime.now(timezone.utc)
        
        # Check if trace needs compression
        if len(trace.steps) > self.compression_threshold:
            await self._compress_and_store_trace(trace)
        
        logger.debug("Added step to trace %s: %s", trace_id, step.id)
        return step.id
    
    async def update_trace(self, trace_id: str, updates: Dict[str, Any]) -> None:
        """Update trace metadata or status"""
        if trace_id not in self.active_traces:
            # Try to load from database
            trace = await self._load_trace_from_db(trace_id)
            if trace:
                self.active_traces[trace_id] = trace
            else:
                raise ValueError(f"Trace {trace_id} not found")
        
        trace = self.active_traces[trace_id]
        
        if "status" in updates:
            trace.status = TraceStatus(updates["status"])
            if trace.status == TraceStatus.COMPLETED:
                trace.completed_at = datetime.now(timezone.utc)
        
        if "metadata" in updates:
            trace.metadata.update(updates["metadata"])
        
        if "tags" in updates:
            trace.tags.extend(updates["tags"])
        
        trace.updated_at = datetime.now(timezone.utc)
        
        logger.debug("Updated trace: %s", trace_id)
    
    async def complete_trace(self, trace_id: str) -> Dict[str, Any]:
        """Complete a trace and perform final analysis"""
        if trace_id not in self.active_traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        trace = self.active_traces[trace_id]
        trace.status = TraceStatus.COMPLETED
        trace.completed_at = datetime.now(timezone.utc)
        
        # Perform final analysis
        analysis = await self.analyzer.analyze_trace(trace)
        
        # Store trace to database
        await self._store_trace_to_db(trace, analysis)
        
        # Remove from active traces
        del self.active_traces[trace_id]
        
        logger.info("Completed trace: %s", trace_id)
        return analysis
    
    async def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID"""
        if trace_id in self.active_traces:
            return self.active_traces[trace_id]
        
        return await self._load_trace_from_db(trace_id)
    
    async def search_traces(self, 
                          user_id: Optional[str] = None,
                          trace_type: Optional[TraceType] = None,
                          tags: Optional[List[str]] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Search traces with filters"""
        query = "SELECT id, type, status, created_at, metadata, tags FROM traces WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if trace_type:
            query += " AND type = ?"
            params.append(trace_type.value)
        
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    trace_data = {
                        "id": row[0],
                        "type": row[1],
                        "status": row[2],
                        "created_at": row[3],
                        "metadata": json.loads(row[4]) if row[4] else {},
                        "tags": json.loads(row[5]) if row[5] else []
                    }
                    
                    # Filter by tags if specified
                    if tags and not any(tag in trace_data["tags"] for tag in tags):
                        continue
                    
                    results.append(trace_data)
                
                return results
    
    async def get_trace_analytics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics for traces"""
        query = "SELECT type, status, created_at, analysis_results FROM traces"
        params = []
        
        if user_id:
            query += " WHERE user_id = ?"
            params.append(user_id)
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                
                analytics = {
                    "total_traces": len(rows),
                    "traces_by_type": {},
                    "traces_by_status": {},
                    "avg_quality_score": 0.0,
                    "common_patterns": [],
                    "performance_trends": []
                }
                
                quality_scores = []
                
                for row in rows:
                    trace_type = row[0]
                    status = row[1]
                    analysis_data = json.loads(row[3]) if row[3] else {}
                    
                    # Count by type and status
                    analytics["traces_by_type"][trace_type] = analytics["traces_by_type"].get(trace_type, 0) + 1
                    analytics["traces_by_status"][status] = analytics["traces_by_status"].get(status, 0) + 1
                    
                    # Collect quality scores
                    if "quality_score" in analysis_data:
                        quality_scores.append(analysis_data["quality_score"])
                
                # Calculate average quality score
                if quality_scores:
                    analytics["avg_quality_score"] = sum(quality_scores) / len(quality_scores)
                
                return analytics
    
    async def _compress_and_store_trace(self, trace: Trace) -> None:
        """Compress and store trace to save memory"""
        original_size = len(json.dumps([step.__dict__ for step in trace.steps]))
        compressed_data = self.compressor.compress_trace(trace)
        compressed_size = len(compressed_data)
        
        trace.compressed_data = compressed_data
        trace.compression_ratio = compressed_size / original_size if original_size > 0 else 0
        
        # Clear steps from memory but keep compressed data
        trace.steps = []
        
        logger.debug("Compressed trace %s: %.2f%% reduction", 
                    trace.id, (1 - trace.compression_ratio) * 100)
    
    async def _store_trace_to_db(self, trace: Trace, analysis: Optional[Dict[str, Any]] = None) -> None:
        """Store trace to database"""
        # Compress if not already compressed
        if not trace.compressed_data and trace.steps:
            compressed_data = self.compressor.compress_trace(trace)
            compression_ratio = len(compressed_data) / len(json.dumps([step.__dict__ for step in trace.steps]))
        else:
            compressed_data = trace.compressed_data
            compression_ratio = trace.compression_ratio
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO traces 
                (id, type, status, created_at, updated_at, completed_at, request_id, user_id, session_id, 
                 metadata, tags, compressed_data, compression_ratio, analysis_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace.id,
                trace.type.value,
                trace.status.value,
                trace.created_at.isoformat(),
                trace.updated_at.isoformat(),
                trace.completed_at.isoformat() if trace.completed_at else None,
                trace.request_id,
                trace.user_id,
                trace.session_id,
                json.dumps(trace.metadata),
                json.dumps(trace.tags),
                compressed_data,
                compression_ratio,
                json.dumps(analysis) if analysis else None
            ))
            await db.commit()
    
    async def _load_trace_from_db(self, trace_id: str) -> Optional[Trace]:
        """Load trace from database"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT * FROM traces WHERE id = ?
            """, (trace_id,)) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                # Reconstruct trace
                if row[11]:  # compressed_data
                    trace = self.compressor.decompress_trace(row[11])
                else:
                    trace = Trace(id=trace_id)
                
                # Update metadata from database
                trace.type = TraceType(row[1])
                trace.status = TraceStatus(row[2])
                trace.created_at = datetime.fromisoformat(row[3])
                trace.updated_at = datetime.fromisoformat(row[4])
                if row[5]:
                    trace.completed_at = datetime.fromisoformat(row[5])
                trace.request_id = row[6] or ""
                trace.user_id = row[7] or ""
                trace.session_id = row[8] or ""
                trace.metadata = json.loads(row[9]) if row[9] else {}
                trace.tags = json.loads(row[10]) if row[10] else []
                
                return trace
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get Weaver metrics"""
        return {
            "active_traces": len(self.active_traces),
            "total_steps": sum(len(trace.steps) for trace in self.active_traces.values()),
            "avg_steps_per_trace": (
                sum(len(trace.steps) for trace in self.active_traces.values()) / 
                len(self.active_traces) if self.active_traces else 0
            ),
            "memory_usage_mb": sum(
                len(json.dumps([step.__dict__ for step in trace.steps])) 
                for trace in self.active_traces.values()
            ) / (1024 * 1024)
        }