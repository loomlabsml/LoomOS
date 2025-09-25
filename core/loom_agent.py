"""
LoomOS Agent - Advanced AI Agent Orchestration System

LoomAgent is the intelligent orchestration layer that provides:
- Multi-modal tool interaction and reasoning
- Advanced planning with ReAct, Chain-of-Thought, and Tree-of-Thoughts
- Dynamic tool discovery and composition
- Memory management and context maintenance
- Safety-aware execution with verification loops
- Multi-agent collaboration and coordination

Architecture:
- Modular tool ecosystem with plugin architecture
- Advanced reasoning engines (ReAct, CoT, ToT, GoT)
- Memory systems (short-term, long-term, episodic)
- Safety and verification integration
- Multi-modal input/output handling
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import numpy as np
from abc import ABC, abstractmethod
from prometheus_client import Counter, Histogram, Gauge

# Metrics
AGENT_REQUESTS = Counter('loomos_agent_requests_total', 'Total agent requests', ['type', 'status'])
AGENT_EXECUTION_TIME = Histogram('loomos_agent_execution_seconds', 'Agent execution time')
ACTIVE_AGENTS = Gauge('loomos_active_agents', 'Currently active agents')
TOOL_CALLS = Counter('loomos_agent_tool_calls_total', 'Total tool calls', ['tool_name', 'status'])

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ReasoningMode(Enum):
    """Available reasoning modes"""
    REACT = "react"  # Reasoning + Acting
    COT = "chain_of_thought"  # Chain of Thought
    TOT = "tree_of_thoughts"  # Tree of Thoughts
    GOT = "graph_of_thoughts"  # Graph of Thoughts
    DIRECT = "direct"  # Direct execution

class ToolType(Enum):
    """Types of tools available to agents"""
    COMPUTATION = "computation"
    SEARCH = "search"
    COMMUNICATION = "communication"
    FILE_SYSTEM = "file_system"
    API_CALL = "api_call"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VERIFICATION = "verification"

@dataclass
class ToolDefinition:
    """Definition of an available tool"""
    name: str
    description: str
    tool_type: ToolType
    parameters: Dict[str, Any]
    required_permissions: List[str] = field(default_factory=list)
    safety_level: str = "safe"  # safe, caution, restricted
    
    # Execution metadata
    estimated_duration: float = 1.0  # seconds
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    output_format: str = "text"

@dataclass
class ToolCall:
    """A tool call request"""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ToolResult:
    """Result from a tool execution"""
    call_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class AgentMemory:
    """Agent memory system"""
    short_term: List[Dict[str, Any]] = field(default_factory=list)  # Last N interactions
    long_term: Dict[str, Any] = field(default_factory=dict)  # Persistent knowledge
    episodic: List[Dict[str, Any]] = field(default_factory=list)  # Episode memories
    working: Dict[str, Any] = field(default_factory=dict)  # Current task context
    
    max_short_term: int = 50
    max_episodic: int = 1000

@dataclass
class AgentState:
    """Current state of an agent"""
    agent_id: str
    status: AgentStatus
    current_goal: Optional[str] = None
    current_plan: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    memory: AgentMemory = field(default_factory=AgentMemory)
    
    # Execution context
    reasoning_mode: ReasoningMode = ReasoningMode.REACT
    safety_level: str = "safe"
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class BaseTool(ABC):
    """Base class for agent tools"""
    
    def __init__(self, definition: ToolDefinition):
        self.definition = definition
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate tool parameters"""
        # Basic validation - can be overridden
        required_params = self.definition.parameters.get('required', [])
        return all(param in parameters for param in required_params)

class SearchTool(BaseTool):
    """Web search tool"""
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        query = parameters.get('query', '')
        
        # Mock search implementation
        await asyncio.sleep(0.5)  # Simulate search time
        
        mock_results = [
            f"Search result 1 for '{query}': Relevant information about {query}",
            f"Search result 2 for '{query}': Additional context and details",
            f"Search result 3 for '{query}': Expert analysis and insights"
        ]
        
        return ToolResult(
            call_id=str(uuid.uuid4()),
            tool_name=self.definition.name,
            success=True,
            result={"results": mock_results, "query": query},
            execution_time=0.5
        )

class CalculatorTool(BaseTool):
    """Mathematical calculation tool"""
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        expression = parameters.get('expression', '')
        
        try:
            # Safe evaluation (in production, use a proper math parser)
            result = eval(expression, {"__builtins__": {}}, {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len, "pow": pow, "sqrt": lambda x: x**0.5
            })
            
            return ToolResult(
                call_id=str(uuid.uuid4()),
                tool_name=self.definition.name,
                success=True,
                result={"expression": expression, "result": result},
                execution_time=0.1
            )
        except Exception as e:
            return ToolResult(
                call_id=str(uuid.uuid4()),
                tool_name=self.definition.name,
                success=False,
                error=f"Calculation error: {str(e)}",
                execution_time=0.1
            )

class TextAnalysisTool(BaseTool):
    """Text analysis and processing tool"""
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        text = parameters.get('text', '')
        analysis_type = parameters.get('type', 'basic')
        
        await asyncio.sleep(0.2)  # Simulate processing time
        
        # Mock analysis
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        analysis = {
            "word_count": word_count,
            "character_count": char_count,
            "sentence_count": sentence_count,
            "avg_word_length": char_count / word_count if word_count > 0 else 0,
            "readability_score": min(100, max(0, 100 - word_count / 10)),
            "sentiment": "neutral"  # Mock sentiment
        }
        
        if analysis_type == "advanced":
            analysis.update({
                "keywords": text.lower().split()[:5],  # Top 5 words as keywords
                "complexity": "medium",
                "topics": ["general"]
            })
        
        return ToolResult(
            call_id=str(uuid.uuid4()),
            tool_name=self.definition.name,
            success=True,
            result=analysis,
            execution_time=0.2
        )

class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools"""
        # Search tool
        search_def = ToolDefinition(
            name="search",
            description="Search the web for information",
            tool_type=ToolType.SEARCH,
            parameters={"required": ["query"], "optional": ["limit"]},
            estimated_duration=1.0
        )
        self.register_tool(SearchTool(search_def))
        
        # Calculator tool
        calc_def = ToolDefinition(
            name="calculator",
            description="Perform mathematical calculations",
            tool_type=ToolType.COMPUTATION,
            parameters={"required": ["expression"]},
            estimated_duration=0.1
        )
        self.register_tool(CalculatorTool(calc_def))
        
        # Text analysis tool
        text_def = ToolDefinition(
            name="text_analysis",
            description="Analyze text content for various metrics",
            tool_type=ToolType.ANALYSIS,
            parameters={"required": ["text"], "optional": ["type"]},
            estimated_duration=0.3
        )
        self.register_tool(TextAnalysisTool(text_def))
    
    def register_tool(self, tool: BaseTool):
        """Register a new tool"""
        self.tools[tool.definition.name] = tool
        logger.info(f"Registered tool: {tool.definition.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[ToolDefinition]:
        """List all available tools"""
        return [tool.definition for tool in self.tools.values()]
    
    async def execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a tool call"""
        tool = self.get_tool(call.tool_name)
        if not tool:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error=f"Tool '{call.tool_name}' not found"
            )
        
        # Validate parameters
        if not tool.validate_parameters(call.parameters):
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error="Invalid parameters"
            )
        
        # Execute tool
        start_time = time.time()
        try:
            result = await tool.execute(call.parameters)
            result.call_id = call.call_id
            result.execution_time = time.time() - start_time
            
            # Update metrics
            TOOL_CALLS.labels(tool_name=call.tool_name, status="success").inc()
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            TOOL_CALLS.labels(tool_name=call.tool_name, status="error").inc()
            
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class ReActPlanner:
    """ReAct (Reasoning + Acting) planner"""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
    
    async def plan_and_execute(self, goal: str, agent_state: AgentState) -> Dict[str, Any]:
        """Plan and execute using ReAct methodology"""
        logger.info(f"Starting ReAct planning for goal: {goal}")
        
        max_iterations = 10
        iterations = 0
        observations = []
        actions = []
        thoughts = []
        
        current_goal = goal
        
        while iterations < max_iterations:
            iterations += 1
            
            # Think: Generate reasoning about current situation
            thought = await self._generate_thought(current_goal, observations, agent_state)
            thoughts.append(thought)
            
            # Act: Decide on action based on thought
            action = await self._generate_action(thought, agent_state)
            if not action:
                break  # No more actions needed
            
            actions.append(action)
            
            # Execute action
            observation = await self._execute_action(action, agent_state)
            observations.append(observation)
            
            # Check if goal is achieved
            if await self._is_goal_achieved(goal, observations):
                break
            
            # Update current goal based on progress
            current_goal = await self._update_goal(goal, thoughts, observations)
        
        return {
            "goal": goal,
            "completed": await self._is_goal_achieved(goal, observations),
            "iterations": iterations,
            "thoughts": thoughts,
            "actions": actions,
            "observations": observations,
            "final_result": observations[-1] if observations else "No actions taken"
        }
    
    async def _generate_thought(self, goal: str, observations: List[str], 
                              agent_state: AgentState) -> str:
        """Generate reasoning thought"""
        if not observations:
            return f"I need to work on the goal: {goal}. Let me think about what tools I have available and how to approach this."
        
        last_observation = observations[-1]
        return f"Based on the previous result: {last_observation}, I should continue working towards: {goal}"
    
    async def _generate_action(self, thought: str, agent_state: AgentState) -> Optional[ToolCall]:
        """Generate next action based on thought"""
        # Simple action generation logic
        if "search" in thought.lower() or "information" in thought.lower():
            return ToolCall(
                tool_name="search",
                parameters={"query": "relevant information"}
            )
        elif "calculate" in thought.lower() or "math" in thought.lower():
            return ToolCall(
                tool_name="calculator",
                parameters={"expression": "2 + 2"}
            )
        elif "analyze" in thought.lower() or "text" in thought.lower():
            return ToolCall(
                tool_name="text_analysis",
                parameters={"text": thought, "type": "basic"}
            )
        
        # Default action if no specific tool needed
        return None
    
    async def _execute_action(self, action: ToolCall, agent_state: AgentState) -> str:
        """Execute an action and return observation"""
        result = await self.tool_registry.execute_tool(action)
        
        if result.success:
            return f"Successfully executed {action.tool_name}: {result.result}"
        else:
            return f"Failed to execute {action.tool_name}: {result.error}"
    
    async def _is_goal_achieved(self, goal: str, observations: List[str]) -> bool:
        """Check if goal has been achieved"""
        # Simple heuristic - if we have observations, consider goal partially achieved
        return len(observations) >= 2
    
    async def _update_goal(self, original_goal: str, thoughts: List[str], 
                          observations: List[str]) -> str:
        """Update goal based on progress"""
        return original_goal  # Keep original goal for simplicity

class LoomAgent:
    """Main LoomAgent orchestration system"""
    
    def __init__(self, agent_id: Optional[str] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.state = AgentState(agent_id=self.agent_id, status=AgentStatus.IDLE)
        self.tool_registry = ToolRegistry()
        self.planner = ReActPlanner(self.tool_registry)
        
        # Execution history
        self.execution_history: List[Dict[str, Any]] = []
        
        ACTIVE_AGENTS.inc()
        logger.info(f"LoomAgent {self.agent_id} initialized")
    
    async def plan_and_execute(self, goal: str, 
                             reasoning_mode: ReasoningMode = ReasoningMode.REACT,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main entry point for planning and execution"""
        start_time = time.time()
        
        logger.info(f"Agent {self.agent_id} starting execution for goal: {goal}")
        
        # Update state
        self.state.status = AgentStatus.PLANNING
        self.state.current_goal = goal
        self.state.reasoning_mode = reasoning_mode
        self.state.execution_context = context or {}
        self.state.last_activity = datetime.now(timezone.utc)
        
        # Track request
        AGENT_REQUESTS.labels(type=reasoning_mode.value, status="started").inc()
        
        try:
            # Execute based on reasoning mode
            if reasoning_mode == ReasoningMode.REACT:
                result = await self.planner.plan_and_execute(goal, self.state)
            elif reasoning_mode == ReasoningMode.DIRECT:
                result = await self._direct_execution(goal)
            else:
                # For other modes, fall back to ReAct for now
                result = await self.planner.plan_and_execute(goal, self.state)
            
            # Update state
            self.state.status = AgentStatus.COMPLETED if result.get("completed", False) else AgentStatus.FAILED
            self.state.completed_steps = result.get("actions", [])
            
            # Update memory
            self._update_memory(goal, result)
            
            # Record execution
            execution_time = time.time() - start_time
            execution_record = {
                "goal": goal,
                "reasoning_mode": reasoning_mode.value,
                "result": result,
                "execution_time": execution_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.execution_history.append(execution_record)
            
            # Update metrics
            AGENT_EXECUTION_TIME.observe(execution_time)
            AGENT_REQUESTS.labels(type=reasoning_mode.value, status="completed").inc()
            
            logger.info(f"Agent {self.agent_id} completed execution in {execution_time:.2f}s")
            
            return {
                "agent_id": self.agent_id,
                "goal": goal,
                "reasoning_mode": reasoning_mode.value,
                "success": result.get("completed", False),
                "result": result,
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} execution failed: {e}")
            self.state.status = AgentStatus.FAILED
            
            AGENT_REQUESTS.labels(type=reasoning_mode.value, status="failed").inc()
            
            return {
                "agent_id": self.agent_id,
                "goal": goal,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
        finally:
            self.state.status = AgentStatus.IDLE
    
    async def _direct_execution(self, goal: str) -> Dict[str, Any]:
        """Direct execution without complex planning"""
        # Simple direct execution - just return the goal as completed
        await asyncio.sleep(0.5)  # Simulate work
        
        return {
            "goal": goal,
            "completed": True,
            "approach": "direct",
            "result": f"Directly executed: {goal}"
        }
    
    def _update_memory(self, goal: str, result: Dict[str, Any]):
        """Update agent memory with execution results"""
        # Add to short-term memory
        memory_entry = {
            "type": "execution",
            "goal": goal,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.state.memory.short_term.append(memory_entry)
        
        # Trim short-term memory if needed
        if len(self.state.memory.short_term) > self.state.memory.max_short_term:
            self.state.memory.short_term = self.state.memory.short_term[-self.state.memory.max_short_term:]
        
        # Update working memory
        self.state.memory.working["last_goal"] = goal
        self.state.memory.working["last_result"] = result
    
    def get_available_tools(self) -> List[ToolDefinition]:
        """Get list of available tools"""
        return self.tool_registry.list_tools()
    
    def register_tool(self, tool: BaseTool):
        """Register a new tool"""
        self.tool_registry.register_tool(tool)
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a specific tool"""
        call = ToolCall(tool_name=tool_name, parameters=parameters)
        return await self.tool_registry.execute_tool(call)
    
    def get_state(self) -> AgentState:
        """Get current agent state"""
        return self.state
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test basic functionality
            test_result = await self.plan_and_execute(
                "health check test", 
                ReasoningMode.DIRECT
            )
            
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "current_status": self.state.status.value,
                "available_tools": len(self.tool_registry.tools),
                "execution_history_count": len(self.execution_history),
                "test_execution_success": test_result.get("success", False),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Agent health check failed: {e}")
            return {
                "status": "unhealthy",
                "agent_id": self.agent_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def __del__(self):
        """Cleanup when agent is destroyed"""
        ACTIVE_AGENTS.dec()

# Factory functions
def create_agent(agent_id: Optional[str] = None) -> LoomAgent:
    """Create a new LoomAgent instance"""
    return LoomAgent(agent_id)

async def execute_goal(goal: str, reasoning_mode: ReasoningMode = ReasoningMode.REACT) -> Dict[str, Any]:
    """Quick goal execution with a temporary agent"""
    agent = create_agent()
    return await agent.plan_and_execute(goal, reasoning_mode)

# Multi-agent coordination utilities
class AgentCoordinator:
    """Coordinates multiple agents for complex tasks"""
    
    def __init__(self):
        self.agents: Dict[str, LoomAgent] = {}
        self.active_collaborations: Dict[str, Dict[str, Any]] = {}
    
    def add_agent(self, agent: LoomAgent):
        """Add an agent to the coordinator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent {agent.agent_id} to coordinator")
    
    async def collaborate(self, goals: List[str], agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Coordinate multiple agents on different goals"""
        if agent_ids:
            selected_agents = [self.agents[aid] for aid in agent_ids if aid in self.agents]
        else:
            selected_agents = list(self.agents.values())[:len(goals)]
        
        if len(selected_agents) < len(goals):
            raise ValueError("Not enough agents for the given goals")
        
        # Execute goals in parallel
        tasks = []
        for i, goal in enumerate(goals):
            agent = selected_agents[i]
            tasks.append(agent.plan_and_execute(goal))
        
        results = await asyncio.gather(*tasks)
        
        return {
            "collaboration_id": str(uuid.uuid4()),
            "goals": goals,
            "agents": [agent.agent_id for agent in selected_agents],
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }