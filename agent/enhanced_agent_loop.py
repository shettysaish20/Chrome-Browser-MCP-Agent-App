# enhanced_agent_loop.py - Enhanced agent loop with multi-directional routing
"""
Enhanced Agent Loop with Multi-Directional Routing Support

Extends the existing agent loop to support:
- Multi-directional agent routing with return_to parameter
- BrowserAgent integration
- Enhanced error handling and recovery
- Agent scope and priority management
"""

import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import re
import json
import time

# Import existing modules
from perception.perception import Perception, build_perception_input
from decision.decision import Decision, build_decision_input
from summarization.summarizer import Summarizer
from agent.contextManager import ContextManager
from agent.agentSession import AgentSession
from memory.memory_search import MemorySearch
from action.execute_step import execute_step_with_mode
from utils.utils import log_step, log_error, save_final_plan, log_json_block

# Import browser-related modules
from agent.browserAgent import BrowserAgent, create_browser_agent
from agent.browser_tasks import Browser

class Route:
    SUMMARIZE = "summarize"
    DECISION = "decision"
    BROWSER = "browser"  # New route for browser operations
    DELEGATE = "delegate"  # New route for agent delegation

class StepType:
    ROOT = "ROOT"
    CODE = "CODE"
    BROWSER = "BROWSER"  # New step type for browser operations

class AgentType(Enum):
    PERCEPTION = "perception"
    DECISION = "decision"
    EXECUTION = "execution"
    BROWSER = "browser"  # New agent type
    SUMMARIZER = "summarizer"

class AgentPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class RoutingContext:
    """Manages routing between agents"""
    
    def __init__(self):
        self.current_agent = AgentType.PERCEPTION
        self.agent_stack = []
        self.routing_history = []
        self.return_paths = {}
        
    def push_agent(self, agent_type, return_to=None):
        """Push current agent to stack and switch to new agent"""
        # Convert string to AgentType if needed
        if isinstance(agent_type, str):
            try:
                agent_type = AgentType(agent_type.lower())
            except ValueError:
                agent_type = AgentType.PERCEPTION
                
        self.agent_stack.append({
            "agent": self.current_agent,
            "return_to": return_to,
            "timestamp": datetime.now()
        })
        self.current_agent = agent_type
        
    def pop_agent(self):
        """Pop agent from stack and return to it"""
        if self.agent_stack:
            previous = self.agent_stack.pop()
            self.current_agent = previous["agent"]
            return self.current_agent.value if hasattr(self.current_agent, 'value') else str(self.current_agent)
        return None
        
    def get_current_agent(self):
        """Get the current active agent"""
        if hasattr(self.current_agent, 'value'):
            return self.current_agent.value
        return str(self.current_agent)
        
    def get_return_path(self) -> Optional[AgentType]:
        """Get the return path for current agent"""
        if self.agent_stack:
            return self.agent_stack[-1].get("return_to")
        return None

class EnhancedAgentLoop:
    """Enhanced agent loop with multi-directional routing support"""
    
    def __init__(self, perception_prompt, decision_prompt, summarizer_prompt, browser_prompt, multi_mcp, strategy="exploratory"):
        # Initialize existing components
        self.perception = Perception(perception_prompt)
        self.decision = Decision(decision_prompt, multi_mcp)
        self.summarizer = Summarizer(summarizer_prompt)
        self.browser = Browser(browser_prompt)

        self.multi_mcp = multi_mcp
        self.strategy = strategy
        self.status: str = "in_progress"
        self.final_output = None  # Initialize final output
        
        # Initialize execution state
        self.code_variants = {}
        self.next_step_id = "ROOT"
        self.d_out = None
        self.p_out = {}  # Initialize as empty dict instead of None
        # Initialize routing and agents
        self.routing_context = RoutingContext()
        self.logger = logging.getLogger("EnhancedAgentLoop")
        self.browser_agent = None  # Will be initialized when needed
        self.query = ""
    
    async def run(self, query: str, initial_agent: AgentType = AgentType.PERCEPTION):
        """Enhanced run method with multi-agent routing support"""
        try:
            self._initialize_session(query)
            
            # Set initial agent
            self.routing_context.current_agent = initial_agent
            
            # Always start with perception unless explicitly overridden
            if initial_agent == AgentType.PERCEPTION:
                return await self._run_perception_workflow(query)
            else:
                return await self._run_custom_workflow(query, initial_agent)
                
        except Exception as e:
            self.logger.error(f"Enhanced agent loop failed: {str(e)}")
            return await self._handle_critical_failure(str(e))
    
    async def _run_perception_workflow(self, query: str):
        """Enhanced perception workflow with routing support"""
        await self._run_initial_perception()

        if self._should_early_exit():
            return await self._summarize()

        # Let perception decide the next route
        route = self.p_out.get("route")
        if route == Route.BROWSER:
            # Initialize browser agent when needed
            if not self.browser_agent:
                self.browser_agent = await create_browser_agent(self.session_id, self.multi_mcp)
            return await self._run_browser_workflow(query)
        elif route == Route.DECISION:
            return await self._run_decision_loop()
        elif route == Route.SUMMARIZE:
            return await self._summarize()
        else:
            log_error("🚩 Invalid perception route. Exiting.")
            return "Invalid route from perception"
    
    async def _run_browser_workflow(self, query: str):
        """Run workflow starting with browser agent"""
        try:
            # self.logger.info("Starting browser-first workflow")
            
            # Initialize browser agent if needed
            if not self.browser_agent:
                self.browser_agent = await create_browser_agent(self.session_id)
                
            # Get browser context
            
            # Parse query into tasks using LLM
            tasks = await self.browser.generate_tasks(query)
            
            results = []
            error = None
            for task in tasks:
                # Skip error-only tasks
                if "error" in task and len(task) == 1:
                    error = task["error"]
                    continue
                
                # Execute task with automatic routing
                task["return_to"] = "perception"
                result = await self.browser_agent.execute_task(task)
                results.append(result)
                
                if task.get("action") == "get_session_snapshot":
                    self.browser.session_snapshot = result['mcp_result'].content[0].text
                
                # Try fallback on failure
                if not result.get("success") and "fallback" in task:
                    self.logger.info(f"Task failed, trying fallback")
                    task["fallback"]["return_to"] = "perception"
                    fallback_result = await self.browser_agent.execute_task(task["fallback"])
                    results.append(fallback_result)
                    if fallback_result.get("success"):
                        continue
                
            self.browser.results.extend(results)
            
            # Return control based on results
            if error or not results:
                return await self._return_to_perception([{
                    "success": False,
                    "error": error or "No valid browser tasks",
                    "agent": "browser"
                }])
            elif all(r.get("success") for r in results):
                # Success - continue with decision phase
                return await self._return_to_perception(results)
            else:
                # Some tasks failed - return to perception for reanalysis
                return await self._return_to_perception(results)
            
        except Exception as e:
            return await self._handle_agent_error("browser", str(e))
    
    async def _run_custom_workflow(self, query: str, initial_agent: AgentType):
        """Run custom workflow starting with specified agent"""
        try:
            self.logger.info(f"Starting custom workflow with {initial_agent}")
            
            # Route to the specified initial agent
            context = {"query": query}
            result = await self._route_to_agent(initial_agent, context)
            
            # Handle both dict and string results
            if isinstance(result, dict):
                # Continue with standard flow based on results
                if result.get("success"):
                    # If successful, continue to decision or summarization
                    if self.p_out and self.p_out.get("route") == Route.DECISION:
                        return await self._run_decision_loop()
                    else:
                        return await self._summarize()
                else:
                    return await self._handle_agent_error(str(initial_agent), result.get("error", "Unknown error"))
            else:
                # Handle string results - assume success
                return result
                
        except Exception as e:
            return await self._handle_agent_error("custom_workflow", str(e))
    
    async def _route_to_agent(self, target_agent: AgentType, context: Dict[str, Any]):
        """Generic agent routing method"""
        try:
            self.routing_context.push_agent(target_agent, self.routing_context.current_agent)
            
            if target_agent == AgentType.BROWSER:
                return await self._execute_browser_tasks(context)
            elif target_agent == AgentType.PERCEPTION:
                return await self._execute_perception_tasks(context)
            elif target_agent == AgentType.DECISION:
                return await self._execute_decision_tasks(context)
            elif target_agent == AgentType.EXECUTION:
                return await self._execute_execution_tasks(context)
            else:
                raise ValueError(f"Unknown target agent: {target_agent}")
                
        except Exception as e:
            return await self._handle_agent_error(str(target_agent), str(e))
    
    async def _extract_browser_tasks(self, query: str, error: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract browser-specific tasks from query using LLM

        Args:
            query (str): The user's query
            error (str, optional): Previous error for retry scenarios

        Returns:
            List[Dict[str, Any]]: List of browser tasks to execute
        """
        try:
            # Initialize task generator if needed
            if not hasattr(self, 'browser'):
                self.browser = Browser("prompts/browser_prompt.txt")
            
            # Get current browser context
            context = {}
            if self.browser_agent:
                context = {
                    "session_id": self.session_id,
                    "current_url": getattr(self.browser_agent, "current_url", None),
                    "last_action": getattr(self.browser_agent, "last_action", None),
                }
            
            # Parse query into tasks using LLM
            tasks = await self.browser.generate_tasks(query, context, error)
            
            if not isinstance(tasks, list):
                tasks = [tasks]
            
            # Ensure all tasks are dictionaries
            typed_tasks: List[Dict[str, Any]] = []
            for task in tasks:
                if isinstance(task, dict):
                    typed_tasks.append(task)
                else:
                    typed_tasks.append({"error": str(task)})
            
            return typed_tasks
            
        except Exception as e:
            self.logger.error(f"Failed to parse browser tasks: {e}")
            # Fallback to basic element detection
            return [{
                "action": "get_elements",
                "parameters": {"strict_mode": True, "viewport_mode": "visible"}
            }]
    
    # async def _requires_browser_operations(self, query: str) -> bool:
    #     """Check if query requires browser operations by attempting to parse it"""
    #     try:
    #         if not hasattr(self, 'browser_parser'):
    #             from agent.browser_parser import BrowserParser
    #             self.browser_parser = BrowserParser("prompts/browser_prompt.txt")
                
    #         # Try to parse into browser tasks
    #         tasks = await self.browser_parser.parse_to_tasks(query)
            
    #         # If we got valid tasks (not just error), it's a browser operation
    #         return any(not ("error" in task and len(task) == 1) for task in tasks)
            
    #     except Exception as e:
    #         self.logger.warning(f"Failed to check browser operations: {e}")
    #         # Fallback to keyword matching
    #         browser_keywords = [
    #             "click", "navigate", "browser", "webpage", "website", "url",
    #             "button", "link", "form", "input", "extract", "screenshot"
    #         ]
    #         return any(keyword in query.lower() for keyword in browser_keywords)
    
    async def _continue_with_decision(self, browser_results: List[Dict[str, Any]]):
        """Continue with decision phase after successful browser operations"""
        try:
            # first make these JSON-safe
            browser_results = self._serialize_browser_results(browser_results)
            
            # Update context with browser results
            self.ctx.add_browser_results(browser_results)
            
            # Run decision with enhanced context
            return await self._run_decision_loop()
            
        except Exception as e:
            return await self._handle_agent_error("decision_continuation", str(e))
        
    def _serialize_browser_results(self, browser_results: List[Any]) -> List[Dict[str, Any]]:
        """Convert CallToolResult or other objects to JSON-serializable dicts."""
        serial_results: List[Dict[str, Any]] = []
        for r in browser_results:
            # if it’s already a dict, use it directly
            if isinstance(r, dict):
                d = r.copy()
            # else if it has a __dict__, shallow-copy that
            elif hasattr(r, "__dict__"):
                d = r.__dict__.copy()
            # otherwise stringify the whole object
            else:
                d = {"result": str(r)}

            # coerce any non-serializable field to string
            for k, v in list(d.items()):
                try:
                    json.dumps(v)
                except (TypeError, ValueError):
                    d[k] = str(v)

            serial_results.append(d)
        return serial_results
    
    async def _return_to_perception(self, browser_results: List[Dict[str, Any]]):
        """Return to perception with browser results for reanalysis"""
        try:
            # first make these JSON-safe
            browser_results = self._serialize_browser_results(browser_results)
            # Update context with browser results (including failures)
            self.ctx.add_browser_results(browser_results)
            
            # Re-run perception with updated context
            p_input = build_perception_input(self.query, self.memory, self.ctx, snapshot_type="browser_feedback")
            self.p_out = await self.perception.run(p_input, session=self.session)
            
            # Process new perception output
            if self.p_out.get("route") == Route.DECISION:
                return await self._run_decision_loop()
            elif self.p_out.get("route") == Route.BROWSER:
            # Initialize browser agent when needed
                if not self.browser_agent:
                    self.browser_agent = await create_browser_agent(self.session_id, self.multi_mcp)
                return await self._run_browser_workflow(self.query)
            else:
                return await self._summarize()
                
        except Exception as e:
            return await self._handle_agent_error("perception_return", str(e))
    
    async def _execute_browser_tasks(self, context: Dict[str, Any]):
        """Execute browser-specific tasks"""
        if not self.browser_agent:
            self.browser_agent = await create_browser_agent(self.session_id)
        
        query = context.get("query", "")
        error = context.get("error")
        
        # Get tasks using LLM parser
        tasks = await self._extract_browser_tasks(query, error)
        
        results = []
        for task in tasks:
            # Skip error messages
            if "error" in task and len(task) == 1:
                self.logger.warning(f"Skipping error task: {task['error']}")
                continue
                
            result = await self.browser_agent.execute_task(task)
            results.append(result)
            
            # If task failed and has fallback, try the fallback
            if not result.get("success") and "fallback" in task:
                self.logger.info(f"Task failed, trying fallback: {task['fallback']}")
                fallback_result = await self.browser_agent.execute_task(task["fallback"])
                results.append(fallback_result)
        
        return {"success": True, "results": results, "agent": "browser"}
    
    async def _execute_perception_tasks(self, context: Dict[str, Any]):
        """Execute perception tasks"""
        p_input = build_perception_input(
            context.get("query", ""), 
            self.memory, 
            self.ctx, 
            snapshot_type="routed_perception"        )
        result = await self.perception.run(p_input, session=self.session)
        return {"success": True, "results": result, "agent": "perception"}
    
    async def _execute_decision_tasks(self, context: Dict[str, Any]):
        """Execute decision tasks"""
        # Ensure we have perception output for decision input
        p_out = getattr(self, 'p_out', {})
        if not p_out:
            # If no perception output available, create minimal one
            p_out = {"route": "decision", "result_requirement": context.get("query", "")}
            
        d_input = build_decision_input(
            self.ctx,                  # ctx (first parameter)
            context.get("query", ""),  # query (second parameter)
            p_out,                     # p_out (third parameter)
            self.strategy              # strategy (fourth parameter)
        )
        result = await self.decision.run(d_input, session=self.session)
        return {"success": True, "results": result, "agent": "decision"}
    
    async def _execute_execution_tasks(self, context: Dict[str, Any]):
        """Execute execution tasks"""
        # This would integrate with the existing execution system
        return {"success": True, "results": "Execution completed", "agent": "execution"}
    
    async def _summarize_results(self, results: List[Dict[str, Any]]):
        """Summarize results from multiple agents"""
        try:
            summary_input = {
                "query": self.query,
                "results": results,
                "session_context": self.ctx.get_context_snapshot()
            }
            
            summary = await self.summarizer.summarize(summary_input, self.ctx, self.p_out, session=self.session)
            return summary
            
        except Exception as e:
            return f"Summarization failed: {str(e)}"
    
    async def _handle_agent_error(self, agent_name: str, error: str):
        """Handle errors from specific agents"""
        self.logger.error(f"Agent {agent_name} failed: {error}")
        
        # Try to recover by routing to a different agent
        if agent_name == "browser" and self.routing_context.current_agent != AgentType.PERCEPTION:            
            return await self._route_to_agent(AgentType.PERCEPTION, {
                "query": self.query,
                "error_context": f"Browser agent failed: {error}"
            })
        
        return f"Agent {agent_name} failed: {error}"
    
    async def _handle_critical_failure(self, error: str):
        """Handle critical failures that can't be recovered"""
        self.logger.critical(f"Critical failure in enhanced agent loop: {error}")
        return f"Critical system failure: {error}"
    
    # Include all existing methods from the original AgentLoop
    def _initialize_session(self, query):
        """Initialize session (existing method)"""
        self.session_id = str(uuid.uuid4())
        self.query = query
        self.session = AgentSession(
            session_id=self.session_id,
            original_query=query
        )
        
        self.memory = MemorySearch().search_memory(query)
        # Remove load() call as MemorySearch doesn't have this method
        
        self.ctx = ContextManager(session_id=self.session_id, original_query=query)
        
        log_step(f"🚀 Starting Enhanced Agent Session: {self.session_id}")
        log_step(f"📝 Query: {query}")
        log_step(f"🎯 Strategy: {self.strategy}")
    
    async def _run_initial_perception(self):
        """Run initial perception (existing method)"""
        # Ensure context is properly initialized
        if not hasattr(self, 'ctx') or not hasattr(self.ctx, 'graph'):
            self.logger.error(f"Context not properly initialized. Type: {type(getattr(self, 'ctx', None))}")
            raise ValueError("ContextManager not properly initialized")
        
        p_input = build_perception_input(self.query, self.memory, self.ctx, snapshot_type="initial")
        self.p_out = await self.perception.run(p_input, session=self.session)
        self.ctx.attach_perception(StepType.ROOT, self.p_out)
        log_json_block("📌 Initial Perception", self.p_out)
    
    def _should_early_exit(self):
        """Check for early exit conditions (existing method)"""
        return (self.p_out.get("original_goal_achieved") or 
                self.p_out.get("route") == Route.SUMMARIZE)
    
    async def _summarize(self):
        """Generate summary (existing method)"""
        if hasattr(self, 'ctx') and self.ctx:
            summary_input = {
                "session_id": self.session_id,
                "query": self.query,
                "context": self.ctx.get_context_snapshot()
            }
            # import pdb; pdb.set_trace()
            return await self.summarizer.summarize(self.query, self.ctx, self.p_out, session=self.session)
        return "Summary unavailable - no context"
    
    async def _run_decision_loop(self):
        """Run decision loop with step execution (complete implementation)"""
        # Ensure we have perception output for decision input
        p_out = getattr(self, 'p_out', {})
        if not p_out:
            # If no perception output available, create minimal one
            p_out = {"route": "decision", "result_requirement": self.query}
        
        d_input = build_decision_input(
            self.ctx,       # ctx (first parameter)
            self.query,     # query (second parameter)  
            p_out,          # p_out (third parameter)
            self.strategy   # strategy (fourth parameter)
        )
        self.d_out = await self.decision.run(d_input, session=self.session)
        
        from utils.utils import log_json_block
        log_json_block("📌 Decision Output", self.d_out)
        
        if not self.d_out or not self.d_out.get("plan_graph"):
            self.status = "failed"
            return
            
        # Extract decision output components
        self.code_variants = self.d_out.get("code_variants", {})
        self.next_step_id = self.d_out.get("next_step_id", "ROOT")
        
        # Add steps from plan graph to context
        plan_graph = self.d_out.get("plan_graph", {})
        for node in plan_graph.get("nodes", []):
            from agent.agent_loop3 import StepType
            self.ctx.add_step(
                step_id=node["id"],
                description=node["description"],
                step_type=StepType.CODE,
                from_node=StepType.ROOT
            )
        # Execute the planned steps
        await self._execute_steps_loop()
    
    async def _execute_steps_loop(self):
        """Execute the planned steps in a loop"""
        from agent.agent_loop3 import StepExecutionTracker, StepType, Route
        from action.execute_step import execute_step_with_mode
        from perception.perception import build_perception_input
        from utils.utils import log_step, log_error, log_json_block
        
        tracker = StepExecutionTracker(max_steps=12, max_retries=5)
        AUTO_EXECUTION_MODE = "fallback"

        while tracker.should_continue():
            tracker.increment()
            log_step(f"🔁 Loop {tracker.tries} — Executing step {self.next_step_id}")

            if self.ctx.is_step_completed(self.next_step_id):
                log_step(f"✅ Step {self.next_step_id} already completed. Skipping.")
                self.next_step_id = self._pick_next_step(self.ctx)
                continue

            retry_step_id = tracker.retry_step_id(self.next_step_id)
            success = await execute_step_with_mode(
                retry_step_id,
                self.code_variants,
                self.ctx,
                AUTO_EXECUTION_MODE,
                self.session,
                self.multi_mcp
            )

            if not success:
                self.ctx.mark_step_failed(self.next_step_id, "All fallback variants failed")
                tracker.record_failure(self.next_step_id)

                if tracker.has_exceeded_retries(self.next_step_id):
                    if self.next_step_id == StepType.ROOT:
                        if tracker.register_root_failure():
                            log_error("🚨 ROOT failed too many times. Halting execution.")
                            return
                    else:
                        log_error(f"⚠️ Step {self.next_step_id} failed too many times. Forcing replan.")
                        self.next_step_id = StepType.ROOT
                continue

            self.ctx.mark_step_completed(self.next_step_id)

            # 🔍 Perception after execution
            p_input = build_perception_input(self.query, self.memory, self.ctx, snapshot_type="step_result")
            self.p_out = await self.perception.run(p_input, session=self.session)

            self.ctx.attach_perception(self.next_step_id, self.p_out)
            log_json_block(f"📌 Perception output ({self.next_step_id})", self.p_out)
            self.ctx._print_graph(depth=3)

            if self.p_out.get("original_goal_achieved") or self.p_out.get("route") == Route.SUMMARIZE:
                self.status = "success"
                self.final_output = await self._summarize()
                return

            if self.p_out.get("route") != Route.DECISION:
                log_error("🚩 Invalid route from perception. Exiting.")
                return

            # 🔁 Decision again
            d_input = build_decision_input(self.ctx, self.query, self.p_out, self.strategy)
            d_out = await self.decision.run(d_input, session=self.session)

            log_json_block(f"📌 Decision Output ({tracker.tries})", d_out)

            self.next_step_id = d_out["next_step_id"]
            self.code_variants = d_out["code_variants"]
            plan_graph = d_out["plan_graph"]
            self.update_plan_graph(self.ctx, plan_graph, self.next_step_id)
    
    def _pick_next_step(self, ctx) -> str:
        """Pick the next pending step from the context graph"""
        from agent.agent_loop3 import StepType
        for node_id in ctx.graph.nodes:
            node = ctx.graph.nodes[node_id]["data"]
            if node.status == "pending":
                return node.index
        return StepType.ROOT
    
    def update_plan_graph(self, ctx, plan_graph, from_step_id):
        """Update the plan graph with new nodes"""
        from agent.agent_loop3 import StepType
        for node in plan_graph["nodes"]:
            step_id = node["id"]
            if step_id in ctx.graph.nodes:
                existing = ctx.graph.nodes[step_id]["data"]
                if existing.status != "pending":
                    continue
            ctx.add_step(step_id, description=node["description"], step_type=StepType.CODE, from_node=from_step_id)
    
    async def _handle_failure(self):
        """Handle failure (existing method)"""
        return "Task execution failed"
    
    async def _retry_with_timeout(self, task: Dict[str, Any], timeout_ms: int = 5000) -> Dict[str, Any]:
        """Retry a task with timeout and exponential backoff"""
        try:
            start_time = time.time()
            backoff = 100  # Start with 100ms
            max_backoff = 1000  # Cap at 1 second
            
            # Ensure browser agent is initialized
            if not self.browser_agent:
                self.browser_agent = await create_browser_agent(self.session_id)
            
            while (time.time() - start_time) * 1000 < timeout_ms:
                result = await self.browser_agent.execute_task(task)
                if result.get("success"):
                    return result
                
                # Exponential backoff
                await asyncio.sleep(backoff / 1000)  # Convert to seconds
                backoff = min(backoff * 2, max_backoff)
            
            return {
                "success": False,
                "error": f"Task timed out after {timeout_ms}ms",
                "agent": "browser"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Retry failed: {str(e)}",
                "agent": "browser"
            }
            
    # async def _handle_browser_error(self, error: str, task: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    #     """Handle browser agent errors with automatic recovery"""
    #     self.logger.error(f"Browser error: {error}")
        
    #     if not task:
    #         return {
    #             "success": False,
    #             "error": error,
    #             "agent": "browser"
    #         }
            
    #     try:
    #         # Ensure browser agent is initialized
    #         if not self.browser_agent:
    #             self.browser_agent = await create_browser_agent(self.session_id)
            
    #         # Extract key info
    #         action = task.get("action", "").lower()
    #         parameters = task.get("parameters", {})
            
    #         # Handle specific error cases
    #         if "element not found" in error.lower() or "no such element" in error.lower():
    #             # Try getting elements again with extended viewport
    #             get_elements_task = {
    #                 "action": "get_elements",
    #                 "parameters": {"strict_mode": False, "viewport_mode": "all"}
    #             }
    #             return await self.browser_agent.execute_task(get_elements_task)
                
    #         elif "timeout" in error.lower():
    #             # Retry with extended timeout
    #             return await self._retry_with_timeout(task, timeout_ms=10000)
                
    #         elif "navigation" in error.lower():
    #             # Try navigation with fallback options
    #             if action in ["open_tab", "navigate"]:
    #                 url = parameters.get("url", "")
    #                 if url:
    #                     # Try alternate URL formations
    #                     if not url.startswith(("http://", "https://")):
    #                         alt_task = {**task, "parameters": {"url": f"https://{url}"}}
    #                         return await self.browser_agent.execute_task(alt_task)
                
    #         # Default error response
    #         return {
    #             "success": False,
    #             "error": error,
    #             "agent": "browser",
    #             "recovery_attempted": True
    #         }
            
    #     except Exception as e:
    #         return {
    #             "success": False,
    #             "error": f"Error recovery failed: {str(e)}",
    #             "agent": "browser"
    #         }
