# browserAgent.py - Specialized browser automation agent
"""
BrowserAgent: Specialized agent for browser-specific operations with multi-directional routing support

This agent handles:
- Browser automation tasks with proper element access
- Multi-directional routing between agents
- Smart error handling for browser operations
- Integration with existing Perception/Decision/Execution modules
"""

# Module level imports
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

# Global logger setup  
logger = logging.getLogger("BrowserAgent")

# Convenience functions for easy integration
async def create_browser_agent(
    session_id: Optional[str] = None,
    multi_mcp: Any = None
) -> 'BrowserAgent':
    """Create and initialize a BrowserAgent instance"""
    return BrowserAgent(session_id=session_id, multi_mcp=multi_mcp)

async def execute_browser_task(
    task: Dict[str, Any],
    session_id: Optional[str] = None,
    return_to: Optional[str] = None,
    multi_mcp: Any = None
) -> Dict[str, Any]:
    """Convenience function to execute a single browser task"""
    agent = await create_browser_agent(session_id, multi_mcp)
    return await agent.execute_task(task, return_to)

# Module exports - must be defined after the classes and functions
__all__: List[str] = [
    "BrowserAgent",
    "create_browser_agent",
    "execute_browser_task"
]

class BrowserAgent:
    """Specialized browser agent with multi-directional routing capabilities"""
    
    def __init__(self, session_id: Optional[str] = None, multi_mcp: Any = None):
        """Initialize BrowserAgent with session management"""
        self.session_id = session_id or "browser_agent_session"
        self.multi_mcp = multi_mcp
        
        # Routing configuration
        self.current_agent = "browserAgent"
        self.return_to: Optional[str] = None
        self.routing_stack: List[str] = []
        
        # Browser-specific configurations
        self.browser_context = {
            "structured_output": True,  # Always use structured output
            "strict_mode": True,        # Default to strict filtering
            "viewport_mode": "visible"  # Default to visible elements
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"BrowserAgent.{self.session_id}")
        
        if not multi_mcp:
            self.logger.warning("BrowserAgent initialized without MCP executor - browser operations will fail!")

    async def _execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool with proper error handling"""
        try:
            if not self.multi_mcp:
                raise ValueError("MCP executor not initialized")
            
            # Execute the tool directly through MultiMCP
            result = await self.multi_mcp.call_tool(tool_name, parameters)
            return result
        except Exception as e:
            self.logger.error(f"MCP tool execution failed: {e}")
            raise

    async def execute_task(self, task: Dict[str, Any], return_to: Optional[str] = None) -> Dict[str, Any]:
        """Execute a browser-specific task with routing support"""
        try:
            self.return_to = return_to
            
            # Extract task information
            action = task.get("action", "").lower()
            parameters = task.get("parameters", {})
            
            self.logger.info(f"Executing browser task: {action}")
            
            # Route task based on action type
            if action in ["get_elements", "get_interactive_elements"]:
                return await self._handle_get_elements_task(action, parameters)
            elif action in ["click", "click_element", "click_element_by_index", "click_by_text"]:
                return await self._handle_click_task(parameters)
            elif action in ["input", "input_text", "type_text"]:
                return await self._handle_input_task(parameters)
            elif action in ["navigate", "go_to_url", "navigate_to_url", "open_tab"]:
                return await self._handle_navigation_task(action, parameters)
            elif action in ["extract", "get_content", "get_page_structure", "extract_page_content"]:
                return await self._handle_extraction_task(action, parameters)
            elif action in ["route_to", "delegate_to"]:
                return await self.route_to_agent(parameters.get("agent"), task)
            else:
                return await self._handle_generic_task(action, parameters)
            
        except Exception as e:
            self.logger.error(f"Error executing browser task: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent": self.current_agent
            }
    
    async def _handle_get_elements_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle element retrieval tasks with proper field access"""
        try:
            fixed_parameters = {
                "structured_output": True,
                "strict_mode": parameters.get("strict_mode", True),
                "visible_only": parameters.get("visible_only", True),
                "viewport_mode": parameters.get("viewport_mode", "visible")
            }
            
            result = await self._execute_mcp_tool("get_interactive_elements", fixed_parameters)
            # print(result)
            return {
                "success": True,
                "elements": result.content[0].text,
                "message": f"Found interactive elements",
                "agent": self.current_agent,
                "structured_output": True,
                "mcp_result": result
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get elements: {str(e)}",
                "agent": self.current_agent
            }

    async def _handle_click_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle click tasks with proper element ID handling"""
        try:
            element_id = parameters.get("index")
            text_to_find = parameters.get("text")
            
            if element_id is not None:
                result = await self._execute_mcp_tool("click_element_by_index", {"index": element_id})
                return {
                    "success": True,
                    "message": f"Clicked element with ID: {element_id}",
                    "element_id": element_id,
                    "agent": self.current_agent,
                    "mcp_result": result
                }
            elif text_to_find:
                # First get elements to find by text
                elements_result = await self._handle_get_elements_task("get_elements", {})
                if not elements_result.get("success"):
                    return elements_result
                
                # Find element with matching text
                import json
                elements_result = json.loads(elements_result.get("elements", "[]"))
                data = elements_result.get("nav", []) + elements_result.get("forms", [])
                for element in data:
                    if text_to_find.lower() in element.get("desc", "").lower():
                        result = await self._execute_mcp_tool("click_element_by_index", {"index": element["id"]})
                        return {
                            "success": True,
                            "message": f"Clicked element with text '{text_to_find}' (ID: {element['id']})",
                            "element_id": element["id"],
                            "agent": self.current_agent,
                            "mcp_result": result
                        }
                
                return {
                    "success": False,
                    "error": f"Element with text '{text_to_find}' not found",
                    "agent": self.current_agent
                }
            else:
                return {
                    "success": False,
                    "error": "No element ID provided",
                    "agent": self.current_agent
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to click element: {str(e)}",
                "agent": self.current_agent
            }
    
    async def _handle_input_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multiple input/text entry tasks"""
        try:
            element_id = parameters.get("index")
            text = parameters.get("text")
            result = None
            if element_id is not None:
                result = await self._execute_mcp_tool("input_text", {"index": element_id, "text": text})
                if not result.isError:
                    return {
                        "success": True,
                        "message": f"Entered '{text}' text in field with index: {element_id}",
                        "element_id": element_id,
                        "agent": self.current_agent,
                        "mcp_result": result
                    }
            return {
                "success": False,
                "error": "No element ID provided for input or the tool execution failed",
                "agent": self.current_agent,
                "mcp_result": result
            }
            # elif parameters:
            #     # First get elements to find by text
            #     elements_result = await self._handle_get_elements_task("get_elements", {})
            #     if not elements_result.get("success"):
            #         return elements_result
                
            #     # Find element with matching text
            #     import json
            #     elements_result = json.loads(elements_result.get("elements", "[]"))
            #     for param, value in parameters.items():
            #         for element in elements_result.get("forms", []):
            #             if param.lower() in element.get("desc", "").lower():
            #                 result = await self._execute_mcp_tool("input_text", {"index": element["id"], "text": value})
            # return {
            #     "success": True,
            #     "message": f"Input parameters '{json.dumps(parameters)}' to webpage",
            #     "element_id": element_id,
            #     "text": json.dumps(parameters),
            #     "agent": self.current_agent,
            #     "mcp_result": result
            # }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to input text: {str(e)}",
                "agent": self.current_agent
            }
    
    async def _handle_navigation_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle navigation tasks"""
        try:
            url = parameters.get("url")
            
            if not url:
                return {
                    "success": False,
                    "error": "No URL provided for navigation",
                    "agent": self.current_agent
                }

            # Only try to open a new tab if requested
            if parameters.get("new_tab", True):
                try:
                    await self._execute_mcp_tool("open_tab", {})
                except Exception as e:
                    self.logger.error(f"Failed to open new tab (continuing with current tab): {e}")
            
            # Navigate to URL
            result = await self._execute_mcp_tool("go_to_url", {"url": url})
            return {
                "success": True,
                "message": f"Navigated to: {url}",
                "url": url,
                "agent": self.current_agent,
                "mcp_result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to navigate: {str(e)}",
                "agent": self.current_agent
            }
    
    async def _handle_extraction_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content extraction tasks"""
        try:
            format_type = parameters.get("format", "markdown")
            
            if format_type == "structure":
                result = await self._execute_mcp_tool("get_enhanced_page_structure", {})
            else:
                result = await self._execute_mcp_tool("get_comprehensive_markdown", {})
            
            self.logger.info(f"Extracted content in {format_type} format")
            self.logger.info(f"Extraction result: {result}")  # Log first 100 chars
            
            return {
                "success": True,
                "content": result.content[0].text,
                "format": format_type,
                "agent": self.current_agent,
                "mcp_result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to extract content: {str(e)}",
                "agent": self.current_agent
            }
    
    async def _handle_generic_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic browser tasks"""
        try:
            result = await self._execute_mcp_tool(action, parameters)
            return {
                "success": True,
                "message": f"Executed browser action: {action}",
                "action": action,
                "parameters": parameters,
                "agent": self.current_agent,
                "mcp_result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to execute {action}: {str(e)}",
                "agent": self.current_agent
            }
    
    # Routing methods for multi-directional agent communication
    async def route_to_agent(self, target_agent: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Route task to another agent and manage routing stack"""
        if not target_agent:
            return {
                "success": False,
                "error": "No target agent specified",
                "agent": self.current_agent
            }

        try:
            # Add current agent to routing stack
            self.routing_stack.append(self.current_agent)
            
            # For now, return a mock response since we can't import other agents
            if target_agent == "perception":
                result = {
                    "success": True,
                    "message": "Routed to perception agent",
                    "agent": "perception",
                    "routed_from": self.current_agent
                }
            elif target_agent == "decision":
                result = {
                    "success": True,
                    "message": "Routed to decision agent",
                    "agent": "decision",
                    "routed_from": self.current_agent
                }
            elif target_agent == "execution":
                result = {
                    "success": True,
                    "message": "Routed to execution agent",
                    "agent": "execution",
                    "routed_from": self.current_agent
                }
            else:
                result = {
                    "success": False,
                    "error": f"Unknown target agent: {target_agent}",
                    "agent": self.current_agent
                }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to route to {target_agent}: {str(e)}",
                "agent": self.current_agent
            }
    
    async def return_control(self) -> Optional[str]:
        """Return control to the previous agent in the routing stack"""
        if self.routing_stack:
            return self.routing_stack.pop()
        return self.return_to
    
    def get_routing_context(self) -> Dict[str, Any]:
        """Get current routing context for debugging/monitoring"""
        return {
            "current_agent": self.current_agent,
            "return_to": self.return_to,
            "routing_stack": list(self.routing_stack),  # Create a copy
            "session_id": self.session_id
        }