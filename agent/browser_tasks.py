"""Browser task generation and execution utilities"""
from pathlib import Path
import json
import logging
import sys
from typing import Dict, Any, Iterable, List, Optional
from browserMCP.mcp_tools import get_tools

from agent.model_manager import ModelManager

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('browser_agent.log')
    ]
)

class Browser:
    """Generates browser tasks from natural language using LLM"""
    
    def __init__(self, prompt_path: str = "prompts/browser_prompt.txt"):
        """Initialize the task generator
        
        Args:
            prompt_path (str): Path to the browser prompt template
        """
        self.prompt_path = prompt_path
        self.model = ModelManager()
        # Create logger with the class name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        self.session_snapshot = None  # Placeholder for session snapshot
        self.results = []  # Placeholder for results
        
    def _serialize_results(self, browser_results: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all values in the results dict are JSON-serializable."""
        def _safe_serialize(value: Any) -> Any:
            try:
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                if isinstance(value, dict):
                    return {k: _safe_serialize(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [_safe_serialize(v) for v in value]
                return str(value)

        return {k: _safe_serialize(v) for k, v in browser_results.items()}

    def _add_fallback_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add fallback tasks for common failure scenarios"""
        enhanced_tasks = []
        
        for task in tasks:
            # Skip tasks that already have fallbacks
            if "fallback" in task:
                enhanced_tasks.append(task)
                continue
                
            action = task.get("action", "").lower()
            parameters = task.get("parameters", {})
            
            # Add fallbacks based on action type
            if action in ["click_element_by_index", "input_text"]:
                # For element interactions, fallback to finding elements
                task["fallback"] = {
                    "action": "get_elements",
                    "parameters": {
                        "strict_mode": True,
                        "viewport_mode": "visible"
                    }
                }
            elif action in ["open_tab", "navigate"]:
                # For navigation, fallback to browser history
                task["fallback"] = {
                    "action": "go_back",
                    "parameters": {}
                }
            elif action in ["scroll_down", "scroll_up"]:
                # For scrolling, fallback to smaller amount
                amount = parameters.get("amount", 500)
                task["fallback"] = {
                    "action": task["action"],
                    "parameters": {"amount": amount // 2}
                }
                
            enhanced_tasks.append(task)
            
        return enhanced_tasks

    async def generate_tasks(
        self,
        query: str,
        error: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate browser tasks from natural language query
        
        Args:
            query: The user's query
            context: Optional browser context (current URL, state etc)
            error: Optional previous error message
            
        Returns:
            List of browser task dictionaries
        """
        try:
            # Load prompt template
            prompt = Path(self.prompt_path).read_text(encoding="utf-8")
            self.logger.info(f"Using prompt template from {self.prompt_path}")
            
            # Build input context
            input_json = {
                "query": query,
                "history": self.results,
                "session_snapshot": self.session_snapshot,
                "error": error or ""
            }
            
            input_json = self._serialize_results(input_json)
            
            full_prompt = prompt.format(context = json.dumps(input_json, indent=2), tools = get_tools()).strip()
            
            self.logger.info(f"Generated prompt: {full_prompt}")
            
            # Get LLM response
            response = await self.model.generate_text(prompt=full_prompt)
            
            self.logger.info(f"LLM response: {response}")
            
            try:
                # Remove any leading/trailing whitespace and backticks
                response = response.strip().strip("`")
                
                # Remove "json" or "JSON" from the start of the response if present
                if response.lower().startswith("json"):
                    response = response[4:].strip()
                    
                tasks = json.loads(response).get("actions", [])
                
                if not isinstance(tasks, list):
                    tasks = [tasks]
                    
                # Add fallback tasks    
                tasks = self._add_fallback_tasks(tasks)
                
                return tasks
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response as JSON: {e}")
                return [{
                    "action": "get_elements",
                    "parameters": {"strict_mode": True, "viewport_mode": "visible"}
                }]
                
        except Exception as e:
            self.logger.error(f"Task generation failed: {e}")
            return [{
                "action": "get_elements",
                "parameters": {"strict_mode": True, "viewport_mode": "visible"}
            }]
