# """
# Browser Parser - LLM-based parser for browser actions
# """
# from pathlib import Path
# import json
# from typing import List, Dict, Any, Optional
# from agent.model_manager import ModelManager

# class BrowserParser:
#     """Parser that uses LLM to convert natural language to browser actions"""
    
#     def __init__(self, prompt_path: str, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
#         """Initialize browser parser
        
#         Args:
#             prompt_path (str): Path to browser prompt template
#             api_key (Optional[str], optional): API key. Defaults to None.
#             model (str, optional): Model to use. Defaults to "gemini-2.0-flash".
#         """
#         self.prompt_path = prompt_path
#         self.model = ModelManager()

#     async def parse_to_tasks(self, query: str, error: str = "", context: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
#         """Parse natural language query into browser tasks
        
#         Args:
#             query (str): User's query
#             context (Dict[str, Any], optional): Browser context. Defaults to None.
#             error (str, optional): Previous error for retry scenarios. Defaults to None.
        
#         Returns:
#             List[Dict[str, Any]]: List of browser tasks to execute
#         """
#         prompt_template = Path(self.prompt_path).read_text(encoding="utf-8")
        
#         # Build input for LLM
#         input_json = {
#             "query": query,
#             "context": context or {},
#             "error": error
#         }

#         full_prompt = (
#             f"{prompt_template.strip()}\n\n"
#             "```json\n"
#             f"{json.dumps(input_json, indent=2)}\n"
#             "```"
#         )

#         try:
#             response = await self.model.generate_text(prompt=full_prompt)
            
#             # Parse response as JSON
#             tasks = json.loads(response)
#             if not isinstance(tasks, list):
#                 tasks = [tasks]
                
#             return tasks
            
#         except Exception as e:
#             return [{
#                 "error": f"Failed to parse browser tasks: {str(e)}",
#                 "action": "get_elements",
#                 "parameters": {"strict_mode": True, "viewport_mode": "visible"}
#             }]
