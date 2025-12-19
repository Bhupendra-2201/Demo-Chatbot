
"""
Core Agent Framework

Implements a lightweight agent class that wraps Bedrock's Converse API
to handle tool use cycles, memory, and orchestration.
"""

from typing import List, Dict, Any, Callable
import json
import logging
import boto3
from botocore.exceptions import ClientError

import logging
import inspect
import json
import boto3
from typing import List, Dict, Any, Callable, Optional

# Initialize Bedrock Runtime
try:
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
except Exception:
    bedrock = None  # robust handling if run without creds

LOG = logging.getLogger(__name__)

class Agent:
    def __init__(self, name: str, model_id: str, instructions: str, tools: List[Callable] = None):
        self.name = name
        self.model_id = model_id
        self.instructions = instructions
        self.tools = tools or []
        self.tool_map = {func.__name__: func for func in self.tools}
        self.tool_config = self._generate_tool_config()

    def _generate_tool_config(self):
        """Generates Bedrock tool configuration from python functions."""
        if not self.tools:
            return None
            
        tool_specs = []
        for func in self.tools:
            # Basic introspection (could be improved with Pydantic)
            sig = inspect.signature(func)
            doc = func.__doc__ or "No description."
            
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self': continue
                
                param_type = "string" # Default
                if param.annotation == int: param_type = "integer"
                elif param.annotation == bool: param_type = "boolean"
                elif param.annotation == list: param_type = "array"
                elif param.annotation == dict: param_type = "object"
                
                properties[param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}" 
                }
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
            
            tool_specs.append({
                "toolSpec": {
                    "name": func.__name__,
                    "description": doc.strip(),
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    }
                }
            })
            
        return {"tools": tool_specs}

    def run(self, message: str, history: List[Dict] = None, stop_after_tool: bool = False):
        """Converses with the LLM, handling tool calls loop."""
        if not bedrock:
            raise RuntimeError("Bedrock client not initialized")
            
        # Clean history to match Bedrock Converse API format
        clean_history = []
        if history:
            for msg in history:
                clean_msg = {
                    "role": msg.get("role"),
                    "content": msg.get("content")
                }
                # Ensure content is a list of dicts or handle text conversion
                if isinstance(clean_msg["content"], str):
                     clean_msg["content"] = [{"text": clean_msg["content"]}]
                
                clean_history.append(clean_msg)
        
        messages = clean_history
        
        # Handle models that don't support system prompts (e.g. Titan)
        if "amazon.titan" in self.model_id:
            # Prepend system prompt to user message
            final_message = f"System Instructions:\n{self.instructions}\n\nUser Query:\n{message}"
            system_prompts = None
        else:
            final_message = message
            system_prompts = [{"text": self.instructions}]
            
        # Add user message
        messages.append({"role": "user", "content": [{"text": final_message}]})
        
        # Inference Loop (Max 5 turns)
        for _ in range(5):
            LOG.info(f"Agent {self.name} invoking model...")
            
            converse_args = {
                "modelId": self.model_id,
                "messages": messages,
                "inferenceConfig": {"maxTokens": 2000, "temperature": 0}
            }
            
            if system_prompts:
                converse_args["system"] = system_prompts
            if self.tool_config:
                converse_args["toolConfig"] = self.tool_config

            # === SMART MODE DEBUG: EXACT PROMPT & OUTPUT ===
            LOG.info(f"--- [Step {_+1}] Bedrock Request ---")
            if "system" in converse_args:
                LOG.info(f"System Prompt:\n{converse_args['system'][0]['text']}")
            
            # Log Tool Config (Simplified)
            if self.tool_config:
                 tool_names = [t['toolSpec']['name'] for t in self.tool_config['tools']]
                 LOG.info(f"Tools Available: {tool_names}")
            
            LOG.info(f"User Message: {messages[-1]['content'][0].get('text', '...tool_result...')}")
            
            # Invoke
            response = bedrock.converse(**converse_args)
            output_message = response['output']['message']
            
            # Log Response
            LOG.info(f"--- [Step {_+1}] Bedrock Response ---")
            LOG.info(f"Raw Output: {json.dumps(output_message, indent=2)}")
            # ===============================================
            
            messages.append(output_message) # Add assistant response to history
            
            content_blocks = output_message['content']
            
            # Check for tool use
            tool_requests = [b for b in content_blocks if 'toolUse' in b]
            
            if not tool_requests:
                # No tools used, return final text
                final_text = next((b['text'] for b in content_blocks if 'text' in b), "")
                return final_text
            
            # Handle Tool Execution
            tool_results = []
            for req in tool_requests:
                tool_use = req['toolUse']
                tool_name = tool_use['name']
                tool_id = tool_use['toolUseId']
                inputs = tool_use['input']
                
                LOG.info(f"Agent {self.name} calling tool: {tool_name} with {inputs}")
                
                try:
                    func = self.tool_map.get(tool_name)
                    if not func:
                        raise ValueError(f"Tool {tool_name} not found")
                        
                    # Call the actual python function
                    # Handle 'self' if it's a bound method? 
                    # The tool list usually contains bound methods if added from self.
                    
                    result = func(**inputs)
                    result_text = json.dumps(result, default=str)
                    
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_id,
                            "content": [{"text": result_text}],
                            "status": "success"
                        }
                    })
                except Exception as e:
                    LOG.error(f"Tool execution failed: {e}")
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_id,
                            "content": [{"text": str(e)}],
                            "status": "error"
                        }
                    })
            
            # Append tool results to messages
            messages.append({"role": "user", "content": tool_results})
            
            # OPTIMIZATION: If configured, return tool output immediately (skip synthesis)
            if stop_after_tool and tool_results:
                # Return the concatenated text of all tool results
                # We assume the caller wants the raw data from the tool
                final_output = []
                for res in tool_results:
                    for content in res.get("toolResult", {}).get("content", []):
                        if "text" in content:
                            final_output.append(content["text"])
                
                return "\n\n".join(final_output)

        return "Agent loop limit reached."

def tool(func):
    """Marker decorator."""
    func._is_tool = True
    return func
