# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example math expression tool for Letta React Agent Loop.

This demonstrates how to create a simple tool that can be used with the
Letta React Agent Loop implementation.
"""

import json
from typing import Any, Dict

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions safely.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
    
    Returns:
        The result of the calculation as a string
    """
    try:
        # Simple evaluation for basic math operations
        # In a real implementation, you'd want more robust expression parsing
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def search_web(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: The search query
    
    Returns:
        Search results as a string
    """
    # This is a placeholder implementation
    # In a real implementation, you would integrate with a web search API
    return f"Search results for '{query}': This is a placeholder response. In a real implementation, this would return actual search results from a web search API."


# List of available tools for the Letta React Agent Loop
LETTA_TOOLS = [calculator, search_web]
