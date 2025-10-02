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
Letta React Agent Loop.

This implementation provides a React (Reasoning and Acting) agent loop using Letta client,
replacing LangGraph with Letta's native conversation management and tool execution.

Ref: https://docs.letta.com/
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from letta_client import AsyncLetta, MessageCreate
from letta_client.types import LettaResponse

from recipe.letta_agent.letta_chat_model import (
    LettaChatModel,
    MaxTokenExceededError,
    convert_to_agent_output,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO to see our debug messages

# Global tool cache to avoid recreating tools
_tool_cache = {}
_agent_cache = {}
_letta_client = None


async def retry_with_backoff(func, max_retries=5, base_delay=1.0, max_delay=60.0):
    """Retry a function with exponential backoff and jitter."""
    import random
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # Check if it's a connection limit error
            if "too many clients already" in str(e).lower() or "connection" in str(e).lower():
                # Exponential backoff with jitter to avoid thundering herd
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0.1, 0.5) * delay  # Add 10-50% jitter
                total_delay = delay + jitter
                
                logger.warning(f"Connection error on attempt {attempt + 1}, retrying in {total_delay:.2f}s: {e}")
                await asyncio.sleep(total_delay)
            else:
                # For other errors, don't retry
                raise e


class LettaConversationManager:
    """Manages conversation state and tool execution with Letta."""
    
    def __init__(self, client: AsyncLetta, agent_id: str, max_assistant_turns: int = 10):
        self.client = client
        self.agent_id = agent_id
        self.max_assistant_turns = max_assistant_turns
        self.conversation_messages = []
        self.assistant_turn_count = 0
    
    async def send_message(self, content: str, sampling_params: Dict[str, Any]) -> LettaResponse:
        """Send a message to the Letta agent and get response."""
        # Convert to Letta message format
        message = MessageCreate(role="user", content=content)
        
        async def _send_message():
            response = await self.client.agents.messages.create(
                agent_id=self.agent_id,
                messages=[message],
            )
            return response
        
        try:
            # Send message with retry logic
            response = await retry_with_backoff(_send_message, max_retries=8, base_delay=1.0, max_delay=60.0)
            
            # Track conversation state
            self.conversation_messages.append(message)
            self._process_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error sending message to Letta agent after retries: {e}")
            raise
    
    def _process_response(self, response: LettaResponse):
        """Process Letta response and update conversation state."""
        if response.messages:
            for msg in response.messages:
                if msg.message_type == "assistant_message":
                    self.assistant_turn_count += 1
                self.conversation_messages.append(msg)
    
    def should_continue(self) -> bool:
        """Check if conversation should continue based on turn limits."""
        return self.assistant_turn_count < self.max_assistant_turns
    
    def has_tool_calls(self, response: LettaResponse) -> bool:
        """Check if the response contains tool calls."""
        if not response.messages:
            return False
        
        for msg in response.messages:
            if msg.message_type == "tool_call_message" and msg.tool_call:
                return True
        return False


@register("letta_react_agent")
class ReactAgentLoop(AgentLoopBase):
    """React Agent Loop implementation using Letta client."""

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        """Initialize the class with shared resources."""
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ReactAgentLoop initialization", flush=True)

        # Initialize tools from config file
        from verl.tools.utils.tool_registry import initialize_tools_from_config
        
        cls.tokenizer = tokenizer
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_list = tool_list  # Store the list for tool registration
        print(f"Initialized tools: {cls.tools}", flush=True)

        # Initialize Letta client (singleton pattern)
        global _letta_client
        if _letta_client is None:
            _letta_client = cls._initialize_letta_client(config)
            print("Initialized Letta client", flush=True)
        cls.letta_client = _letta_client

    @classmethod
    def _initialize_letta_client(cls, config) -> AsyncLetta:
        """Initialize Letta client based on configuration."""
        # Get Letta configuration from config
        letta_config = getattr(config, 'letta', {})
        
        base_url = letta_config.get('base_url', 'http://localhost:8283')
        client = AsyncLetta(
            base_url=base_url,
        )
        return client

    @classmethod
    async def create_agent(cls, config) -> str:
        """Create a Letta agent with the configured tools and memory blocks."""
        # Get model configuration
        model_path = config.actor_rollout_ref.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        
        # Map Qwen models to compatible OpenAI models for Letta
        # Letta only supports OpenAI models, so we need to map to a compatible one
        if "Qwen" in model_name or "qwen" in model_name.lower():
            # Use a compatible OpenAI model for Qwen training
            # This is for the Letta agent interface only, not the actual training model
            letta_model = "openai/gpt-4o-mini"  # Use a cost-effective model for agent interface
        else:
            letta_model = f"openai/{model_name}"
        
        # Create tools using client.tools.create() method with caching and retry
        tool_names = []
        for i, tool in enumerate(cls.tool_list):
            tool_schema = tool.get_openai_tool_schema().model_dump()
            tool_name = tool_schema['function']['name']
            
            # Check if tool is already cached
            if tool_name in _tool_cache:
                logger.info(f"Using cached tool: {tool_name}")
                tool_names.append(tool_name)
                continue
            
            # Add small delay between tool creations to reduce server load
            if i > 0:
                await asyncio.sleep(0.5)
            
            # Create tool file content based on tool type
            if tool_name == "calc_gsm8k_reward":
                tool_file_path = "/home/ec2-user/letta-synthetic-data/verl/recipe/letta_agent/gsm8k_tool.py"
                with open(tool_file_path, "r") as f:
                    tool_source_code = f.read()
                
                async def create_tool():
                    return await cls.letta_client.tools.create(
                        source_code=tool_source_code
                    )
                
                try:
                    # Create the tool in Letta with retry logic
                    tool_obj = await retry_with_backoff(create_tool)
                    tool_names.append(tool_obj.name)
                    _tool_cache[tool_name] = tool_obj.name  # Cache the tool name
                    logger.info(f"Successfully created tool: {tool_obj.name}")
                except Exception as e:
                    if "duplicate key value violates unique constraint" in str(e) or "already exists" in str(e):
                        # Tool already exists, use the existing tool name
                        logger.info(f"Tool {tool_name} already exists, using existing tool")
                        tool_names.append(tool_name)
                        _tool_cache[tool_name] = tool_name  # Cache the tool name
                    else:
                        logger.error(f"Failed to create tool {tool_name} after retries: {e}")
                        # Skip this tool but continue with others
                        continue
            else:
                # For other tools, create a generic tool file
                # This is a simplified approach - you might want to create specific files for each tool
                logger.warning(f"Tool {tool_name} not implemented for Letta, skipping...")
        
        # Create memory blocks for the agent
        memory_blocks = [
            {
                "label": "persona",
                "value": "I am an AI assistant that can help with various tasks using available tools."
            },
            {
                "label": "human", 
                "value": "The user is working with a React agent loop system."
            }
        ]
        
        # Create the agent with retry logic
        async def _create_agent():
            return await cls.letta_client.agents.create(
                memory_blocks=memory_blocks,
                tools=tool_names,  # Pass tool names instead of schemas
                model=letta_model,
                embedding="openai/text-embedding-3-small"
            )
        
        agent = await retry_with_backoff(_create_agent, max_retries=10, base_delay=2.0, max_delay=120.0)
        
        return agent.id

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the React agent loop using Letta.

        Args:
            sampling_params: Sampling parameters for the model
            **kwargs: Additional arguments including raw_prompt

        Returns:
            AgentLoopOutput: The result of the agent loop execution
        """
        print("DEBUG: ReactAgentLoop.run() called", flush=True)
        logger.warning("DEBUG: ReactAgentLoop.run() called")
        messages = list(kwargs["raw_prompt"])
        print(f"DEBUG: messages count: {len(messages)}", flush=True)
        logger.warning(f"DEBUG: messages count: {len(messages)}")

        rollout = self.config.actor_rollout_ref.rollout
        
        # Create or get agent ID
        agent_id = await self._get_or_create_agent()
        
        # Initialize conversation manager
        conversation_manager = LettaConversationManager(
            client=self.letta_client,
            agent_id=agent_id,
            max_assistant_turns=rollout.multi_turn.max_assistant_turns
        )
        
        # Extract the user message from the prompt
        user_message = self._extract_user_message(messages)
        
        # Run the conversation loop
        all_messages = []
        max_iterations = rollout.multi_turn.max_assistant_turns or 10
        
        for iteration in range(max_iterations):
            print(f"DEBUG: Conversation iteration {iteration + 1}", flush=True)
            
            try:
                # Send message to Letta agent
                response = await conversation_manager.send_message(
                    user_message, 
                    sampling_params
                )
                
                # Process the response
                processed_messages = self._process_letta_response(response)
                all_messages.extend(processed_messages)
                
                # Check if we should continue
                if not conversation_manager.should_continue():
                    print("DEBUG: Max assistant turns reached", flush=True)
                    break
                
                if not conversation_manager.has_tool_calls(response):
                    print("DEBUG: No tool calls, conversation complete", flush=True)
                    break
                
                # For subsequent iterations, we don't send new user messages
                # Letta handles the tool execution internally
                user_message = ""
                
            except MaxTokenExceededError:
                print("DEBUG: Max tokens exceeded", flush=True)
                break
            except Exception as e:
                print(f"DEBUG: Error in conversation loop: {e}", flush=True)
                logger.error(f"Error in conversation loop: {e}")
                break

        print(f"DEBUG: Conversation completed with {len(all_messages)} messages", flush=True)
        
        # Convert to agent output format
        print("DEBUG: About to call convert_to_agent_output", flush=True)
        output = convert_to_agent_output(all_messages, rollout.response_length)
        print("DEBUG: convert_to_agent_output completed", flush=True)
        return output

    async def _get_or_create_agent(self) -> str:
        """Get or create a Letta agent for this session."""
        # Create a cache key based on configuration
        config_key = f"{self.config.actor_rollout_ref.model.path}_{len(self.tool_list)}"
        
        # Check if agent is already cached
        if config_key in _agent_cache:
            logger.info(f"Using cached agent: {config_key}")
            return _agent_cache[config_key]
        
        # Create new agent
        agent_id = await self.create_agent(self.config)
        _agent_cache[config_key] = agent_id
        logger.info(f"Cached new agent: {config_key} -> {agent_id}")
        return agent_id

    def _extract_user_message(self, messages: List[Any]) -> str:
        """Extract the user message from the prompt messages."""
        # Find the last human message
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'human':
                return message.content
            elif hasattr(message, 'role') and message.role == 'user':
                return message.content
        return ""

    def _process_letta_response(self, response: LettaResponse) -> List[Any]:
        """Process Letta response and convert to LangChain message format."""
        from langchain_core.messages import AIMessage, ToolMessage
        
        processed_messages = []
        
        if response.messages:
            for msg in response.messages:
                if msg.message_type == "assistant_message":
                    # Create AIMessage with content
                    ai_msg = AIMessage(content=msg.content or "")
                    processed_messages.append(ai_msg)
                    
                elif msg.message_type == "tool_call_message" and msg.tool_call:
                    # Create AIMessage with tool calls
                    tool_calls = []
                    if msg.tool_call:
                        tool_calls.append({
                            "name": msg.tool_call.name,
                            "args": msg.tool_call.arguments,
                            "id": str(msg.tool_call.id) if hasattr(msg.tool_call, 'id') else "1"
                        })
                    
                    ai_msg = AIMessage(
                        content="",
                        tool_calls=tool_calls
                    )
                    processed_messages.append(ai_msg)
                    
                elif msg.message_type == "tool_return_message" and msg.tool_return:
                    # Create ToolMessage
                    tool_msg = ToolMessage(
                        content=str(msg.tool_return),
                        tool_call_id=str(msg.tool_return.tool_call_id) if hasattr(msg.tool_return, 'tool_call_id') else "1"
                    )
                    processed_messages.append(tool_msg)
        
        return processed_messages
