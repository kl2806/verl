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
similar to the LangGraph implementation but adapted for Letta's conversation management.

Ref: https://langchain-ai.github.io/langgraph/tutorials/workflows/
"""

from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from recipe.letta_agent.letta_chat_model import (
    LettaChatModel,
    MaxTokenExceededError,
    convert_to_agent_output,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register


async def call_model(state: MessagesState, config: RunnableConfig):
    """Call the Letta model to generate a response."""
    model = config["configurable"]["model"]
    sampling_params = config["configurable"]["sampling_params"]
    try:
        message = await model.ainvoke(state["messages"], sampling_params=sampling_params)
        return {"messages": [message]}
    except MaxTokenExceededError:
        # last message is ToolMessage
        return {"messages": []}


def should_continue(state: MessagesState, config: RunnableConfig) -> Literal["tools", END]:
    """Determine whether to continue with tool calls or end the conversation."""
    max_assistant_turns = config["configurable"]["max_assistant_turns"]
    num_assistant_turns = 0
    for message in state["messages"]:
        if message.type == "ai":
            num_assistant_turns += 1

    last_message = state["messages"][-1]

    # LLM call failed, e.g: max response length exceeded
    if last_message.type == "tool":
        return END

    # max assistant turns exceeded
    if max_assistant_turns and num_assistant_turns >= max_assistant_turns:
        return END

    # no tool calls
    if not last_message.tool_calls:
        return END

    return "tools"


@register("letta_react_agent")
class ReactAgentLoop(AgentLoopBase):
    """React Agent Loop implementation using Letta client."""

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        """Initialize the class with shared resources."""
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ReactAgentLoop initialization")

        # Initialize tools from config file
        from verl.tools.utils.tool_registry import initialize_tools_from_config
        
        cls.tokenizer = tokenizer
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        print(f"Initialized tools: {cls.tools}")

        # build graph
        cls.graph = cls.build_graph()

    @classmethod
    def build_graph(cls) -> StateGraph:
        """Build the state graph for the React agent loop."""
        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(cls.tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END,
            },
        )

        workflow.add_edge("tools", "agent")
        graph = workflow.compile()
        return graph

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the React agent loop.

        Args:
            sampling_params: Sampling parameters for the model
            **kwargs: Additional arguments including raw_prompt

        Returns:
            AgentLoopOutput: The result of the agent loop execution
        """
        messages = list(kwargs["raw_prompt"])

        model_path = self.config.actor_rollout_ref.model.path
        model_name = "/".join(model_path.split("/")[-2:])

        rollout = self.config.actor_rollout_ref.rollout
        
        # Create Letta client (this would need to be configured properly in a real implementation)
        from letta_client import AsyncLetta
        letta_client = AsyncLetta()  # This would need proper configuration
        
        model = LettaChatModel(
            model=model_name,
            client=letta_client,
            tokenizer=self.tokenizer,
            max_tokens=rollout.response_length,
            max_parallel_calls=rollout.multi_turn.max_parallel_calls,
            tool_parser=rollout.multi_turn.format,
        )

        # Set agent ID (this would need to be configured properly)
        # For now, we'll use a placeholder - in a real implementation, this would be
        # created or retrieved from the Letta client
        model.agent_id = "placeholder_agent_id"

        model = model.bind_tools(self.tools, tool_choice="any")

        config = {
            "configurable": {
                "model": model,
                "sampling_params": sampling_params,
                "max_user_turns": rollout.multi_turn.max_user_turns,
                "max_assistant_turns": rollout.multi_turn.max_assistant_turns,
            }
        }

        # TODO: how to handle multiple trajectories in a graph invocation?
        # Each graph node may have its own LLM calls and state, e.g:
        # https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart
        state = await self.graph.ainvoke(input={"messages": messages}, config=config)

        output = convert_to_agent_output(state["messages"], rollout.response_length)
        return output
