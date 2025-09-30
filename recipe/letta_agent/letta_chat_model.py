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
Letta Chat Model implementation for VERL Agent Loop.

This implementation provides a Letta-based chat model that integrates with VERL's
agent loop framework, similar to the LangGraph implementation but using Letta client.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.messages.tool import InvalidToolCall, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from letta_client import AsyncLetta, LlmConfig, MessageCreate
from letta_client.types import LettaResponse

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, AsyncLLMServerManager
from verl.experimental.agent_loop.tool_parser import ToolParser

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MaxTokenExceededError(Exception):
    """Indicate that history chat messages + tool message exceeds LLM max_tokens."""

    pass


class LettaChatModel(BaseChatModel):
    """Letta-based chat model for VERL agent loops."""

    model_name: str = Field(alias="model")
    """The name of the model"""

    client: AsyncLetta
    """Letta client for communication"""

    tokenizer: Any
    """Tokenizer for the model"""

    max_tokens: int
    """Max tokens to generate"""

    tool_parser: str = "hermes"
    """Tool parser for the model"""

    max_parallel_calls: int = 1
    """Max parallel tool calls"""

    temperature: float = 1.0
    """Temperature for sampling"""

    top_p: float = 1.0
    """Top p for sampling"""

    repetition_penalty: float = 1.0
    """Repetition penalty for sampling"""

    agent_id: Optional[str] = None
    """Letta agent ID for conversation"""

    def bind_tools(self, tools, **kwargs) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the model.

        Args:
            tools: Sequence of tools to bind to the model.

        Returns:
            A Runnable that returns a message.
        """
        formatted_tools: list = [convert_to_openai_tool(tool) for tool in tools]

        # used to remove system prompt prefix when encoding tool response
        system_prompt = self.tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
        kwargs["system_prompt"] = system_prompt

        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseChatModel]:
        """Ref: https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/"""
        raise NotImplementedError

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate chat completion message using Letta client.

        Args:
            messages (list[BaseMessage]): List of messages.
            stop (Optional[list[str]], optional): Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings. Defaults to None.

        Returns:
            ChatResult: Chat result.
        """
        request_id, prompt_ids, response_mask = await self._preprocess(messages, **kwargs)

        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }
        if "sampling_params" in kwargs:
            sampling_params.update(kwargs["sampling_params"])

        # Use Letta client to generate response
        response = await self._generate_with_letta(messages, sampling_params, **kwargs)

        message = await self._postprocess(request_id, prompt_ids, response_mask, response, **kwargs)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _generate_with_letta(
        self, 
        messages: list[BaseMessage], 
        sampling_params: dict[str, Any], 
        **kwargs: Any
    ) -> LettaResponse:
        """Generate response using Letta client."""
        if not self.agent_id:
            raise ValueError("Agent ID must be set before generating responses")

        # Convert messages to Letta format
        letta_messages = []
        for msg in messages:
            if msg.type == "human":
                letta_messages.append(MessageCreate(role="user", content=msg.content))
            elif msg.type == "ai":
                letta_messages.append(MessageCreate(role="assistant", content=msg.content))
            elif msg.type == "tool":
                # Tool messages are handled differently in Letta
                letta_messages.append(MessageCreate(role="user", content=f"Tool response: {msg.content}"))

        # Create Letta response
        response = await self.client.agents.messages.create(
            agent_id=self.agent_id,
            messages=letta_messages,
        )

        return response

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name

    async def _preprocess(self, messages: list[BaseMessage], **kwargs: Any) -> tuple[str, list[int], list[int]]:
        """Preprocess messages for chat completion.

        To ensure strong consistency with policy model, we store trajectory
        (prompt_ids, response_mask) in latest AIMessage.response_metadata.

        1. Encode ToolMessage to token ids.
        2. Retrieve trajectory (prompt_ids, response_mask) from latest AIMessage.response_metadata.
        3. Append ToolMessage token ids to prompt_ids, and append 0 to response_mask.

        Args:
            messages (list[BaseMessage]): List of messages.

        Returns:
            tuple[str, list[int], list[int]]: Request id, prompt ids, response mask.
        """
        # messages: [system], human, ai, human|tool, ai, human|tool, ...
        assert messages[-1].type in ["human", "tool"], (
            f"Last message must be human or tool, but got {messages[-1].type}"
        )
        loop = asyncio.get_running_loop()

        # Case 1: initial chat completion: [system], human
        if messages[-1].type == "human" and (len(messages) == 1 or messages[-2].type != "ai"):
            prompt_ids = await loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    convert_to_openai_messages(messages),
                    tools=kwargs.get("tools"),
                    add_generation_prompt=True,
                    tokenize=True,
                ),
            )
            return str(uuid.uuid4()), prompt_ids, []

        # Case 2: follow up chat completion with tool/human response: [system], human, ai, human|tool, ...
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].type == "ai":
                break
        assert "prompt_ids" in messages[i].response_metadata, "Last message must have prompt_ids in response_metadata"
        assert "response_mask" in messages[i].response_metadata, (
            "Last message must have response_mask in response_metadata"
        )

        # encode tool response
        tool_responses = convert_to_openai_messages(messages[i + 1 :])
        tool_response_ids = await loop.run_in_executor(
            None,
            lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        tool_response_ids = tool_response_ids[len(kwargs["system_prompt"]) :]

        # stop generation if response length exceeds max response length
        if len(messages[i].response_metadata["response_mask"]) + len(tool_response_ids) >= self.max_tokens:
            raise MaxTokenExceededError(f"Max response length {self.max_tokens} exceeded")

        # append tool response to prompt
        request_id = messages[i].response_metadata.pop("request_id")
        prompt_ids = messages[i].response_metadata.pop("prompt_ids")
        response_mask = messages[i].response_metadata.pop("response_mask")
        prompt_ids += tool_response_ids
        response_mask += [0] * len(tool_response_ids)

        return request_id, prompt_ids, response_mask

    async def _postprocess(
        self, request_id: str, prompt_ids: list[int], response_mask: list[int], response: LettaResponse, **kwargs: Any
    ) -> AIMessage:
        """Postprocess Letta response when chat completion is done.

        1. Decode response, parse tool calls to AIMessage.
        2. Append response_ids to prompt_ids, and append 1 to response_mask.
        3. Store trajectory (prompt_ids, response_mask) in AIMessage.response_metadata.

        Args:
            request_id (str): Unique request id.
            prompt_ids (list[int]): Input prompt token ids in this chat completion.
            response_mask (list[int]): Response mask before this chat completion.
            response (LettaResponse): Letta response object.

        Returns:
            AIMessage: Postprocessed message.
        """
        # Extract content and tool calls from Letta response
        content = ""
        tool_calls = []
        invalid_tool_calls = []

        if response.messages:
            last_message = response.messages[-1]
            if hasattr(last_message, 'content'):
                content = last_message.content or ""
            
            # Extract tool calls if present
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    try:
                        args = json.loads(tool_call.arguments) if isinstance(tool_call.arguments, str) else tool_call.arguments
                        if not isinstance(args, dict):
                            error = f"Tool arguments must be a JSON object, got {type(args).__name__}"
                            invalid_tool_calls.append(
                                InvalidToolCall(
                                    name=tool_call.name,
                                    args=tool_call.arguments,
                                    id=str(uuid.uuid4()),
                                    error=error,
                                )
                            )
                        else:
                            tool_calls.append(
                                ToolCall(
                                    name=tool_call.name,
                                    args=args,
                                    id=str(uuid.uuid4()),
                                )
                            )
                    except json.JSONDecodeError as e:
                        error = f"Invalid JSON tool arguments: {e}"
                        invalid_tool_calls.append(
                            InvalidToolCall(
                                name=tool_call.name,
                                args=tool_call.arguments,
                                id=str(uuid.uuid4()),
                                error=error,
                            )
                        )

        # For now, we'll need to simulate response_ids since Letta doesn't provide them directly
        # This is a limitation that would need to be addressed in a real implementation
        response_ids = self.tokenizer.encode(content, add_special_tokens=False)
        
        prompt_ids += response_ids
        response_mask += [1] * len(response_ids)

        message = AIMessage(
            content=content,
            tool_calls=tool_calls[: self.max_parallel_calls],
            invalid_tool_calls=invalid_tool_calls[: self.max_parallel_calls],
            response_metadata={
                "request_id": request_id,
                "prompt_ids": prompt_ids,
                "response_mask": response_mask,
            },
        )
        return message


class TruncateStructuredTool(StructuredTool):
    """Structured tool with response truncation."""

    tool_response_truncate_side: str
    """truncate side of tool response: left, middle, right"""

    max_tool_response_length: int
    """max length of tool response"""

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Any:
        tool_response = await super()._arun(*args, config=config, **kwargs)
        tool_response = str(tool_response)

        if len(tool_response) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response = tool_response[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response = "(truncated)..." + tool_response[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response = tool_response[:length] + "...(truncated)..." + tool_response[-length:]

        return tool_response


def convert_to_agent_output(messages: list[BaseMessage], response_length: int) -> AgentLoopOutput:
    """Convert messages to AgentLoopOutput.

    Args:
        messages (List[BaseMessage]): List of messages, last message must be assistant
            with response_metadata containing `prompt_ids` and `response_mask`.
        response_length (int): Max length of response.

    Returns:
        AgentLoopOutput: agent loop output trajectory used for training.
    """
    # skip last tool calls
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type != "tool":
            break
    last_message = messages[i]
    assert last_message.type == "ai", f"Last message must be assistant, but got {last_message.type}"
    assert "prompt_ids" in last_message.response_metadata, "Last message must have prompt_ids in response_metadata"
    assert "response_mask" in last_message.response_metadata, (
        "Last message must have response_mask in response_metadata"
    )

    num_turns = 0
    for i in range(len(messages)):
        if messages[i].type == "system":
            continue
        # parallel tool calls are in single turn
        if i == 0 or messages[i].type != messages[i - 1].type:
            num_turns += 1

    prompt_ids = last_message.response_metadata["prompt_ids"]
    response_mask = last_message.response_metadata["response_mask"]

    response_ids = prompt_ids[-len(response_mask) :]
    prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

    output = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids[:response_length],
        response_mask=response_mask[:response_length],
        num_turns=num_turns,
        metrics={},
    )
    return output
