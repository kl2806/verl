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
Test file for Letta React Agent Loop.

This file contains basic tests to verify the Letta React Agent Loop implementation.
Note: These are placeholder tests that would need to be adapted for actual Letta integration.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from recipe.letta_agent.react_agent_loop import ReactAgentLoop
from recipe.letta_agent.letta_chat_model import LettaChatModel, MaxTokenExceededError
from recipe.letta_agent.example.math_expression import LETTA_TOOLS


class TestReactAgentLoop:
    """Test cases for ReactAgentLoop."""

    def test_init_class(self):
        """Test class initialization."""
        # Mock the required dependencies
        config = Mock()
        tokenizer = Mock()
        
        # Reset class state
        ReactAgentLoop._class_initialized = False
        
        # Test initialization
        ReactAgentLoop.init_class(config, tokenizer)
        
        assert ReactAgentLoop._class_initialized
        assert hasattr(ReactAgentLoop, 'graph')

    def test_build_graph(self):
        """Test graph building."""
        # Reset class state
        ReactAgentLoop._class_initialized = False
        
        # Mock tools
        ReactAgentLoop.tools = LETTA_TOOLS
        
        # Build graph
        graph = ReactAgentLoop.build_graph()
        
        assert graph is not None
        # Verify graph has the expected nodes
        assert "agent" in graph.nodes
        assert "tools" in graph.nodes

    @pytest.mark.asyncio
    async def test_run_method_structure(self):
        """Test the structure of the run method."""
        # Mock dependencies
        config = Mock()
        config.actor_rollout_ref.model.path = "test/model/path"
        config.actor_rollout_ref.rollout.response_length = 1000
        config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 1
        config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
        config.actor_rollout_ref.rollout.multi_turn.max_user_turns = 10
        config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 5
        
        server_manager = Mock()
        tokenizer = Mock()
        processor = Mock()
        
        # Mock tools
        ReactAgentLoop.tools = LETTA_TOOLS
        
        # Create agent loop instance
        agent_loop = ReactAgentLoop(
            trainer_config=Mock(config=config),
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor
        )
        
        # Mock the graph invoke method
        mock_messages = [{"role": "user", "content": "Test message"}]
        agent_loop.graph = AsyncMock()
        agent_loop.graph.ainvoke.return_value = {"messages": mock_messages}
        
        # Test run method
        sampling_params = {"temperature": 0.7}
        kwargs = {"raw_prompt": [{"role": "user", "content": "Test message"}]}
        
        # This would need proper Letta client mocking in a real test
        with patch('recipe.letta_agent.react_agent_loop.AsyncLetta'):
            try:
                result = await agent_loop.run(sampling_params, **kwargs)
                # The result should be an AgentLoopOutput
                assert hasattr(result, 'prompt_ids')
                assert hasattr(result, 'response_ids')
                assert hasattr(result, 'response_mask')
            except Exception as e:
                # Expected to fail due to missing Letta client setup
                assert "Agent ID must be set" in str(e) or "AsyncLetta" in str(e)


class TestLettaChatModel:
    """Test cases for LettaChatModel."""

    def test_letta_chat_model_initialization(self):
        """Test LettaChatModel initialization."""
        client = Mock()
        tokenizer = Mock()
        
        model = LettaChatModel(
            model="test-model",
            client=client,
            tokenizer=tokenizer,
            max_tokens=1000
        )
        
        assert model.model_name == "test-model"
        assert model.client == client
        assert model.tokenizer == tokenizer
        assert model.max_tokens == 1000

    def test_max_token_exceeded_error(self):
        """Test MaxTokenExceededError exception."""
        error = MaxTokenExceededError("Test error")
        assert str(error) == "Test error"


if __name__ == "__main__":
    pytest.main([__file__])
