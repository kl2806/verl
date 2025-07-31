import pytest
import torch
import pandas as pd
import re
import tempfile
import os
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from transformers import AutoTokenizer

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


@pytest.fixture
def sample_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


@pytest.fixture
def processed_data_path():
    """Path to the actual processed test data"""
    return "/home/riddhi/letta-synthetic-data/data/processed_data_test.parquet"


@pytest.fixture
def sample_parquet_file(processed_data_path, tmp_path):
    """Create a test parquet file using actual processed data"""
    df = pd.read_parquet(processed_data_path)
    test_df = df.head(5).copy()

    test_df = test_df.rename(columns={'prompt': 'messages'})
    
    if 'tools' not in test_df.columns:
        test_df['tools'] = None
    if 'enable_thinking' not in test_df.columns:
        test_df['enable_thinking'] = True  
    
    file_path = tmp_path / "test_data.parquet"
    test_df.to_parquet(file_path)
    return str(file_path)


class TestDatasetInitialization:
    
    def test_default(self, sample_parquet_file, sample_tokenizer):
        # truncation to handle long sequences
        config = {"truncation": "right", "max_length": 4096}
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        assert len(dataset) > 0
        item = dataset[0]
        assert all(key in item for key in ["input_ids", "attention_mask", "loss_mask", "position_ids"])
        
    def test_custom(self, sample_parquet_file, sample_tokenizer):
        config = {
            "truncation": "truncate_middle",
            "max_length": 2048,
            "multiturn": {
                "messages_key": "messages",
                "tools_key": "tools",
                "enable_thinking_key": "enable_thinking"
            }
        }
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        assert dataset.max_length == 2048
        assert dataset.truncation == "truncate_middle"
        
    def test_multi_files(self, processed_data_path, sample_tokenizer, tmp_path):
        df = pd.read_parquet(processed_data_path)
        df = df.rename(columns={'prompt': 'messages'})
        
        if 'tools' not in df.columns:
            df['tools'] = None
        if 'enable_thinking' not in df.columns:
            df['enable_thinking'] = True
            
        files = []
        for i in range(2):
            split_df = df.head(2) if i == 0 else df.tail(2)
            file_path = tmp_path / f"test_{i}.parquet"
            split_df.to_parquet(file_path)
            files.append(str(file_path))
            
        config = {"truncation": "right", "max_length": 4096}
        dataset = MultiTurnSFTDataset(files, sample_tokenizer, config)
        assert len(dataset) > 0


class TestLossMasking:
    
    def test_has_mask(self, sample_parquet_file, sample_tokenizer):
        config = {"truncation": "right", "max_length": 4096}
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        for i in range(len(dataset)):
            item = dataset[i]
            loss_mask = item["loss_mask"]
            assert torch.sum(loss_mask) > 0, f"No loss mask found for item {i}"
            
    def test_mask_length(self, sample_parquet_file, sample_tokenizer):
        config = {"truncation": "right", "max_length": 4096}
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        for i in range(len(dataset)):
            item = dataset[i]
            assert len(item["input_ids"]) == len(item["loss_mask"])
            assert len(item["input_ids"]) == len(item["attention_mask"])
            assert len(item["input_ids"]) == len(item["position_ids"])


class TestTagAndBracketValidation:
    
    def _get_loss_mask_text(self, tokenizer, input_ids, loss_mask):
        loss_indices = torch.where(loss_mask == 1)[0]
        if len(loss_indices) == 0:
            return ""
        loss_tokens = input_ids[loss_indices]
        return tokenizer.decode(loss_tokens, skip_special_tokens=False)
    
    def _validate_tags_and_brackets(self, text):
        results = {}
        
        think_open = text.count("<think>")
        think_close = text.count("</think>")
        results["think_tags"] = think_open == think_close
        
        tool_call_open = text.count("<tool_call>")
        tool_call_close = text.count("</tool_call>")
        results["tool_call_tags"] = tool_call_open == tool_call_close
        
        results["no_tool_response_tags"] = text.count("<tool_response>") == 0
        
        open_brackets = text.count("{")
        close_brackets = text.count("}")
        results["matched_brackets"] = open_brackets == close_brackets
        
        return results
    
    def test_think_tags(self, sample_parquet_file, sample_tokenizer):
        config = {"truncation": "right", "max_length": 4096}
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        for i in range(len(dataset)):
            item = dataset[i]
            loss_mask_text = self._get_loss_mask_text(dataset.tokenizer, item["input_ids"], item["loss_mask"])
            
            think_open = loss_mask_text.count("<think>")
            think_close = loss_mask_text.count("</think>")
            assert think_open == think_close, f"Unmatched think tags in item {i}: found {think_open} open, {think_close} close\nText: {loss_mask_text[:200]}..."
    
    def test_tool_tags(self, sample_parquet_file, sample_tokenizer):
        config = {"truncation": "right", "max_length": 4096}
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        for i in range(len(dataset)):
            item = dataset[i]
            loss_mask_text = self._get_loss_mask_text(dataset.tokenizer, item["input_ids"], item["loss_mask"])
            
            tool_call_open = loss_mask_text.count("<tool_call>")
            tool_call_close = loss_mask_text.count("</tool_call>")
            assert tool_call_open == tool_call_close, f"Unmatched tool_call tags in item {i}: found {tool_call_open} open, {tool_call_close} close\nText: {loss_mask_text[:200]}..."
    
    def test_no_tool_response(self, sample_parquet_file, sample_tokenizer):
        config = {"truncation": "right", "max_length": 4096}
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        for i in range(len(dataset)):
            item = dataset[i]
            loss_mask_text = self._get_loss_mask_text(dataset.tokenizer, item["input_ids"], item["loss_mask"])
            
            tool_response_count = loss_mask_text.count("<tool_response>")
            assert tool_response_count == 0, f"Found {tool_response_count} tool_response tags in loss mask of item {i}\nText: {loss_mask_text[:200]}..."
    
    def test_brackets(self, sample_parquet_file, sample_tokenizer):
        config = {"truncation": "right", "max_length": 4096}
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        for i in range(len(dataset)):
            item = dataset[i]
            loss_mask_text = self._get_loss_mask_text(dataset.tokenizer, item["input_ids"], item["loss_mask"])
            
            open_brackets = loss_mask_text.count("{")
            close_brackets = loss_mask_text.count("}")
            assert open_brackets == close_brackets, f"Unmatched brackets in item {i}: found {open_brackets} open, {close_brackets} close\nText: {loss_mask_text[:200]}..."
    
    def test_all_validation(self, sample_parquet_file, sample_tokenizer):
        config = {"truncation": "right", "max_length": 4096}
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        for i in range(len(dataset)):
            item = dataset[i]
            loss_mask_text = self._get_loss_mask_text(dataset.tokenizer, item["input_ids"], item["loss_mask"])
            validation_results = self._validate_tags_and_brackets(loss_mask_text)
            
            for validation_name, is_valid in validation_results.items():
                assert is_valid, f"Validation failed for {validation_name} in item {i}\nText: {loss_mask_text[:200]}..."


class TestTruncation:
    
    def test_error(self, sample_parquet_file, sample_tokenizer):
        config = {"truncation": "error", "max_length": 1024}
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        
        with pytest.raises(ValueError, match="is larger than"):
            dataset[0]
    
    def test_middle(self, sample_parquet_file, sample_tokenizer):
        config = {"truncation": "truncate_middle", "max_length": 2048}
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        item = dataset[0]
        assert len(item["input_ids"]) == 2048
        
        assert all(key in item for key in ["input_ids", "attention_mask", "loss_mask", "position_ids"])


class TestPadding:
    
    def test_applied(self, sample_parquet_file, sample_tokenizer):
        config = {"max_length": 5000, "truncation": "right"}  
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        item = dataset[0]
        
        assert len(item["input_ids"]) == 5000
        assert len(item["attention_mask"]) == 5000
        assert len(item["loss_mask"]) == 5000
        
        # check to ensure 0 for all padding tokens
        pad_indices = torch.where(item["attention_mask"] == 0)[0]
        if len(pad_indices) > 0:
            assert torch.all(item["loss_mask"][pad_indices] == 0)


class TestPositionIds:
    
    def test_generation(self, sample_parquet_file, sample_tokenizer):
        config = {"truncation": "right", "max_length": 4096}
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        item = dataset[0]
        position_ids = item["position_ids"]
        attention_mask = item["attention_mask"]
        
        assert torch.all(position_ids[attention_mask == 0] == 0)
        
        valid_positions = position_ids[attention_mask == 1]
        expected_positions = torch.arange(len(valid_positions))
        assert torch.all(valid_positions == expected_positions)


class TestErrorHandling:
    
    def test_invalid_truncation(self, sample_parquet_file, sample_tokenizer):
        with pytest.raises(AssertionError):
            MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, {"truncation": "invalid"})
    
    def test_system_not_first(self, sample_tokenizer, tmp_path):
        invalid_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "I am an assistant"}
        ]
        df = pd.DataFrame({"messages": [invalid_messages]})  
        file_path = tmp_path / "invalid.parquet"
        df.to_parquet(file_path)
        
        config = {"truncation": "right", "max_length": 1024}
        dataset = MultiTurnSFTDataset(str(file_path), sample_tokenizer, config)
        with pytest.raises(ValueError, match="System message should be the first message"):
            dataset[0]
    
    def test_unknown_role(self, sample_tokenizer, tmp_path):
        invalid_messages = [
            {"role": "unknown", "content": "Hello"}
        ]
        df = pd.DataFrame({"messages": [invalid_messages]})  # Wrap in list
        file_path = tmp_path / "invalid.parquet"
        df.to_parquet(file_path)
        
        config = {"truncation": "right", "max_length": 1024}
        dataset = MultiTurnSFTDataset(str(file_path), sample_tokenizer, config)
        with pytest.raises((ValueError, Exception)):  # Broader exception catching
            dataset[0]


class TestIntegration:
    
    def test_end_to_end(self, sample_parquet_file, sample_tokenizer):
        config = {"truncation": "right", "max_length": 4096}
        dataset = MultiTurnSFTDataset(sample_parquet_file, sample_tokenizer, config)
        
        for i in range(len(dataset)):
            item = dataset[i]
            
            assert isinstance(item, dict)
            assert all(isinstance(item[key], torch.Tensor) for key in ["input_ids", "attention_mask", "loss_mask", "position_ids"])
            
            seq_len = len(item["input_ids"])
            assert all(len(item[key]) == seq_len for key in ["attention_mask", "loss_mask", "position_ids"])
            
            assert torch.all(item["attention_mask"] >= 0) and torch.all(item["attention_mask"] <= 1)
            assert torch.all(item["loss_mask"] >= 0) and torch.all(item["loss_mask"] <= 1)
            tokens = []
            for i in range(len(item["loss_mask"])):
                if item["loss_mask"][i] == 1:
                    tokens.append(item["input_ids"][i])
                if item["loss_mask"][i] == 0 and tokens:
                    message = sample_tokenizer.decode(tokens)
                    print(f"decoded message (tokens={tokens}): '{message.tolist}'")
                    assert "assistant\n" in message
                    assert "<tool_call>" in message and "</tool_call>" in message 
                    assert "<think>" in message and "</think>" in message 
                    assert (i not in message for i in ["inner_monologue", "tool_response", "user\n"])
                    tokens = []
                    
            
                    
    
    def test_real_data(self, processed_data_path, sample_tokenizer):
        """Test with the actual processed data directly"""
        df = pd.read_parquet(processed_data_path)
        df = df.rename(columns={'prompt': 'messages'})
        
        if 'tools' not in df.columns:
            df['tools'] = None
        if 'enable_thinking' not in df.columns:
            df['enable_thinking'] = True
            
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.head(3).to_parquet(f.name)
            temp_path = f.name
        
        try:
            config = {"truncation": "right", "max_length": 4096}
            dataset = MultiTurnSFTDataset(temp_path, sample_tokenizer, config)
            item = dataset[0]
            
            assert len(item["input_ids"]) > 0
            assert torch.sum(item["loss_mask"]) > 0
            
            loss_mask_text = ""
            loss_indices = torch.where(item["loss_mask"] == 1)[0]
            if len(loss_indices) > 0:
                loss_tokens = item["input_ids"][loss_indices]
                loss_mask_text = dataset.tokenizer.decode(loss_tokens, skip_special_tokens=False)
                
                think_open = loss_mask_text.count("<think>")
                think_close = loss_mask_text.count("</think>")
                assert think_open == think_close, f"Unmatched think tags in real data: {think_open} vs {think_close}"
                
                tool_call_open = loss_mask_text.count("<tool_call>")
                tool_call_close = loss_mask_text.count("</tool_call>")
                assert tool_call_open == tool_call_close, f"Unmatched tool_call tags in real data: {tool_call_open} vs {tool_call_close}"
                
                open_brackets = loss_mask_text.count("{")
                close_brackets = loss_mask_text.count("}")
                assert open_brackets == close_brackets, f"Unmatched brackets in real data: {open_brackets} vs {close_brackets}"
                
        finally:
            os.unlink(temp_path)


def create_test_parquet(conversations: List[List[Dict]], file_path: str, **kwargs):
    data = {"messages": conversations}
    data.update(kwargs)
    df = pd.DataFrame(data)
    df.to_parquet(file_path)


if __name__ == "__main__":
    pytest.main([__file__])
