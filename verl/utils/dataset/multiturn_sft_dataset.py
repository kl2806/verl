# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.loss_mask_key = multiturn_config.get("loss_mask_key", "loss_mask")

        assert self.truncation in ["error", "left", "right", "truncate_middle"]

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()
        self.loss_mask = self.dataframe[self.loss_mask_key].apply(series_to_item).tolist()

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]

        # # First, get the full conversation tokens
        full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=False, skip_special_tokens=False)
        input_ids = full_tokens[0]  # The output is already a tensor
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
        current_pos = 0
        def subseq_position_helper(full_sequence, subsequence):
            full_len = len(full_sequence)
            sub_len = len(subsequence)
            
            for i in range(full_len - sub_len + 1):
                if torch.equal(full_sequence[i:i + sub_len], subsequence):
                    return i
            return -1
        
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant" and self.loss_mask[item][i]:
                single_msg_formatted = tokenizer.apply_chat_template([msg], tokenize=True, return_tensors="pt", add_generation_prompt=False, skip_special_tokens=False)[0]
                
                found_pos = subseq_position_helper(input_ids[current_pos:], single_msg_formatted)
                
                if found_pos != -1:
                    actual_start = current_pos + found_pos
                    actual_end = actual_start + len(single_msg_formatted)
                    loss_mask[actual_start:actual_end] = 1
                    current_pos = actual_end
                else:
                    if i + 1 < len(messages):
                        next_msg = [messages[i + 1]]
                        next_tokens = tokenizer.apply_chat_template(next_msg, tokenize=True, return_tensors="pt", add_generation_prompt=False, skip_special_tokens=False)[0]
                        next_pos = subseq_position_helper(input_ids[current_pos:], next_tokens)
                        if next_pos != -1:
                            current_pos = current_pos + next_pos
        

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length <= self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]

            elif self.truncation == "truncate_middle":
                first_system_idx = None
                first_user_idx = None
                
                for i, msg in enumerate(messages):
                    if msg["role"] == "system" and first_system_idx is None:
                        first_system_idx = i
                    elif msg["role"] == "user" and first_user_idx is None:
                        first_user_idx = i
                        break 
                
                keep_end = 0
                
                if first_system_idx is not None and first_user_idx is not None:
                    keep_messages = messages[:first_user_idx]
                    keep_tokens = tokenizer.apply_chat_template(keep_messages, tokenize=True, return_tensors="pt", add_generation_prompt=False, skip_special_tokens=False)
                    keep_end = keep_tokens[0].shape[0]
                elif first_user_idx is not None:
                    keep_messages = messages[:first_user_idx + 1]
                    keep_tokens = tokenizer.apply_chat_template(keep_messages, tokenize=True, return_tensors="pt", add_generation_prompt=False, skip_special_tokens=False)
                    keep_end = keep_tokens[0].shape[0]
                else:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    loss_mask = loss_mask[:self.max_length]
                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": torch.arange(len(input_ids), dtype=torch.long) * attention_mask,
                        "loss_mask": loss_mask,
                    }
                
                if keep_end > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    loss_mask = loss_mask[:self.max_length]
                else:
                    remaining_length = self.max_length - keep_end
                    if remaining_length > 0:
                        end_tokens = input_ids[-remaining_length:]
                        end_attention = attention_mask[-remaining_length:]
                        end_loss_mask = loss_mask[-remaining_length:]

                        input_ids = torch.cat([input_ids[:keep_end], end_tokens])
                        attention_mask = torch.cat([attention_mask[:keep_end], end_attention])
                        loss_mask = torch.cat([loss_mask[:keep_end], end_loss_mask])
                    else:
                        # keep_end equals max_length, just truncate
                        input_ids = input_ids[:keep_end]
                        attention_mask = attention_mask[:keep_end]
                        loss_mask = loss_mask[:keep_end]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        import json

        # to_save = {
        #     "input_ids": input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids,
        #     "attention_mask": attention_mask.tolist() if hasattr(attention_mask, "tolist") else attention_mask,
        #     "position_ids": position_ids.tolist() if hasattr(position_ids, "tolist") else position_ids,
        #     "loss_mask": loss_mask.tolist() if hasattr(loss_mask, "tolist") else loss_mask,
        #     "FULL_decoded_text": self.tokenizer.decode(input_ids)
        # }
        # try:
        #     with open("losses_and_inputs.txt", "w") as f:
        #         json.dump(to_save, f)
        #         f.write("\n")
        #         # Add a line with the decoded input_ids for sections where loss_mask == 1
        #         if hasattr(loss_mask, "tolist"):
        #             loss_mask_list = loss_mask.tolist()
        #         else:
        #             loss_mask_list = loss_mask
        #         if hasattr(input_ids, "tolist"):
        #             input_ids_list = input_ids.tolist()
        #         else:
        #             input_ids_list = input_ids
        #         indices = [i for i, v in enumerate(loss_mask_list) if v == 1]
        #         # Get the corresponding input_ids
        #         input_ids_with_loss = [input_ids_list[i] for i in indices]
        #         # Decode using the tokenizer if available
        #         decoded_text = ""
        #         if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "decode"):
        #             decoded_text = self.tokenizer.decode(input_ids_with_loss)
        #         else:
        #             decoded_text = str(input_ids_with_loss)
        #         f.write("TEXT: " + decoded_text + "\n")
        # except Exception as e:
        #     print(f"Failed to save losses_and_inputs.txt: {e}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
