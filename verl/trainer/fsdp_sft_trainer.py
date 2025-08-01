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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation

Features:
- FSDP training with sequence parallelism support
- Validation with sample generation
- Configurable generation parameters for validation
- Logging of generated samples to tracking systems
"""

from hmac import new
import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
import time
from contextlib import nullcontext
import tempfile
import subprocess
import pandas as pd
from pathlib import Path
from json_parser import OptimisticJSONParser
import hydra
import torch
import torch.distributed
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, GenerationConfig

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def validate_json_generation(generation_text):
    import json
    try:
        start_idx = generation_text.find("{\"name\"")
        if start_idx == -1:
            returnable = False, "No JSON object starting with {\"name\" found", None

        end_idx = generation_text.find("}}", start_idx)
        if end_idx == -1:
            returnable = False, "No closing '}}' found for JSON object", None

        to_parse = generation_text[start_idx:end_idx+2]
        
        parsed_json = OptimisticJSONParser().parse(to_parse)
        returnable = True, None, parsed_json
    except json.JSONDecodeError as json_err:
        returnable = False, f"JSON decode error: {json_err}", None
    except Exception as parse_err:
        returnable = False, f"Parsing error: {parse_err}", None
    
    if not returnable[0]:
        with open("not_parsed.txt", "a") as f:
            f.write(generation_text + "\n")
    return returnable

def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


def evaluate_generations(generations):
    """Evaluate a list of generation strings for JSON validity and tag consistency.

    The behaviour intentionally mirrors the inline logic previously
    implemented in the validation loop so that downstream code expecting
    identical prints / side-effects continues to function unchanged.

    Parameters
    ----------
    generations : list[str]
        The generated strings to validate.

    Returns
    -------
    tuple[list[int], list[int]]
        json_rates – 100 or 0 per sample depending on JSON validity
        tag_errors – cumulative count of tag related errors per sample
    """
    json_rates: list[int] = []
    tag_errors: list[int] = []
 
    total_generations = len(generations)
    

    for i, generation in enumerate(generations):
        tag_errors_count = 0
        json_valid, error_msg, _ = validate_json_generation(generation)

        if generation.count("<tool_call>") != generation.count("</tool_call>"):
            tag_errors_count += 1
            print(f"✗ Generation {i+1} has a mismatched number of <tool_call> tags")
        else:
            print(f"✓ Generation {i+1} has a correct number of <tool_call> tags")

        if generation.count("<think>") != generation.count("</think>"):
            tag_errors_count += 1
            print(f"✗ Generation {i+1} has a mismatched number of think tags")
        else:
            print(f"✓ Generation {i+1} has a correct number of think tags")

        if generation.count("{") != generation.count("}"):
            tag_errors_count += 1
            print("✗ Generation {i+1} has a mismatched number of '{' and '}' tags")
        else:
            print("✓ Generation {i+1} has a correct number of '{' and '}' tags")

        if "<inner_monologue>" in generation or "</inner_monologue>" in generation:
            tag_errors_count += 1
            print(f"✗ Generation {i+1} has an inner monologue")

        tag_errors.append(tag_errors_count)

        if json_valid:
            json_valid_count = 1
            print(f"✓ Generation {i+1} parses as valid JSON")
        else:
            json_valid_count = 0
            print(f"✗ Generation {i+1} failed JSON parsing: {error_msg}")
            logger.warning(f"FAILED PARSING: {generation}")
        json_rates.append(json_valid_count * 100)

    json_valid_rate = (sum(json_rates) / total_generations * 100) if total_generations > 0 else 0
    print(
        f"JSON Validation Summary: {sum(json_rates)}/{total_generations} --> ({json_valid_rate:.1f}%) valid JSON generations"
    )

    return json_rates, tag_errors


class FSDPSFTTrainer:
    def __init__(self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)
        # build modelx
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size
        
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        
        # Set dropout rates if specified in config
        if hasattr(self.config.model, 'attention_dropout'):
            config.attention_dropout = self.config.model.attention_dropout
        if hasattr(self.config.model, 'hidden_dropout'):
            config.hidden_dropout = self.config.model.hidden_dropout
            
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        if self.device_mesh.size() == 1:
            if self.device_mesh.get_rank() == 0:
                print("Single GPU detected - using model directly without FSDP")
            self.fsdp_model = self.model.to('cuda')
        else:
            if self.device_mesh.get_rank() == 0:
                print("Multi-GPU detected - using FSDP with FULL_SHARD strategy")
            self.fsdp_model = FSDP(
                module=self.model,
                auto_wrap_policy=auto_wrap_policy,
                param_init_fn=init_fn,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                device_mesh=self.device_mesh,
                sync_module_states=True,
                device_id=torch.cuda.current_device(),
                cpu_offload=cpu_offload,
                use_orig_params=False,
            )

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}")

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        position_ids = batch["position_ids"].cuda()
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).cuda()
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            if do_backward:
                loss.backward()
            return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            step_loss += loss.item()

        if self.device_mesh.size() == 1:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        return {"train/loss": step_loss.detach().item(), "train/lr(1e-5)": lr * 1e5}

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def load_validation_prompts(self, max_prompts=None):
        """Load prompts from validation_generation_prompts.parquet file"""
        parquet_path = "/home/riddhi/letta-synthetic-data/data/validation_generation_prompts.parquet"
        
        if not os.path.exists(parquet_path):
            print(f"Warning: Validation prompts file not found at {parquet_path}")
            return []
        
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            
            prompt_data = []
            count = 0
            
            for _, row in df.iterrows():
                if max_prompts and count >= max_prompts:
                    break
                
                messages = row['messages']
                if messages is not None and len(messages) > 0:
                    messages_list = messages.tolist() if hasattr(messages, 'tolist') else messages
                    
                    processed_messages = []
                    for msg in messages_list:
                        if isinstance(msg, dict) and "content" in msg:
                            processed_msg = msg.copy()
                            content = msg["content"]
                            if isinstance(content, str) and content.strip().startswith("<base_instructions>"):
                                processed_msg["content"] = content + "\n/no_think"
                            processed_messages.append(processed_msg)
                        else:
                            processed_messages.append(msg)
                    
                    prompt_data.append(processed_messages)
                    count += 1
            
            print(f"Loaded {len(prompt_data)} prompts from validation_generation_prompts.parquet")
            return prompt_data
            
        except Exception as e:
            print(f"Error loading validation prompts: {e}")
            return []

    def generate_samples(self, batch: dict = None, max_new_tokens=None, use_validation_prompts=False):
        if self.device_mesh.get_rank() != 0:
            return [], []
        
        max_length = self.model.config.max_position_embeddings
        print(f"Starting inline generation with max_length={max_length}")
        start_time = time.time()
        
        self.fsdp_model.eval()
        
        temperature = float(os.getenv("VERL_GENERATION_TEMPERATURE", 0.6))
        top_k = int(os.getenv("VERL_GENERATION_TOP_K", 20))
        top_p = float(os.getenv("VERL_GENERATION_TOP_P", 0.95))
        min_p = 0.0
        do_sample = temperature > 0.0
        
        if use_validation_prompts:
            prompt_data = self.load_validation_prompts(max_prompts=15)
        else:
            prompt_data = []
            if batch is not None:
                input_ids = batch["input_ids"]
                for seq in input_ids:
                    seq = seq[seq != self.tokenizer.pad_token_id]
                    text = self.tokenizer.decode(seq, skip_special_tokens=True)
                    prompt_data.append([{"role": "user", "content": text}])
        
        if not prompt_data:
            return [], []
        
        print(f"Processing {len(prompt_data)} prompts for generation")
        
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        generated_texts = []
        prompt_texts = []
        
        with torch.no_grad():
            batch_size = 1
            for i in range(0, len(prompt_data), batch_size):
                batch_prompts = prompt_data[i:i+batch_size]
                
                try:
                    inputs = self.tokenizer.apply_chat_template(
                        batch_prompts,
                        add_generation_prompt=True,
                        padding=True,
                        truncation=True,
                        max_length=self.model.config.max_position_embeddings,
                        return_tensors="pt",
                        return_dict=True,
                        tokenize=True,
                    )
                except Exception as e:
                    print(f"Warning: Chat template failed for batch {i}: {e}")
                    batch_texts = []
                    for prompt_list in batch_prompts:
                        if isinstance(prompt_list, list) and prompt_list:
                            user_messages = [msg for msg in prompt_list if msg.get('role') == 'user']
                            if user_messages:
                                text = user_messages[-1].get('content', str(prompt_list))
                            else:
                                text = str(prompt_list)
                        else:
                            text = str(prompt_list)
                        batch_texts.append(text)
                    
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.model.config.max_position_embeddings,
                        return_tensors="pt"
                    )
                
                input_ids = inputs["input_ids"].cuda()
                attention_mask = inputs["attention_mask"].cuda()
                
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        generated_outputs = self.fsdp_model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=max_length,
                            temperature=temperature if do_sample else None,
                            top_k=top_k if do_sample else None,
                            top_p=top_p if do_sample else None,
                            min_p=min_p if do_sample else None,
                            do_sample=do_sample,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
                        )
                    
                    for j, generated_seq in enumerate(generated_outputs):
                        input_seq = input_ids[j]
                        
                        prompt_len = len(input_seq)
                        if not torch.equal(generated_seq[:prompt_len], input_seq):
                            if prompt_len > 0 and torch.equal(generated_seq[1:prompt_len+1], input_seq):
                                prompt_len += 1

                        new_tokens = generated_seq[prompt_len:]

                        # Turn OFF special-token stripping just for debugging
                        # dbg_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                        # print(dbg_text)        # → "im_end" token is present if special_token skipping is OFF
                        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                        generated_texts.append(generated_text)
                        prompt_idx = i + j
                        if prompt_idx < len(prompt_data):
                            prompt_list = prompt_data[prompt_idx]
                            if isinstance(prompt_list, list) and prompt_list:
                                if len(prompt_list) > 1:
                                    conversation_parts = []
                                    for msg in prompt_list:
                                        role = msg.get('role', '')
                                        content = msg.get('content', '')
                                        if role and content:
                                            conversation_parts.append(f"{role}\n {content}")
                                    
                                    if conversation_parts:
                                        prompt_text = "\n".join(conversation_parts)
                                    else:
                                        prompt_text = str(prompt_list[-1].get('content', ''))
                                else:
                                    prompt_text = str(prompt_list[-1].get('content', ''))
                            else:
                                prompt_text = str(prompt_list)
                            prompt_texts.append(prompt_text)
                
                        # print(f"Prompt text: {prompt_texts[-1]}")
                        # print(f"Generated tokens: {new_tokens}")
                        # print(f"Decoded generation: {generated_text}")
                
                except Exception as e:
                    print(f"Warning: Generation failed for batch {i}: {e}")
                    for j in range(len(batch_prompts)):
                        generated_texts.append(f"[Generation Error: {str(e)}]")
                        prompt_idx = i + j
                        if prompt_idx < len(prompt_data):
                            prompt_list = prompt_data[prompt_idx]
                            if isinstance(prompt_list, list) and prompt_list:
                                if len(prompt_list) > 1:
                                    # Format the entire conversation for validation generation
                                    conversation_parts = []
                                    for msg in prompt_list:
                                        role = msg.get('role', '').upper()
                                        content = msg.get('content', '')
                                        if role and content:
                                            conversation_parts.append(f"{role}: {content}")
                                    
                                    if conversation_parts:
                                        prompt_text = "\n".join(conversation_parts)
                                    else:
                                        prompt_text = str(prompt_list[-1].get('content', ''))
                                else:
                                    prompt_text = str(prompt_list[-1].get('content', ''))
                            else:
                                prompt_text = str(prompt_list)
                            prompt_texts.append(prompt_text)
        
        self.tokenizer.padding_side = original_padding_side
        
        total_time = time.time() - start_time
        print(f"Inline generation completed in {total_time:.2f}s for {len(generated_texts)} samples")
        
        return prompt_texts, generated_texts

    def save_checkpoint(self, step):
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        
        # Handle both FSDP and non-FSDP cases
        if self.device_mesh.size() == 1:
            # Single GPU case - model is not FSDP wrapped
            state_dict = self.fsdp_model.state_dict()
        else:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.fsdp_model.state_dict()
        
        # save huggingface model
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()

    def fit(self):
        print("=== FIT METHOD STARTED ===")
        rank = self.device_mesh.get_rank()
        print(f"Current rank: {rank}")
        
        # TODO: add a unified tracking
        tracking = None
        val_generations_logger = None
        if rank == 0:
            from omegaconf import OmegaConf
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )
            # Initialize validation generations logger for proper media logging
            from verl.utils.tracking import ValidationGenerationsLogger
            val_generations_logger = ValidationGenerationsLogger()
        
        print("Generating initial samples before training starts...")
        initial_prompts = []
        initial_generations = []
        
        if rank == 0:
            prompts, generations = self.generate_samples(use_validation_prompts=True)
            initial_prompts.extend(prompts)
            initial_generations.extend(generations)
        
        if rank == 0 and initial_prompts and initial_generations and val_generations_logger is not None and tracking is not None:
            json_rates, tag_errors = evaluate_generations(initial_generations)

            initial_samples = []
            num_initial_samples_to_log = min(15, len(initial_prompts))
            for i in range(num_initial_samples_to_log):
                prompt_text = initial_prompts[i]
                generation_text = initial_generations[i]
                json_rate = json_rates[i] if i < len(json_rates) else "N/A"
                tag_error = tag_errors[i] if i < len(tag_errors) else "N/A"
                initial_samples.append([prompt_text, generation_text, f"{json_rate}%", f"{tag_error}"])
            val_generations_logger.log(tracking.logger.keys(), initial_samples, 0)  # Step 0 for initial samples
            
            print(f"\n=== Initial Samples (Before Training) ===")
            for i in range(min(2, len(initial_prompts))):
                print(f"Initial Prompt {i+1}: {initial_prompts[i]}")
                print(f"Initial Generation {i+1}: {initial_generations[i]}")
                print("-" * 50)
            
            if tracking is not None:
                tracking.log(data={"initial/samples_generated": len(initial_generations)}, step=0)
            
            # Log sample examples as text to wandb
            for i, (prompt, generation) in enumerate(zip(initial_prompts[:3], initial_generations[:3])):
                if tracking is not None:
                    tracking.log(data={
                        f"initial/prompt_{i+1}": prompt,
                        f"initial/generation_{i+1}": generation
                    }, step=0)
        
        torch.distributed.barrier()

        global_step = 0
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        # TODO (zhangchi.usc1992) add back checkpoint manager.
        # Currently, it blocks when uploading to hdfs. So very slow.

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
            ):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                metric = self.training_step(data)
                if rank == 0 and tracking is not None:
                    tracking.log(data=metric, step=global_step)

                # for early exit validation
                if global_step >= self.total_training_steps:
                    self.save_checkpoint(step=global_step)
                    break

            # validation
            val_losses = []
            all_prompts = []
            all_generations = []
            
            # Collect a few samples for generation (limit to avoid too much output)
            max_samples_to_generate = int(os.getenv("VERL_MAX_VAL_SAMPLES", 
                getattr(self.config.trainer, "max_val_samples", 15) if hasattr(self.config.trainer, "max_val_samples") else 15))
            samples_generated = 0
            
            for data in self.val_dataloader:
                # data = TensorDict(data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda()
                val_loss = self.validation_step(data)
                val_losses.append(val_loss)
            
            if rank == 0:
                try:
                    prompts, generations = self.generate_samples(use_validation_prompts=True)
                    all_prompts.extend(prompts)
                    all_generations.extend(generations)

                    json_rates, tag_errors = evaluate_generations(generations)
                        
                except Exception as e:
                    print(f"Warning: Failed to generate samples during validation: {e}")
            
            if rank == 0:
                val_loss = torch.mean(torch.stack(val_losses))
                metric = {"val/loss": val_loss.detach().item()}
                
                if all_generations:
                    metric.update({
                        "val/samples_generated": len(all_generations)
                    })
                
                if all_prompts and all_generations:
                    print(f"\n=== Validation Samples (Step {global_step}) ===")
                    for i in range(min(2, len(all_prompts))):
                        print(f"Prompt {i+1}: {all_prompts[i]}")
                        print(f"Generation {i+1}: {all_generations[i]}")
                        print("-" * 50)
                    
                    # Log sample examples as text to wandb
                    for i, (prompt, generation) in enumerate(zip(all_prompts[:3], all_generations[:3])):
                        if tracking is not None:
                            tracking.log(data={
                                f"val/prompt_{i+1}": prompt,
                                f"val/generation_{i+1}": generation
                            }, step=global_step)
                
                
                    val_samples = []
                    for prompt, generation, json_rate, tag_error in zip(all_prompts, all_generations, json_rates, tag_errors):
                        val_samples.append([prompt, generation,  f"{json_rate}%", f"{tag_error}"])
                    if val_generations_logger is not None and tracking is not None:
                        val_generations_logger.log(tracking.logger.keys(), val_samples, global_step)
                
                if tracking is not None:
                    tracking.log(data=metric, step=global_step)
            torch.distributed.barrier()

            # save checkpoint
            self.save_checkpoint(step=global_step)

 
@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)
    trainer.fit()


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
        # ONLY USES MULTI-TURN DATASET FOR VALIDATION AS OF 07/09/25
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()