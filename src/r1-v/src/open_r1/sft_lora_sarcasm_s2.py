# Copyright 2024. All rights reserved.
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
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    train_video_llm.py \
    --dataset_name mfarre/simplevideoshorts \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir video-llm-output \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 8 \
    --lora_alpha 16
"""

import os
import json
import random
import requests
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.models.qwen2_vl import (
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
)
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from peft import LoraConfig

from datasets import Dataset, DatasetDict

import wandb

from typing import List, Dict, Any

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def load_gt_boxes(label_file):
    gt_boxes = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            i = 0
            while i + 4 < len(parts):
                x1, y1, x2, y2 = map(float, parts[i:i+4])
                gt_boxes.append([x1, y1, x2, y2])
                i += 5  # 跳过class字段
    return gt_boxes

def load_text(text_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""

    system_message = "You are a helpful assistant"

    QUESTION_TEMPLATE =  ('Please combine the content of the tweet and the pictures to identify the targets being sarcastic in the pictures. '
    'The target of satire may be more than one, or there may be none. '
    'If they exist, output the pixel coordinates of all the targets,'
    'in formats such as [{{"bbox_2d":[x1,y1,x2,y2]}},{{"bbox_2d":[x1,y1,x2,y2]}}]. If it does not exist, return [].'
    'Try to position each target separately as much as possible. The answer only provides the coordinates.'
    '\nThe provided text is as follows:\n{text}')

    labels = []
    for label in example['label']:
        if label[0] == 0 and label[1] == 0 and label[2] == 0 and label[3] == 0:
            break;
        else:
            labels.append({
                "bbox_2d": label
            })

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image":  f"{example['image_id']}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(text=example['text']) 
                }
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(labels)}]
        }
    ]
    

    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    image_inputs, video_inputs = [], []

    for i, example in enumerate(examples):
        try:
            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            cur_image_inputs, cur_video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True)
            if cur_image_inputs:
                image_inputs.extend(cur_image_inputs)
            if cur_video_inputs:
                video_inputs.extend(cur_video_inputs)
            
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser([ScriptArguments, SFTConfig, ModelConfig])
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # if model_config.use_peft:
    #     model_config.lora_modules_to_save = ["merger", "lm_head"]


    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    IMAGE_DIR = "/data/guojian.li/Dataset/MSTI/img/"
    TEXT_DIR = "/data/guojian.li/Dataset/MSTI/Textual sentences/train/"
    VISUAL_LABEL_DIR = "/data/guojian.li/Dataset/MSTI/Visual target labels/"
    
    # Prepare dataset
    prepared_dataset = []

    text_files = [f for f in os.listdir(TEXT_DIR) if f.endswith('.txt')]
    print(f'共{len(text_files)}数据')
    for text_file in text_files:
        imgid = os.path.splitext(text_file)[0]
        img_path = os.path.join(IMAGE_DIR, f'{imgid}.jpg')
        label_path = os.path.join(VISUAL_LABEL_DIR, text_file)
        text_path = os.path.join(TEXT_DIR, f'{imgid}.txt')
        if not (os.path.exists(img_path) and os.path.exists(text_path)):
            continue
        gt_boxes = load_gt_boxes(label_path)
        text = load_text(text_path)
        prepared_dataset.append(prepare_dataset({
            "image_id": img_path,
            "text": text,
            "label": gt_boxes
        }))

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # Build PEFT config
    peft_config = None
    if model_config.use_peft:
        # target_modules = [
        #     r".*model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
        #     r".*model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)",
        # ]
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            bias="none",
            target_modules=find_target_linear_names(model,  lora_namespan_exclude=["visual","embed_tokens"]),
            modules_to_save=["merger"], # 全量微调
        )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
    )

    # Train model
    trainer.train()

     # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    if training_args.report_to == "wandb":
        wandb.finish() 