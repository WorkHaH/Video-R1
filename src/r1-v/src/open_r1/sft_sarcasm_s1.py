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
    --gradient_checkpointing
"""

import os
import json
import random
import requests
import torch
import re
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict

import wandb

from typing import List, Dict, Any

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def download_video(url: str, folder: str = '/tmp/videos/') -> str:
    """Download video if not already present locally."""
    filename = url.split("/")[-1]
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}")
    
def extract_sarcasm_output(text: str) -> dict:
    """
    Extracts information from the XML-like output format for satire detection.

    Args:
        text (str): The text output from the model.

    Returns:
        dict: A dictionary containing 'think', 'bboxes', and 'answer'.
              Returns empty values if parts are not found.
    """
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    bbox_match = re.search(r"<bbox_2d>(.*?)</bbox_2d>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    think = think_match.group(1).strip() if think_match else ""
    bbox_str = bbox_match.group(1).strip() if bbox_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    bboxes = []
    if bbox_str.lower() != "none":
        try:
            # The format is '[x1,y1,x2,y2],[x1,y1,x2,y2]'
            bbox_pattern = re.compile(
                r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
            )
            matches = bbox_pattern.findall(bbox_str)
            for match in matches:
                box = [int(c) for c in match]
                if box[0] < box[2] and box[1] < box[3]:  # filter negative bbox
                    bboxes.append(box)
        except Exception:
            # If parsing fails, return empty list of bboxes
            bboxes = []

    return {"think": think, "bboxes": bboxes, "answer": answer.lower()}

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""

    system_message = (
    "You are a multimodal satire analysis assistant.\n"
    "Please output strictly in accordance with the following structure:\n"
    "'<think>Natural Language Interpretation</think> <bbox_2d>coordinates or none</bbox_2d> <answer>sarcasm/non-sarcasm</answer> '\n\n"
    
    "Explanation:\n"
    "1. In <think> Explain 'Why judge (not) sarcasm'.\n"
    "2. <bbox_2d> section:\n"
        "If answer= sarcasm, give the coordinates of satirical target '[x1,y1,x2,y2] ,[x1,y1,x2,y2]'. "
        "The target of satire may be more than one, or there may be none. If not, write single token 'none'.\n"
        "If answer= non-sarcasm, write single token 'none'."
    "3. Only output the above three tags. The order of the tags cannot be changed. The tag name, Angle brackets, and slashes must be complete.\n\n"
    
    "The following are some relevant examples:\n"
    "1.An example of satire:\n"
        "<think>The description of the environment in the text does not match the promotion, and irony is used to express satire.</think> "
        "<bbox_2d>[23,45,44,78],[10,20,32,40]</bbox_2d> <answer>sarcasm</answer> \n"
    "2.An example of non-sarcasm:\n"
        "<think>The text normally describes the feelings brought by fine weather, without any sarcasm.</think> "
        "<bbox_2d>none</bbox_2d> <answer>non-sarcasm</answer> "
)

    QUESTION_TEMPLATE =   (
            "Text:{text}\n"
            "Please judge whether this picture and text are satirical. If so, point out the location of the satirical target and give the reason."
            "There are only two categories: sarcasm and non-sarcasm. Just output it in the prescribed format. No extra text is needed."
       )

    answer_text = "sarcasm" if example['label'] == 1 else "non-sarcasm"
    # Create a templated response as we don't have ground truth for think/bboxes.
    # The model will learn to generate this structure.
    # For non-sarcasm, bboxes is 'none' according to the prompt. We'll use 'none' for sarcasm too as we lack data.
    think_text = "The model provides its reasoning here."
    bboxes_text = "none"

    assistant_response = f"<think>{think_text}</think> <bbox_2d>{bboxes_text}</bbox_2d> <answer>{answer_text}</answer> "

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
                    "image":  os.path.join(IMG_DIR, f"{example['image_id']}.jpg")
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(text=example['text']) 
                }
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_response}]
        }
    ]
    

    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate batch of examples for training.
    This function first locates the assistant's response, and then masks everything in that response
    except for the hardcoded answer part.
    """
    # Each 'example' is a dict with a 'messages' key from prepare_dataset
    texts = [processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False) for ex in examples]
    
    # This part remains the same: process vision info for all examples
    image_inputs_list = []
    video_inputs_list = []
    for example in examples:
        image_inputs, video_inputs, _ = process_vision_info(example["messages"], return_video_kwargs=True)
        image_inputs_list.extend(image_inputs or [])
        if video_inputs:
            video_inputs_list.extend(video_inputs or [])

    # Tokenize all texts at once
    inputs = processor(
        text=texts,
        images=image_inputs_list,
        videos=video_inputs_list if video_inputs_list else None,
        return_tensors="pt",
        padding=True
    )

    labels = inputs["input_ids"].clone()

    # Pre-tokenize the assistant marker and the two exact answer sequences.
    # Note the spaces, matching the format in `prepare_dataset`.
    assistant_marker_ids = processor.tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
    sarcasm_ids = processor.tokenizer.encode(" <answer>sarcasm</answer> ", add_special_tokens=False)
    non_sarcasm_ids = processor.tokenizer.encode(" <answer>non-sarcasm</answer> ", add_special_tokens=False)

    for i, example in enumerate(examples):
        input_ids_list = inputs["input_ids"][i].tolist()

        # --- Stage 1: Find the start of the assistant's response ---
        response_start_index = -1
        for k in range(len(input_ids_list) - len(assistant_marker_ids) + 1):
            if input_ids_list[k:k + len(assistant_marker_ids)] == assistant_marker_ids:
                response_start_index = k + len(assistant_marker_ids)
                break
        
        if response_start_index == -1:
            print(f"Warning: Assistant marker not found for example {i}. Masking entire example.")
            labels[i, :] = -100
            continue

        # --- Stage 2: Find the target answer sequence *within* the response ---
        assistant_response_str = example["messages"][-1]["content"][0]["text"]
        target_ids = non_sarcasm_ids if "non-sarcasm" in assistant_response_str else sarcasm_ids

        target_start_in_response = -1
        response_ids = input_ids_list[response_start_index:]
        for k in range(len(response_ids) - len(target_ids) + 1):
            if response_ids[k:k+len(target_ids)] == target_ids:
                target_start_in_response = k
                break
        
        # --- Stage 3: Apply two-stage masking ---
        # First, mask everything up to the assistant's response (exclusive of the response itself).
        # The prompt part of the labels should remain the same as input_ids.
        # So we only start masking from the response start.
        labels[i, response_start_index:] = -100

        if target_start_in_response != -1:
            # If the target sequence was found, un-mask it.
            global_target_start = response_start_index + target_start_in_response
            # global_target_end = global_target_start + len(target_ids)
            # labels[i, global_target_start:global_target_end] = inputs["input_ids"][i, global_target_start:global_target_end]
            labels[i, global_target_start:] = inputs["input_ids"][i, global_target_start:]
        else:
            # If the target is not found, mask the entire assistant response.
            print(f"Warning: Target answer sequence not found within the response for example {i}.")
            labels[i, response_start_index:] = -100

    # Mask padding and visual tokens for safety
    labels[labels == processor.tokenizer.pad_token_id] = -100

    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else []
    if hasattr(processor, "image_token") and processor.image_token:
        # For Qwen2VLProcessor, tokenizer is part of the processor itself. For others, it's processor.tokenizer
        tokenizer_obj = processor if isinstance(processor, Qwen2VLProcessor) else processor.tokenizer
        image_token_id = tokenizer_obj.convert_tokens_to_ids(processor.image_token)
        if image_token_id is not None:
             visual_tokens.append(image_token_id)

    for visual_token_id in visual_tokens:
        if visual_token_id is not None:
            labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs

def set_model(model_args, model):
    if model_args["tune_mm_vision"]:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args["tune_mm_mlp"]:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args["tune_mm_llm"]:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser([ScriptArguments, SFTConfig, ModelConfig])
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # # Quantization configuration for 4-bit training
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        # device_map=get_kbit_device_map(), # workhah,用zero3需要注释
        # quantization_config=bnb_config,
    )
    
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    model_args = {
        "tune_mm_vision": False,
        "tune_mm_mlp": True,
        "tune_mm_llm": True
    }
    set_model(model_args, model)

    # 路径配置
    IMG_DIR = "/data/guojian.li/Dataset/MMSD/dataset_image/"

    # Prepare dataset
    prepared_dataset = [prepare_dataset(example) for example in dataset['train']]


    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer
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
    wandb.finish()
