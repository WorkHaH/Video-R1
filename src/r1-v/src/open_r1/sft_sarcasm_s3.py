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
This script fine-tunes a model to produce a complex output format,
but calculates loss only on the <answer> part.
The masking logic is handled entirely within a custom compute_loss method.
"""

import os
import requests
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
import wandb
from typing import List, Dict, Any

class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overrides the default loss computation.
        This method masks the prompt and the parts of the response before the <answer> tag,
        ensuring loss is only calculated on the final answer.
        """
        # The `labels` are currently identical to `input_ids`. We need to mask them.
        labels = inputs.get("labels").clone()
        
        # Get the token sequence that marks the start of an assistant's response.
        # For Qwen2-VL, this is typically '<|im_start|>assistant\n'.
        # We tokenize it without special tokens to get the pure IDs.
        assistant_prompt_ids = self.tokenizer.encode(
            self.tokenizer.apply_chat_template([{"role": "assistant", "content": ""}], tokenize=False, add_generation_prompt=True)[len(self.tokenizer.bos_token):], 
            add_special_tokens=False
        )

        for i in range(labels.shape[0]):
            label_row = labels[i]
            
            # Find the assistant prompt sequence in the labels
            assistant_start_index = -1
            for k in range(len(label_row) - len(assistant_prompt_ids) + 1):
                if label_row[k:k+len(assistant_prompt_ids)].tolist() == assistant_prompt_ids:
                    assistant_start_index = k
                    break

            if assistant_start_index != -1:
                # Mask everything up to and including the assistant prompt
                labels[i, :assistant_start_index + len(assistant_prompt_ids)] = -100
            else:
                # Fallback if the prompt isn't found (should not happen with correct data)
                # To be safe, we don't mask this row and it will compute loss on everything.
                print(f"Warning: Assistant prompt sequence not found for example {i}.")

        # The model computes the loss internally if labels are provided.
        # We pass our dynamically masked labels to the model.
        outputs = model(**inputs, labels=labels)
        
        # The loss is the first item in the output tuple
        loss = outputs[0]

        return (loss, outputs) if return_outputs else loss

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Prepares the dataset.
    The system prompt asks for a complex format, but the assistant's
    ground-truth response only contains the <answer> tag.
    """
    system_message = (
        "You are a multimodal satire analysis assistant.\n"
        "Please output strictly in accordance with the following structure:\n"
        "'<think> Natural Language Interpretation </think><bbox_2d> coordinates or none</bbox_2d><answer> sarcasm/non-sarcasm </answer>'"
    )

    QUESTION_TEMPLATE = (
        "Text:{text}\n"
        "Please judge whether this picture and text are satirical. If so, point out the location of the satirical target and give the reason."
        "There are only two categories: sarcasm and non-sarcasm. Just output it in the prescribed format. No extra text is needed."
    )
    
    answer_text = "sarcasm" if example['label'] == 1 else "non-sarcasm"
    # The ground truth label ONLY contains the answer part.
    assistant_response = f"<answer> {answer_text} </answer>"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [{"type": "image", "image": os.path.join(IMG_DIR, f"{example['image_id']}.jpg")}, {"type": "text", "text": QUESTION_TEMPLATE.format(text=example['text'])}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}
    ]

    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    A simplified collator. It tokenizes data but performs NO MASKING.
    Masking is delegated entirely to the custom compute_loss method.
    """
    texts = [processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False) for ex in examples]
    
    image_inputs_list = []
    for ex in examples:
        image_inputs, _, _ = process_vision_info(ex["messages"], return_video_kwargs=True)
        if image_inputs:
            image_inputs_list.extend(image_inputs)

    inputs = processor(text=texts, images=image_inputs_list, return_tensors="pt", padding=True)

    # Critical: As per the new strategy, labels are identical to input_ids.
    # The custom compute_loss will handle all masking.
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

if __name__ == "__main__":
     # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
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

    # 路径配置
    IMG_DIR = "/data/guojian.li/Dataset/MMSD/dataset_image/"

    # Prepare dataset
    prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training-sarcasm-s3")

    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer # Pass tokenizer to trainer
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    del model, trainer
    torch.cuda.empty_cache()
    if training_args.report_to == "wandb":
        wandb.finish() 