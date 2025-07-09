cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"


CUDA_VISIBLE_DEVICES=3,4,5,7 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12348" \
    src/open_r1/sft_lora_sarcasm_s2.py \
    --output_dir "./log/Qwen2.5-VL-7B-sarcasm-sft-lora-s4" \
    --model_name_or_path "/data/guojian.li/Weight/Qwen2.5-VL-7B-Instruct/" \
    --dataset_name "" \
    --deepspeed local_scripts/zero3.json \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-7B-sarcasm-sft-lora-s4 \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model true \
    --use_peft true \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05
    # --report_to wandb \ 

