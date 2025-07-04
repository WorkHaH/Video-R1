cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"


CUDA_VISIBLE_DEVICES=3,4,5,6,7 torchrun --nproc_per_node="5" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_sarcasm_s1.py \
    --output_dir "./log/Qwen2.5-VL-7B-sarcasm-sft-s1" \
    --model_name_or_path "/data/guojian.li/Weight/Qwen2.5-VL-7B-Instruct/" \
    --dataset_name "/data/guojian.li/Dataset/MMSD/text_json_final/train.json" \
    --deepspeed local_scripts/zero3.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-7B-sarcasm-sft-s1 \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model true \
    # --report_to wandb \