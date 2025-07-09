from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
import torch

# 设置目标 GPU 设备
device = torch.device("cuda:7")

# 路径配置
base_model_path = "/data/guojian.li/Weight/Qwen2.5-VL-7B-Instruct/"
lora_model_path = "/data/guojian.li/Project/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-sarcasm-sft-lora-s2"
merged_model_path = "/data/guojian.li/Project/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-sarcasm-sft-lora-s2-merged"

# 保存tokenizer（无需移动到GPU）
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(merged_model_path)

# 加载基础模型并移动到指定GPU
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    trust_remote_code=True
).to(device)

# 加载LoRA适配器并应用到基础模型
model = PeftModel.from_pretrained(model, lora_model_path)

# 合并LoRA权重到基础模型，并卸载LoRA组件
model = model.merge_and_unload()

# 将合并后的模型保存
model.save_pretrained(merged_model_path)

print("✅ 合并完成，模型已保存到：", merged_model_path)
