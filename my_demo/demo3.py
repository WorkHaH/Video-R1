import os
import json
import re
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def inference(image_path, prompt, sys_prompt="You are a helpful assistant", max_new_tokens=4096, return_input=False):
    image = Image.open(image_path)
    image_local_path = "file://" + image_path
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"image": image_local_path},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print("text:", text)
    # image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to('cuda:1')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if return_input:
        return output_text[0], inputs
    else:
        return output_text[0]

# 路径配置
MODEL_PATH = "/data/guojian.li/Weight/Qwen2.5-VL-7B-Instruct/"
IMG_DIR = "/data/guojian.li/Dataset/MMSD/dataset_image/"
DATA_JSON = "/data/guojian.li/Dataset/MMSD/text_json_final/test.json"

# 加载模型与处理器
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, 
                                                           attn_implementation="flash_attention_2",device_map=device)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def predict_sarcasm(image_path, text):
    # image = Image.open(image_path).convert("RGB")
    prompt = (
        # 'Look at the image and read the text below. '
        # 'Is there any sarcasm or irony present in the combination of the image and the text? '
        # 'Answer only with "Yes" or "No".\n\n'
        # f'Text: {text1}'
        'I will now provide you with an image and a text. Your task is to analyze the image-text pair and determine if it is sarcastic or non-sarcastic. '
        'You need think step-by-step and analyze the relationship between the text and image carefully.'
        # 'It can be analyzed according to the following steps:\n' 
        #     '1. Extract the core semantics of the image and the text respectively;\n' 
        #     '2. Compare and analyze the differences in their meanings;\n'
        #     '3. Determine whether there is a satirical intention\n\n'
        'Explain your analysis in <think></think> and output the final result in <answer></answer>.If there is a sarcastic answer, The content within <answer></answer> is "Yes". Your final answer should be a JSON object in the following format:\n'
            '<think> your reasoning content </think> <answer>Yes or No</answer> \n\n'
        f'The provided text is as follows:{text}'
    )
    # inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    # outputs = model.generate(**inputs, max_new_tokens=20)
    # response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    response = inference(image_path, prompt)

    # 简单的输出判定，可根据实际微调
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    answer = ''
    if match:
        answer = match.group(1).strip()
   
    if "Yes" in answer:
        return 1
    else:
        return 0

def main():
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    y_true, y_pred = [], []
    for item in tqdm(data):
        image_id = item["image_id"]
        text = item["text"]
        label = item["label"]
        image_path = os.path.join(IMG_DIR, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            continue
        pred_label = predict_sarcasm(image_path, text)
        y_true.append(label)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"准确率: {acc:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

if __name__ == "__main__":
    main()
