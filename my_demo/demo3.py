import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.models.auto.processing_auto import AutoProcessor
import ast

# ========== 配置 ==========
TEXT_DIR = '/data/guojian.li/Dataset/MSTI/Textual sentences/test'  # 假设文本在此目录
VISUAL_LABEL_DIR = '/data/guojian.li/Dataset/MSTI/Visual target labels/'
IMAGE_DIR = '/data/guojian.li/Dataset/MSTI/img'
# MODEL_PATH = '/data/guojian.li/Weight/Qwen2.5-VL-7B-Instruct'
MODEL_PATH = '/data/guojian.li/Project/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-sarcasm-sft-lora-s3'
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'


# ========== VOC AP工具 ==========
def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(preds, targets, threshold=0.5, use_07_metric=False):
    if len(preds) == 0:
        return 0.0
    image_ids = [x[0] for x in preds]
    confidence = np.array([float(x[1]) for x in preds])
    BB = np.array([x[2:] for x in preds])
    if BB.size == 0 or BB.ndim != 2:
        return 0.0
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    npos = sum([len(targets[k]) for k in targets])
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d, image_id in enumerate(image_ids):
        bb = BB[d]
        if image_id in targets:
            BBGT = [gt for gt in targets[image_id]]
            match = False
            for bbgt in BBGT:
                ixmin = np.maximum(bbgt[0], bb[0])
                iymin = np.maximum(bbgt[1], bb[1])
                ixmax = np.minimum(bbgt[2], bb[2])
                iymax = np.minimum(bbgt[3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (bbgt[2] - bbgt[0] + 1.) * (bbgt[3] - bbgt[1] + 1.) - inters
                overlaps = inters / union if union > 0 else 0
                if overlaps > threshold:
                    tp[d] = 1
                    match = True
                    targets[image_id].remove(bbgt)
                    break
            fp[d] = 0 if match else 1
        else:
            fp[d] = 1
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos) if npos > 0 else np.zeros_like(tp)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return ap

# ========== 工具函数 ==========
def load_gt_boxes(label_file):
    gt_boxes = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            i = 0
            while i + 3 < len(parts):
                step = 5 if i + 4 < len(parts) else 4
                x1, y1, x2, y2 = map(float, parts[i:i+4])
                 # 目标全为0也跳过
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                    i += step
                    continue
                gt_boxes.append([x1, y1, x2, y2])
                i += step
    return gt_boxes

def load_text(text_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def parse_qwen_output(output):
    try:
        if '```json' in output:
            output = output.split('```json')[1].split('```')[0]
        boxes = ast.literal_eval(output)
        return [b['bbox_2d'] for b in boxes if 'bbox_2d' in b]
    except Exception as e:
        print(output)
        print('Qwen输出解析失败:', e)
        return []

def qwen_infer(image_path, text, prompt, model, processor):
    image = Image.open(image_path).convert('RGB')
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": [
            {"type": "text", "text": prompt + "\nThe provided text is as follows:\n" + text},
            {"image": image_path}
        ]}
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], images=[image], padding=True, return_tensors="pt").to(DEVICE)
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

# ========== 主流程 ==========
def main():
    print('加载Qwen2.5-VL模型...')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    text_files = [f for f in os.listdir(TEXT_DIR) if f.endswith('.txt')]
    print(f'共{len(text_files)}张数据')

    num_images = 0 # 实际图片数量

    gt_dict = {}
    pred_list = []
    prompt = ('Please combine the content of the tweet and the pictures to identify the targets being sarcastic in the pictures. '
    'The target of satire may be more than one, or there may be none. '
    'If they exist, output the pixel coordinates of all the targets,'
    'in formats such as [{"bbox_2d":[x1,y1,x2,y2]},{"bbox_2d":[x1,y1,x2,y2]}]. If it does not exist, return [].'
    'The answer only provides the coordinates.')
    
    for text_file in tqdm(text_files):
        imgid = os.path.splitext(text_file)[0]
        img_path = os.path.join(IMAGE_DIR, f'{imgid}.jpg')
        label_path = os.path.join(VISUAL_LABEL_DIR, text_file)
        text_path = os.path.join(TEXT_DIR, f'{imgid}.txt')
        if not (os.path.exists(img_path) and os.path.exists(text_path) and os.path.exists(label_path)):
            continue
        gt_boxes = load_gt_boxes(label_path)
        num_images += 1
        gt_dict[imgid] = [box for box in gt_boxes]
        text = load_text(text_path)
        output = qwen_infer(img_path, text, prompt, model, processor)
        
        with open("./data/output3.txt", "a", encoding="utf-8") as file:
            file.write("\n1." + output)  # 追加一行
            file.write("\n2." + str(gt_boxes))  

        pred_boxes = parse_qwen_output(output)
        try:
            pred_boxes = [[float(x) for x in box] for box in pred_boxes]
        except Exception as e:
            print(pred_boxes)
            print('坐标错误:', e)
            continue
        for pb in pred_boxes:
            pred_list.append([imgid, 1.0, pb[0], pb[1], pb[2], pb[3]])
           

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aps = []
    for th in thresholds:
        aps.append(voc_eval(pred_list, {k: [b for b in v] for k, v in gt_dict.items()}, threshold=th))
    ap = np.mean(aps)
    ap50 = aps[0]
    ap75 = aps[5]
    print(f'处理了{num_images}张图片')
    print(f'AP: {ap:.4f}')
    print(f'AP50: {ap50:.4f}')
    print(f'AP75: {ap75:.4f}')

if __name__ == '__main__':
    main()