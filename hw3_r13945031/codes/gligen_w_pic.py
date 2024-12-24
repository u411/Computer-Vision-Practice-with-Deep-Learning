import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
import json
from PIL import Image
import os

# 加載 JSON 文件
with open("label_with_prompts.json", "r") as file:
    data = json.load(file)

# 輸出文件夾
output_folder = "./generation"
os.makedirs(output_folder, exist_ok=True)

# Bounding box 正規化函數
def normalize_bbox(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    return [
        x_min / img_width,
        y_min / img_height,
        x_max / img_width,
        y_max / img_height,
    ]

# 加載 GLIGEN Pipeline
pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # 使用 Stable Diffusion 模型
        torch_dtype=torch.float16
    )
pipe = pipe.to("cuda")

# 遍歷 JSON 條目，生成圖片
for entry in data:
    image_path = os.path.join("images", entry["image"])   # 原始參考圖片路徑
    phrases = entry["labels"]  # 物體的類別
    bboxes = entry["bboxes"]  # bounding boxes
    image_width, image_height = entry["width"], entry["height"]
    normalized_bboxes = [normalize_bbox(bbox, image_width, image_height) for bbox in bboxes]
    prompt = entry["prompt_w_suffix"]  # 生成的文本描述

    # 加載參考圖片
    try:
        reference_image = load_image(image_path)  # 使用 diffusers 提供的 load_image
    except Exception as e:
        #print(f"Failed to load reference image {image_path}: {e}")
        continue

    # 使用 GLIGEN 管線生成圖像
    images = pipe(
    prompt=prompt,
    gligen_phrases=phrases,
    init_image=reference_image,  # 載入參考圖片
    strength=0.95,  # 設定影響參考圖片的強度
    guidance_scale=7,  # 控制生成圖片的文本依從性
    num_inference_steps=30,
).images

    # 保存生成圖片
    original_name = os.path.basename(image_path)  # 提取原始圖片名稱
    output_path = os.path.join(output_folder, original_name)  # 生成輸出路徑
    images[0].save(output_path)  # 保存圖片
    #print(f"Generated image saved as: {output_path}")
