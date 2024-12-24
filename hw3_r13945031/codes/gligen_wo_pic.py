import torch
from diffusers import StableDiffusionGLIGENPipeline
from diffusers.utils import load_image
import json
from PIL import Image
import os

with open("label_with_prompts_t3.json", "r") as file:
    data = json.load(file)

# 提取信息
# 遍历 label.json 中的每个条目

output_folder = "./generation_30"  #  generation_gen /generation_label / generation_suffix
os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在

def normalize_bbox(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    return [
        x_min / img_width,
        y_min / img_height,
        x_max / img_width,
        y_max / img_height,
    ]

# Generate an image described by the prompt and
# insert objects described by text at the region defined by bounding boxes
pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    "masterful/gligen-1-4-generation-text-box", variant="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")


for entry in data:
    image_path = entry["image"]  # 原始图片路径
    phrases = entry["labels"]  # 物体的类别
    bboxes = entry["bboxes"]  # bounding boxes
    image_width, image_height = entry["width"], entry["height"]
    normalized_bboxes = [normalize_bbox(bbox, image_width, image_height) for bbox in bboxes]
    prompt = entry["generated_text"]  #  generated_text/prompt_w_label/prompt_w_suffix

    images = pipe(
        prompt=prompt,
        gligen_phrases=phrases,
        gligen_boxes=bboxes,
        gligen_scheduled_sampling_beta=1,
        output_type="pil",
        num_inference_steps=30,
    ).images
    original_name = os.path.basename(image_path)  # 提取原始图片名称
    output_path = os.path.join(output_folder, original_name)  # 生成输出路径

    images[0].save(output_path)  # 保存图片
    print(f"Generated image saved as: {output_path}")