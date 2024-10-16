import os
import json
import numpy as np
from PIL import Image  

categories = [
    {"id": 0, "name": "Person"},
    {"id": 1, "name": "Ear"},
    {"id": 2, "name": "Earmuffs"},
    {"id": 3, "name": "Face"},
    {"id": 4, "name": "Face-guard"},
    {"id": 5, "name": "Face-mask-medical"},
    {"id": 6, "name": "Foot"},
    {"id": 7, "name": "Tools"},
    {"id": 8, "name": "Glasses"},
    {"id": 9, "name": "Gloves"},
    {"id": 10, "name": "Helmet"},
    {"id": 11, "name": "Hands"},
    {"id": 12, "name": "Head"},
    {"id": 13, "name": "Medical-suit"},
    {"id": 14, "name": "Shoes"},
    {"id": 15, "name": "Safety-suit"},
    {"id": 16, "name": "Safety-vest"},
]

images_dir = "valid/images"  
labels_dir = "valid/labels"  
output_json = "valid/images/annotations.json"  

coco_format = {
    "images": [],
    "annotations": [],
    "categories": categories,
}

annotation_id = 0

for image_id, filename in enumerate(os.listdir(images_dir)):
    if not filename.endswith(('.jpg', '.jpeg', '.png')): 
        continue
    
    img_path = os.path.join(images_dir, filename)
    img = Image.open(img_path)
    width, height = img.size

    image_info = {
        "id": image_id,
        "file_name": filename,
        "height": height,
        "width": width,
    }
    coco_format["images"].append(image_info)

    label_file = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt')
    if not os.path.exists(label_file):
        continue

    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            category_id = int(parts[0])  
            
            x_center, y_center, width_ratio, height_ratio = map(float, parts[1:])
             # Convert YOLO format to COCO format
            x1 = max(0, (x_center - width_ratio / 2))
            y1 = max(0, (y_center - height_ratio / 2))
            x2 = min(1, (x_center + width_ratio / 2))
            y2 = min(1, (y_center + height_ratio / 2))

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1 * width, y1 *height, (x2 - x1) * width, (y2 - y1) * height],
                "area": width * height * width * height,
                "segmentation": [],
                "iscrowd": 0
            }
            coco_format["annotations"].append(annotation)
            annotation_id += 1

with open(output_json, 'w') as json_file:
    json.dump(coco_format, json_file, indent=4)

print(f"COCO：{output_json}")
