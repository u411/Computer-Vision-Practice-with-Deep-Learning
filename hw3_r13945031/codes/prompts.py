import os
import json
from collections import Counter
from tqdm import tqdm

# 自訂後綴文字
custom_suffix = "in a construction site."

# 加載 label.json 文件
label_file = 'label_with_generated_text.json'
if not os.path.exists(label_file):
    print(f"Label file does not exist: {label_file}")
    exit(1)

# 讀取 JSON 數據
with open(label_file, 'r') as file:
    data = json.load(file)

# 處理每個條目，生成 prompt_w_label 和 prompt_w_suffix
for entry in tqdm(data, desc="Processing entries"):
    # 獲取生成的文本描述
    generated_text = entry.get('generated_text', "").strip()
    labels = entry.get('labels', [])
    bboxes = entry.get('bboxes', [])  # 假設 bbox 為 [(x_min, y_min, x_max, y_max), ...]
    image_height = entry.get('height', None)
    image_width = entry.get('width', None)
    
    # 確保 generated_text 末尾有句號
    if generated_text and not generated_text.endswith('.'):
        generated_text += '.'
    
    # 統計每種物品的數量
    label_counts = Counter(labels)
    if label_counts:
        object_count = len(label_counts)  # 物品種類數量
        label_details = []
        for name, count in label_counts.items():
            label_details.append(f"{count} {name}" + ("s" if count > 1 else ""))
        
        label_text = ", ".join(label_details[:-1]) + (f", and {label_details[-1]}" if len(label_details) > 1 else label_details[0])
        object_count_text = f"This photo contains {label_text}."
    else:
        object_count_text = ""

    # 找出最大主體
    largest_subjects = []
    subject_positions = []
    if labels and bboxes and len(labels) == len(bboxes):
        # 計算每個物體的面積
        areas = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            area = (x_max - x_min) * (y_max - y_min)
            areas.append(area)
        
        # 找到最大面積
        max_area = max(areas)
        threshold = max_area * 0.9  # 設置相似閾值 (例如，面積差在10%內算相近)
        
        # 找出所有接近最大面積的物體
        for label, bbox, area in zip(labels, bboxes, areas):
            if area >= threshold:
                largest_subjects.append(label)
                
                # 計算主體中心點
                x_min, y_min, x_max, y_max = bbox
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                # 確定九宮格位置
                row = int(center_y // (image_height / 3)) + 1
                col = int(center_x // (image_width / 3)) + 1
                position = (row - 1) * 3 + col
                subject_positions.append(position)
        
        # 去除重複項
        largest_subjects = list(set(largest_subjects))
        subject_positions = list(set(subject_positions))
    
    # 最大主體的描述
    if largest_subjects:
        largest_text = "The main subject" + ("s are " if len(largest_subjects) > 1 else " is ") + ", ".join(largest_subjects) + ","
    else:
        largest_text = ""
    
    # 主體位置描述
    if subject_positions:
        position_text = "appears in " + ", ".join([f"position {pos}" for pos in sorted(subject_positions)]) + " of the 9-grid layout."
    else:
        position_text = ""
    
    # 圖片尺寸的描述
    if image_height and image_width:
        size_text = f"sized {image_width}x{image_height}"
    else:
        size_text = "with an unknown size"

    # 組合 prompt_w_label
    prompt_w_label = f"{generated_text} {object_count_text}"

    # 在 prompt_w_label 基礎上生成 prompt_w_suffix
    prompt_w_suffix = f"{generated_text} {largest_text} {position_text} {object_count_text} {custom_suffix}"

    # 更新 entry
    entry['generated_text'] = generated_text
    entry['prompt_w_label'] = prompt_w_label
    entry['prompt_w_suffix'] = prompt_w_suffix

# 保存修改後的 JSON 到新文件
output_file = 'label_with_prompts_t3.json'
with open(output_file, 'w') as outfile:
    json.dump(data, outfile, indent=4, ensure_ascii=False)  # 使用 ensure_ascii=False 保留中文
    print(f"Updated label file saved to {output_file}")
