import os
import json
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from transformers import BitsAndBytesConfig

# 设置模型配置
bnb_config = BitsAndBytesConfig(load_in_4bit=False, load_in_8bit=True)

# 清空GPU缓存
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载模型和处理器
try:
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", 
                                                           revision="51572668da0eb669e01a189dc22abe6088589a24", 
                                                           load_in_8bit=True, 
                                                           device_map="auto", 
                                                           torch_dtype=torch.float16 if device == 'cuda' else torch.float32)
    print("Model and Processor loaded successfully")
except Exception as e:
    print(f"Error loading model/processor: {e}")

def generate_caption(image_path):
    try:
        # 打开图片并转换为RGB格式
        image = Image.open(image_path).convert("RGB")
        image = image.resize((384, 384))  # 调整图像大小
        inputs = processor(image, return_tensors="pt").to(device, torch.float16)
        
        # 初始化 input_ids (确保有文本起始符号和影像特殊 token)
        start_tokens = [[processor.tokenizer.bos_token_id]]  # 文本开始符号
        image_token_index = getattr(model.config, "image_token_index", None)

        # 如果 image_token_index 不存在，手动添加
        if image_token_index is None:
            special_tokens_dict = {'additional_special_tokens': ["<image>"]}
            image_token_index = processor.tokenizer.add_special_tokens(special_tokens_dict)
            model.config.image_token_index = image_token_index  # 更新模型配置

        # 在 start_tokens 中加入影像 token
        start_tokens[0].append(image_token_index)

        # 確保 input_ids 和 pixel_values 移動到正確的設備
        input_ids = torch.tensor(start_tokens, dtype=torch.long, device=device)

        # 生成描述
        caption = model.generate(
            input_ids=input_ids, 
            pixel_values=inputs['pixel_values']
        )
        
        # 解码生成的文本
        return processor.batch_decode(caption, skip_special_tokens=True)[0].strip()
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return ""


# 读取图像文件路径
image_dataset = [os.path.join("images", img) for img in os.listdir("images") if img.endswith(('.jpg', '.png', '.jpeg'))]

# 读取原始的label.json文件
with open('label.json', 'r') as file:
    data = json.load(file)

# 为每张图片生成描述
for entry in data:
    img_path = entry['image']
    full_image_path = os.path.join('images', img_path)
    
    # 确保图像路径存在
    if os.path.exists(full_image_path):
        generated_text = generate_caption(full_image_path)
        print(f"{full_image_path}: {generated_text}")


        # 将生成的描述添加到标签中
        entry['generated_text'] = generated_text
    else:
        print(f"Image path does not exist: {full_image_path}")

# 如果需要保存修改后的数据
with open('label_with_generated_text_V2.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)
