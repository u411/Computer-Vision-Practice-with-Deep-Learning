from transformers import AutoModelForObjectDetection, TrainingArguments, Trainer, EvalPrediction, DetrForObjectDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from torchvision.ops import nms
import torch
from pathlib import Path
from datasets import Dataset
from PIL import ImageFile
import argparse, os, json
import gc
from util import *
from dataset import MyInferDataset, MyDataset

parser = argparse.ArgumentParser()
parser.add_argument('--test', default='exp', help='test images dir')
parser.add_argument('--json_name', default='pred.json', help='json name')
args = parser.parse_args()
test_path = Path(args.test)
test_set = MyInferDataset(test_path)
checkpoint_path = './runs4/checkpoint-25000'  # best model checkpoint
image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint_path,
    ignore_mismatched_sizes=True,
).to(device="cuda")

rlt = {f: {"boxes": [], "labels": []} for f in os.listdir(args.test) if f.endswith(('.jpg', '.jpeg', '.png'))}
model.eval()
batch_size = 800  
num_batches = (len(test_set) + batch_size - 1) // batch_size 

with torch.no_grad():
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(test_set))

        for i in range(start_idx, end_idx):
            d = test_set[i]
            inputs = image_processor(images=d["image"], return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            target_sizes = torch.Tensor([[d['height'], d['width']]])            
            results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
            nms_indexes = nms(scores=results["scores"], boxes=results["boxes"], iou_threshold=0.6)
            
            for index, score, label, box in zip(range(len(results["scores"])), results["scores"], results["labels"], results["boxes"]):
                if index in nms_indexes:
                    box = [round(i, 2) for i in box.tolist()]
                    x, y, x2, y2 = tuple(box)
                    rlt[d["image_name"]]["boxes"].append([x, y, x2, y2])
                    rlt[d["image_name"]]["labels"].append(label.item())
            print(d["image_name"])

with open(args.json_name, 'w') as f:
    json.dump(rlt, f)
