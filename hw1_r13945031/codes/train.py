from transformers import (
    AutoModelForObjectDetection, 
    TrainingArguments, 
    Trainer, 
    EvalPrediction, 
    EarlyStoppingCallback
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from datasets import Dataset
from codes.util import *
from codes.dataset import MyDataset

train_path = Path("train/images/annotations.json")
valid_path = Path("valid/images/annotations.json")
train_set, valid_set = MyDataset(train_path), MyDataset(valid_path)
id2label, label2id = train_set.make_labelmaps()
train_set, valid_set = Dataset.from_list([i for i in train_set.tolist() if len(i["objects"]["bbox"])]), Dataset.from_list(valid_set.tolist())
train_set, valid_set = train_set.with_transform(transform_aug_ann), valid_set.with_transform(transform_aug_ann)

# finetune
#checkpoint_path = './runs/checkpoint-25000'  
#image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)

model = AutoModelForObjectDetection.from_pretrained(
    checkpoint_path,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
).to(device="cuda")

training_args = TrainingArguments(
    output_dir="./runs4",
    per_device_train_batch_size=8,
    num_train_epochs=100,
    eval_strategy="steps",         
    eval_steps=500,                
    logging_strategy="steps",      
    logging_steps=250,            
    logging_dir="./logs",         
    save_strategy="steps",         
    save_steps=500,                
    fp16=True,
    learning_rate=1e-5,
    weight_decay=1e-3,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    metric_for_best_model='loss',
    greater_is_better=False,                  
    load_best_model_at_end=True, 
    gradient_accumulation_steps=2,  
    max_grad_norm=1.0  
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_set,
    eval_dataset=valid_set,
    tokenizer=image_processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()


