import os
import json
import torch
from transformers import AutoModelForObjectDetection
from codes.util import *
# 設定運行目錄
runs_dir = './runs4'
# 獲取所有 checkpoint 資料夾
checkpoints = [d for d in os.listdir(runs_dir) if d.startswith("checkpoint")]

best_checkpoint = None
best_eval_loss = float('inf')  

for checkpoint in checkpoints:
    checkpoint_path = os.path.join(runs_dir, checkpoint, 'trainer_state.json')
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            trainer_state = json.load(f)
            
            for record in trainer_state['log_history']:
                if 'eval_loss' in record:
                    eval_loss = record['eval_loss']
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_checkpoint = checkpoint

print(f"best model- {best_checkpoint}，eval_loss: {best_eval_loss}")


