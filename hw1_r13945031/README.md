# Object Detection for Occupational Injury Prevention

## Project Description
This project implements an object detection model to identify and localize objects in images for occupational injury prevention. The model is trained on a dataset with 17 object categories and outputs bounding boxes and classifications for each detected object. The goal is to fine-tune a transformer-based model, DETR (DEtection TRansformers), and achieve high performance on the validation and test sets.

---

## Repository Structure
```
hw1_<student-id>/
│
├── hw1_<student-id>.pdf           # Report for the homework
├── valid_<student-id>.json        # Prediction file for the validation set
├── test_<student-id>.json         # Prediction file for the test set
├── codes/                         # Directory containing the codebase
│   ├── anno.py                    # Script to convert YOLO annotations to COCO format
│   ├── dataset.py                 # Custom dataset handling classes (train, infer, eval)
│   ├── train.py                   # Script for training the object detection model
│   ├── infer.py                   # Script for inference on the test images
│   ├── eval.py                    # Evaluation script for calculating mAP
│   ├── util.py                    # Utilities (data augmentation, formatting, etc.)
│   ├── BestModel.py               # Script to find the best checkpoint
└── README.md                      # Project instructions (this file)
```

---

## Installation Requirements
To run the code, install the following dependencies (my version):

- **Python**: 3.8+ (3.10.15)
- **PyTorch**: 1.12+ (2.4.1+cu124)
- **HuggingFace Transformers**: 4.10+ (4.44.2)
- **Torchmetrics**: 1.4.2
- **Albumentations**: 1.4.18
- **Pillow (PIL)**: 10.4.0

You can install the dependencies by running:
```bash
pip install torch torchvision transformers torchmetrics albumentations pillow
```

---

## Dataset Overview
The dataset consists of **17 object categories** and is divided into:

- **Training Set**: 4319 images
- **Validation Set**: 2160 images (do not use for training)
- **Test Set**: 1620 images

### Categories:
- Person
- Ear
- Earmuffs
- Face
- Face-guard
- Face-mask-medical
- Foot
- Tools
- Glasses
- Gloves
- Helmet
- Hands
- Head
- Medical-suit
- Shoes
- Safety-suit
- Safety-vest

---

## How to Run the Project

### 1. Preparing the Data
Before training, you need to convert the YOLO annotations to the COCO format using the following script:
```bash
python codes/anno.py
```
Make sure the images and labels are organized in the correct directories (`train/images`, `train/labels`, `valid/images`, etc.).

### 2. Training the Model
Run the following command to start the training process:
```bash
python codes/train.py
```
The model is fine-tuned using the DETR architecture with pre-trained weights. Training parameters include:
- **Batch size**: 8
- **Learning rate**: 1e-5
- **Epochs**: 100

Model checkpoints will be saved in the `runs` directory, and the best model will be determined based on evaluation loss.

### 3. Inference
To generate predictions on the test set, run the inference script:
```bash
python codes/infer.py --test <test_images_directory> --json_name test_<student-id>.json
```
The output will be saved in a JSON file (`test_<student-id>.json`), which contains the bounding boxes and labels for each image.

### 4. Evaluation
You can evaluate the model performance on the validation set using:
```bash
python eval.py <your_prediction.json> <valid_target.json>
```
This will compute the **mean Average Precision (mAP)** metric over IoU thresholds ranging from 0.50 to 0.95.

### 5. Best Checkpoint
To automatically select the best model checkpoint based on evaluation loss, run:
```bash
python codes/BestModel.py
```
This script will output the checkpoint directory with the lowest evaluation loss.

---

## Output Files
- `valid_<student-id>.json`: Predictions on the validation set.
- `test_<student-id>.json`: Predictions on the test set.
