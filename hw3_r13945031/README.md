# Text-to-Image Generation with GLIGEN

This project generates high-quality images based on textual prompts, bounding box annotations, and optional reference images. The workflow consists of generating prompts, refining them, and creating images using the GLIGEN pipeline, followed by evaluating the Fréchet Inception Distance (FID) for quality assessment.

---

## Workflow Overview

### Step 1: Generate Initial Prompts (`blip2.py`)
Use the BLIP-2 model to extract textual descriptions (`generated_text`) from raw images. These descriptions are saved in a JSON file and serve as the base for further refinement.

### Step 2: Refine Prompts (`prompts.py`)
Enhance the initial prompts with:
1. Object details (labels, counts).
2. Main subject descriptions (based on bounding box areas).
3. Custom suffixes for enhanced detail.

### Step 3: Generate Images with GLIGEN
Two methods are used:
1. **Without Reference Image (`gligen_wo_pic.py`)**: Generate images based on text prompts and bounding box constraints only.
2. **With Reference Image (`gligen_w_pic.py`)**: Include reference images as guidance during generation.

### Step 4: Evaluate Generated Images (`pytorch-fid`)
Calculate the **FID** score to evaluate the similarity between generated images and the real dataset.

---

## File Structure

```plaintext
project/
│
├── images/                           # Directory containing input images
├── label.json                        # Original dataset annotations
├── label_with_generated_text.json    # Output after running `blip2.py`
├── label_with_prompts.json           # Output after running `prompts.py`
├── generation/                       # Output folder for images generated without reference
├── generation_infer_pic/             # Output folder for images generated with reference
├── blip2.py                          # Script for generating initial prompts
├── prompts.py                        # Script for refining prompts
├── gligen_wo_pic.py                  # Script for generating images without reference
├── gligen_w_pic.py                   # Script for generating images with reference
└── README.md                         # Documentation
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your_username/your_project_name.git
cd your_project_name
```

### 2. Set Up the Environment
Create a virtual environment and install dependencies:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Step 1: Generate Initial Prompts
Run `blip2.py` to create initial prompts (`generated_text`) for each image.

```bash
python blip2.py
```

#### Output:
- Updates `label.json` with `generated_text`.
- Saves the updated file as `label_with_generated_text.json`.

---

### Step 2: Refine Prompts
Run `prompts.py` to enhance prompts with additional context.

```bash
python prompts.py
```

#### Output:
- Adds `prompt_w_label` and `prompt_w_suffix` to the dataset.
- Saves the updated file as `label_with_prompts.json`.

---

### Step 3: Generate Images

#### **Method 1: Without Reference Image**
Use `gligen_wo_pic.py` to generate images based on prompts and bounding boxes only.

```bash
python gligen_wo_pic.py
```

#### Output:
- Saves images in `generation/`.

---

#### **Method 2: With Reference Image**
Use `gligen_w_pic.py` to include reference images during generation.

```bash
python gligen_w_pic.py
```

#### Output:
- Saves images in `generation_infer_pic/`.

---

### Step 4: Evaluate FID Score
Use the `pytorch-fid` tool to calculate the FID score for generated images.

```bash
python -m pytorch_fid images generation_suffix --batch-size 1 --num-workers 2
```

#### Output:
- FID score is printed to the terminal, indicating the similarity between generated and real images.

---

## Key Scripts and Parameters

### `blip2.py`
- **Functionality**: Uses the BLIP-2 model to generate initial captions.
- **Input**: `label.json` and `images/`.
- **Output**: `label_with_generated_text.json`.

---

### `prompts.py`
- **Functionality**: Enhances prompts with:
  - Object counts.
  - Main subject descriptions.
  - Custom suffixes (e.g., "Highly detailed HD, in a construction site").
- **Input**: `label_with_generated_text.json`.
- **Output**: `label_with_prompts.json`.

---

### `gligen_wo_pic.py`
- **Functionality**: Generates images based on prompts and bounding box constraints.
- **Input**: `label_with_prompts.json`.
- **Output**: Images saved in `generation/`.

---

### `gligen_w_pic.py`
- **Functionality**: Includes reference images during generation.
- **Input**: `label_with_prompts.json` and images in `images/`.
- **Output**: Images saved in `generation_infer_pic/`.

---

## Evaluation Metrics

### Fréchet Inception Distance (FID)
- Measures similarity between the generated images and the real dataset.
- **Command**:
  ```bash
  python -m pytorch_fid <real_images_folder> <generated_images_folder>
  ```
- **Recommended Settings**:
  - `--batch-size`: Set to `1` for better memory handling.
  - `--num-workers`: Number of CPU workers for loading data.

---

## Notes

1. **Model Selection**:
   - Replace the BLIP-2 or GLIGEN models with other compatible models if needed.

2. **Customization**:
   - Update `custom_suffix` in `prompts.py` to modify the suffix added to prompts.

3. **CUDA Compatibility**:
   - Ensure PyTorch and Diffusers versions match your CUDA version.

