# Facial Diffusion Generation – Conditional Diffusion Model

This project implements a **Conditional Diffusion Model** capable of generating multi-ethnic facial images conditioned on class labels.  
Using a UNet-based architecture enriched with **self-attention**, **EMA**, **gradient checkpointing**, and **class embeddings**, the model synthesizes faces for three ethnic groups:

- **European**
- **Indian**
- **Oriental**

This work was completed as part of the *Advanced Intelligent Systems and Deep Learning* final project.

---

## 📄 Project Files

| File | Description |
|------|-------------|
| `Final_Report.pdf` | Full research report including methodology, design, results, and evaluation |
| `Term_Project.ipynb` | Jupyter Notebook containing the full implementation of the Conditional Diffusion Model |

---

## 🚀 Overview

Diffusion models are powerful generative models that gradually add noise to an image (forward process) and learn to reverse it (denoising process).  
This project extends the standard diffusion pipeline with **conditional generation**, allowing the model to produce images aligned with specific ethnic labels.

### Key Objectives
- Generate high-quality facial images conditioned on ethnicity  
- Use memory-efficient techniques to allow training on limited hardware  
- Evaluate model performance through visual and quantitative metrics  

---

## 🧠 Features

### **1. Conditional Diffusion Model**
- Forward and reverse diffusion processes  
- Predicts noise at each timestep  
- Embeds ethnicity labels for class-controlled generation  

### **2. UNet Architecture**
- Multi-resolution encoder/decoder  
- Skip connections for spatial detail preservation  
- Integrated **self-attention** to capture long-range dependencies  

### **3. Training Optimizations**
- **Gradient Checkpointing** – reduces GPU memory usage  
- **Exponential Moving Average (EMA)** – stabilizes training and improves image quality  
- **Cosine Learning Rate Scheduling**  
- **Gradient Clipping** for stable optimization  
- **Mixed Precision Training (AMP)**

### **4. Custom Dataset & Augmentation**
- Data sourced from public datasets (e.g., UTKFace)  
- Manual/automated labeling for ethnicity classes  
- Augmentations: flipping, rotations, jitter, cropping  

---

## 📁 Dataset

The dataset includes three classes:
- European
- Indian
- Oriental

⚠️ *Full dataset is not included in this repository due to size and licensing restrictions.*  
If you wish to run training, use:
- UTKFace (original source)  
- Your own cleaned dataset  
- Or provide a custom dataset following the same structure

---

## 🛠 Technologies Used

- **Python**
- **PyTorch**, Torchvision
- **CUDA**, Mixed Precision Training
- **Weights & Biases (wandb)**
- **Matplotlib / NumPy**
- **Jupyter Notebook**

---

## 🧪 Implementation Details

### Training Hyperparameters
| Parameter | Value |
|----------|-------|
| Learning Rate | 0.0002 |
| Batch Size | 10 |
| Image Size | 64×64 |
| Noise Steps | 500 |
| Epochs | 200 |

### Training Pipeline
1. Add noise to images (forward diffusion)  
2. Predict noise using UNet (reverse diffusion)  
3. Minimize MSE between predicted and actual noise  
4. Use conditional embeddings for ethnicity-specific outputs  
5. Apply EMA-averaged weights during sampling  

---

## 📊 Results & Visualization

The model was evaluated using:
- **MSE Loss** across epochs  
- **Qualitative inspection** of generated faces  
- **Attention maps** for interpretability  
- **Mixed-generation tests** (binary & tri-group blending)

Example outputs include:
- Progressive denoising images  
- Per-class generated samples  
- Mixed-ethnicity generated images  

Refer to the *Results and Visualization* section in the PDF report for detailed graphics.

---

## ▶️ How to Run

### Install dependencies:
```bash
pip install torch torchvision matplotlib numpy wandb


Open the notebook:
jupyter notebook Term_Project.ipynb


For training:

Ensure you have a CUDA-enabled GPU

Prepare dataset in the required directory format

Adjust hyperparameters as needed

