# X-ray Image Denoising with Diffusion Probabilistic Models

This project explores the use of **Diffusion Probabilistic Models (DPMs)** for **X-ray image denoising**, simulating realistic low-dose noise and reconstructing high-quality medical images using a U-Net–based architecture built in **PyTorch**.

---

## 🧠 Overview

Medical imaging often involves trade-offs between **image quality** and **radiation dose**.  
This project aims to **reduce noise in low-dose X-ray images** by leveraging diffusion models — a class of generative models that learn to reverse a noise process applied to data.

By training a model to predict and remove added noise step-by-step, we can reconstruct clean images that preserve critical diagnostic details while reducing patient exposure.

---

## ⚙️ Features

- **Forward Diffusion Process:** Adds Gaussian noise progressively to clean X-ray data using a cosine-based β (beta) schedule.  
- **Reverse Diffusion Process:** Iteratively denoises random noise to generate clean X-ray images.  
- **U-Net Denoiser:** Custom-built architecture for noise prediction and image reconstruction.  
- **Cosine Noise Scheduling:** Smooth variance control across timesteps for stable training.  
- **Quantitative Evaluation:** Measures reconstruction performance using **MSE**, **PSNR**, and **SSIM** metrics.  

---

## 🧩 Model Architecture

- **Framework:** PyTorch  
- **Backbone:** U-Net (encoder-decoder with skip connections)  
- **Loss Function:** Mean Squared Error (MSE)  
- **Noise Type:** Gaussian  
- **Noise Schedule:** Cosine β schedule  

---

## 📊 Results

| Metric | Value |
|:-------|:------|
| Mean Squared Error (MSE) | < 0.02 |
| Peak Signal-to-Noise Ratio (PSNR) | ~32 dB |
| Structural Similarity Index (SSIM) | ~0.92 |

Example outputs show significant reduction in structured noise and improved contrast in low-dose X-ray images.

---

## 🧪 Experiments

- Varied β-schedule ranges to observe stability and convergence.  
- Adjusted U-Net depth and channel sizes for optimal reconstruction accuracy.  
- Visualized diffusion steps to interpret the denoising process at different timesteps.  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** PyTorch, NumPy, Matplotlib, tqdm  
- **Environment:** Jupyter / Kaggle Notebook  

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/xray-diffusion-denoising.git
cd xray-diffusion-denoising
