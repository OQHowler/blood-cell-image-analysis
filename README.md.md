# 🧬 Blood Cell Image Analysis Pipeline

## 🚀 Overview

This project implements a complete **end-to-end biomedical image
analysis pipeline** for classifying blood cells as *infected* or
*uninfected*.

It explores both: - 🔹 Classical Machine Learning (feature-based) - 🔹
Deep Learning (CNN-based)

------------------------------------------------------------------------

## 🧠 Key Highlights

-   Built full pipeline from raw images → prediction
-   Improved accuracy from **53% → 95%**
-   Compared **feature engineering vs representation learning**
-   Applied iterative experimentation and debugging

------------------------------------------------------------------------

## 📂 Project Structure

    project/
    │
    ├── classical_model.py
    ├── cnn_model.py
    ├── data/
    │   └── cell_images/
    │       ├── Parasitized/
    │       └── Uninfected/

------------------------------------------------------------------------

## ⚙️ Pipeline

### 1. Preprocessing

-   Resize to 128×128
-   Grayscale (classical pipeline)
-   Normalization

### 2. Segmentation

-   Otsu Thresholding for cell isolation

### 3. Feature Extraction

Extracted biologically relevant features: - Area - Perimeter - Mean
Intensity - Aspect Ratio - Compactness - Solidity

------------------------------------------------------------------------

## 🤖 Models

### 🔹 Classical ML

-   Model: Random Forest
-   Accuracy: **\~80%**

### 🔹 Deep Learning (CNN)

-   Input: RGB images
-   Architecture: 3 Conv layers + FC layers
-   Accuracy: **\~95%**

------------------------------------------------------------------------

## 📈 Performance Evolution

  Step      Change               Accuracy
  --------- -------------------- ----------
  Initial   Baseline             53%

  Fix 1     Otsu Segmentation    65%

  Fix 2     Added Features       68%
 
  Fix 3     Random Forest        78%
 
  Fix 3.2   Stratified Split     81%
 
  Fix 3.5   Feature Refinement   80%
 
  Fix 4     Initial CNN          64%
 
  Fix 5     RGB + Better CNN     72%
 
  Fix 6     More Epochs          93%
 
  Fix 7     Final CNN            **95%**

------------------------------------------------------------------------

## 🔍 Key Learnings

-   Feature quality matters more than model complexity (initially)
-   More features can **hurt performance**
-   Classical ML plateaus due to limited representation
-   CNNs outperform by learning **hierarchical features**

------------------------------------------------------------------------

## 🧠 Final Insight

> Classical methods depend on handcrafted features, while deep learning
> models automatically learn richer representations and achieve higher
> performance.

------------------------------------------------------------------------

## ⚡ How to Run

### Install dependencies

    pip install numpy opencv-python matplotlib scikit-learn torch

### Run Classical Model

    python classical_model.py

### Run CNN Model

    python cnn_model.py

------------------------------------------------------------------------

## 📌 Future Improvements

-   Data augmentation
-   Transfer learning (ResNet, EfficientNet)
-   Advanced segmentation (U-Net)

------------------------------------------------------------------------

## 🙌 Acknowledgment

Dataset: Malaria Cell Images Dataset (Kaggle)
