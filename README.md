# PathologyCVAE

# Anomaly Detection in Breast Histopathology Images with Convolutional Variational Autoencoders

## 📌 Overview
We explore the use of **Convolutional Variational Autoencoders (ConvVAE)** for anomaly detection in breast histopathological images. Our study compares ConvVAEs against a Fully Connected VAE (FC-VAE) and attention-based architectures for distinguishing cancerous and non-cancerous tissue samples. The implemented models include:

1. **Fully Connected Variational Autoencoder (VAE)** – lacks spatial awareness.
2. **Vanilla Convolutional VAE (ConvVAE)** – utilizes convolutional layers for feature extraction.
3. **VAE with a Pre-trained U-Net Encoder (CVAE-U-Net)** – leverages a ResNet-34 encoder for improved representation learning.
4. **Attention-enhanced ConvVAE (Attn-ConvVAE)** – integrates self-attention mechanisms to enhance feature learning.

Our findings indicate that ConvVAEs outperform simple VAEs, emphasizing the importance of convolutional layers for effective feature extraction. Among convolution-based models, ConvVAE achieves the highest classification accuracy and AUC, making it the best-performing model overall.

## 🚀 Project Motivation
Breast cancer is one of the most commonly diagnosed cancers worldwide. Early detection significantly improves survival rates, but traditional histopathological diagnosis is time-consuming and subjective. We investigate how **unsupervised deep learning models**, particularly Variational Autoencoders, can improve automated anomaly detection for breast histopathology images, reducing diagnostic variability and aiding pathologists.

## 📊 Dataset
We use the **Breast Histopathology Images** dataset from Kaggle, which includes:
- **277,524** image patches of size **50 × 50** extracted from **162 whole-mount slide images (WSI)**.
- Binary labels:
  - **198,738 IDC-negative (non-cancerous) samples**
  - **78,786 IDC-positive (cancerous) samples**

🔗 Dataset Link: [Kaggle: Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/data)



## 🔍 Demo

### Requirements

Before running the demo, ensure you have the following:

- **Python 3.9+** installed
- **Jupyter Notebook** installed (`pip install notebook`)
- **Dataset & Utility Files:** Extract the provided zip file. Make sure you see the following files in the same directory:
  - `BreastHistopathology_Small.zip` 
  - `DEMO_ConvVAE.ipynb`
  - `breast_cancer_dataset.py`
  - `environment.yml`
  - `instructions.md`

### Recommended System Setup

- **Linux** (preferred) with access to at least one **GPU** (NVIDIA recommended, CUDA supported)
- **Mac** or **Windows**: Works fine as long as you have **Python 3.9+** and Jupyter Notebook installed
- **Google Colab**: 
  - Manually upload the provided files (dataset, notebook, and utility scripts) to the workspace.
  - Change the runtime to **GPU (Google T4 recommended)**:  
    `Runtime` → `Change runtime type` → Select `GPU`
  - Run all cells in sequence.

### 1. Install Python & Jupyter Notebook
If you haven't already installed Python and Jupyter Notebook, do so using:

```bash
pip install notebook
```
or
```conda
conda install -c conda-forge notebook
```
### 2. Running the Demo

1. **Extract the ZIP file** and navigate to the extracted folder:
   ```bash
   unzip run_demo.zip -d run_demo
   cd run_demo
2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the provided .ipynb file, namely `DEMO_ConvVAE.ipynb`, in Jupyter Notebook.
4. Ensure the dataset and utility files are in the same directory as the notebook.
5. Run the first cell in the notebook to install all required dependencies.
6. Then run all remaining cells sequentially.



## 📝 Results
### Model Performance Summary

| Metric                | VAE     | ConvVAE  | Attn-ConvVAE | Frozen CVAE-U-Net | Unfrozen CVAE-U-Net |
|-----------------------|---------|----------|--------------|-------------------|---------------------|
| **Reconstruction Loss** | 1075.94 | 424.14 | **374.57**  | 639.11           | 540.59             |
| **KL Divergence**      | 48.71   | 261.09   | 392.12       | 20.01         | **12.35**          |
| **Accuracy (%)**       | 54.43   | **65.58** | 57.89        | 51.78            | 53.99              |
| **F1 Score**          | 0.41    | 0.64     | 0.63         | **0.65**          | 0.64               |
| **AUC**               | 0.53    | **0.70** | 0.60         | 0.50             | 0.53               |

**Key Findings:**
- **ConvVAE achieved the highest accuracy (65.58%) and AUC (0.70)**, making it the best overall model for anomaly detection.
- **Attn-ConvVAE had the lowest reconstruction loss (374.57)** but performed slightly worse in classification.
- **U-Net-based models had the lowest KL divergence**, suggesting effective latent space regularization but weaker classification performance.

## 📜 Report
For an in-depth discussion of our methodology, experiments, and findings, check out our **full report**:
📄 [Project Report (PDF)](https://github.com/dssikdar/PathologyCVAE/blob/main/CS175_Final_Project_Report.pdf)

## 📁 Repository Structure
```
📂 PathologyCVAE
 ├── 📂 demo
 ├── 📂 report
 ├── 📂 requirements
 ├── 📂 src
 ├── 📜 .gitignore
 ├── 📜 LICENSE
 └── 📜 README.md
```

## 🤝 Contributors
👤 **Diptanshu Sikdar:**  
📧 Email: dsikdar@uci.edu  

👤 **Travis Tran:**  
📧 Email: travitt1@uci.edu  

👤 **James Xu:**  
📧 Email: xujg@uci.edu  

👤 **Jordan Yee:**  
📧 Email: jordady1@uci.edu  

## 📌 Future Work
- **Explore higher-resolution datasets** to assess model generalization.
- **Enhance preprocessing** using denoising techniques (e.g., Non-Local Means, Wavelet-Based Denoising).
- **Improve feature extraction** by integrating multi-head attention mechanisms.

## ⭐ Acknowledgments
Special thanks to **UCI CS 175: Project in AI** for the opportunity to work on this project.
