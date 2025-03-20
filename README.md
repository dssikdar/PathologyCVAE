# PathologyCVAE

# Anomaly Detection in Breast Histopathology Images with Convolutional Variational Autoencoders

## 📌 Overview
We explore the use of **Convolutional Variational Autoencoders (ConvVAE)** for anomaly detection in breast histopathological images. Our study compares ConvVAEs against Fully Connected VAEs (FC-VAEs) and attention-based architectures for distinguishing cancerous and non-cancerous tissue samples. The implemented models include:

1. **Baseline Variational Autoencoder (VAE)** – lacks spatial awareness.
2. **Standard ConvVAE** – utilizes convolutional layers for feature extraction.
3. **VAE with a Pre-trained U-Net Encoder** – leverages a ResNet-34 encoder for improved representation learning.
4. **Attentive ConvVAE** – integrates attention mechanisms to enhance feature learning.

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
### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/cancer-anomaly-detection.git
cd cancer-anomaly-detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model
```bash
python train.py --epochs 50 --batch_size 64
```

### 4️⃣ Evaluate the Model
```bash
python evaluate.py --model_path saved_model.pth
```

### 5️⃣ Run Demo (Inference)
```bash
python demo.py --input sample_image.jpg
```

## 📝 Results
### Model Performance Summary
| Model | Reconstruction Loss | KL Divergence | Accuracy (%) | F1 Score | AUC |
|--------|----------------------|--------------|--------------|----------|------|
| VAE | 1075.94 | 48.71 | 54.43 | 0.41 | 0.53 |
| ConvVAE | **424.14** | 261.09 | **65.58** | 0.64 | **0.70** |
| Attn-ConvVAE | **374.57** | 392.12 | 57.89 | 0.63 | 0.60 |
| Frozen CVAE-U-Net | 639.11 | **20.01** | 51.78 | **0.65** | 0.50 |
| Unfrozen CVAE-U-Net | 540.59 | **12.35** | 53.99 | 0.64 | 0.53 |

**Key Findings:**
- **ConvVAE achieved the highest accuracy (65.58%) and AUC (0.70)**, making it the best overall model for anomaly detection.
- **Attn-ConvVAE had the lowest reconstruction loss (374.57)** but performed slightly worse in classification.
- **U-Net-based models had the lowest KL divergence**, suggesting effective latent space regularization but weaker classification performance.

## 📜 Report
For an in-depth discussion of our methodology, experiments, and findings, check out our **full report**:
📄 [Project Report (PDF)](link_to_report.pdf)

## 📁 Repository Structure
```
📂 cancer-anomaly-detection
 ├── 📜 README.md            # Project documentation
 ├── 📁 data                 # Data (not included due to size)
 ├── 📁 models               # Saved trained models
 ├── 📁 src                  # Source code
 │   ├── train.py            # Training script
 │   ├── evaluate.py         # Evaluation script
 │   ├── demo.py             # Inference/demo script
 │   ├── dataset.py          # Data loading and preprocessing
 │   ├── conv_vae.py         # Model architecture
 ├── 📁 notebooks            # Jupyter notebooks for analysis
 ├── requirements.txt        # Dependencies
 └── .gitignore              # Ignore unnecessary files
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
- **Apply transfer learning** from larger medical imaging datasets to refine model weights.

## ⭐ Acknowledgments
Special thanks to **UCI CS 175: Project in AI** for the opportunity to work on this project.
