# PathologyCVAE

# Convolutional Variational Autoencoder for Cancer Detection

## 📌 Overview
This repository contains the code, trained models, and analysis for our **Convolutional Variational Autoencoder (ConvVAE)** project, aimed at anomaly detection in **lung and colon cancer histopathological images**. Our goal is to leverage deep generative models to distinguish between **cancerous and non-cancerous** tissue samples, aiding early-stage cancer detection.

## 🚀 Project Motivation
Cancer diagnosis through histopathological images requires expert annotation, which is time-intensive and costly. Our **ConvVAE** model aims to **learn the underlying distribution of healthy tissue** and detect anomalies that may correspond to cancerous regions, providing an **unsupervised learning** approach to assist pathologists.

## 🏗️ Model Architecture
Our **Convolutional Variational Autoencoder (ConvVAE)** consists of three main components:

### ✨ Encoder
- **4 convolutional layers** (Conv2D) with BatchNorm and ReLU activation.
- Strided convolutions to reduce feature map dimensions.
- Fully connected layers to output **mean (`μ`)** and **log variance (`logσ²`)**.

### 🔗 Latent Space
- `μ` and `logσ²` are used for **reparameterization trick** to sample `z`.
- Fully connected layer maps `z` back to feature space.

### 🎨 Decoder
- **4 transposed convolutional layers** (ConvTranspose2D) with BatchNorm and ReLU activation.
- Final output layer with **Tanh() activation** for image reconstruction.

## 📊 Dataset
We use the **Lung and Colon Cancer Histopathological Images Dataset** from Kaggle. It contains:
- **15,000 images** of lung and colon tissue samples.
- Labeled as **cancerous vs. non-cancerous**.

🔗 Dataset Link: [Kaggle: Lung and Colon Cancer Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

## 🔍 Usage
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
Our experiments demonstrate that the **ConvVAE effectively reconstructs normal (non-cancerous) tissue**, while **cancerous samples exhibit higher reconstruction error**, making them identifiable as anomalies. Key observations:
- The model achieves **low reconstruction loss for normal samples**.
- Cancerous samples **show high reconstruction errors**, enabling detection.
- The learned latent space provides **meaningful representations** of tissue features.

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
👤 **Diptanshu Sikdar**  
📧 Email: dsikdar@uci.edu  

👤 **Travis Tran:**
📧 Email: travitt1@uci.edu

👤 **James Xu:** 
📧 Email: xujg@uci.edu

👤 **Jordan Yee:** 
📧 Email: jordady1@uci.edu

## 📌 Future Work
- Experiment with **different VAE architectures** (e.g., β-VAE, WAE).
- Use **self-supervised contrastive learning** to improve feature extraction.
- Fine-tune model using **GAN-based approaches** for enhanced reconstruction.

## ⭐ Acknowledgments
Special thanks to **UCI CS 175: Project in AI** for the opportunity to work on this project.
