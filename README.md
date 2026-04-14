# Zero-Day Attack Detection using Hybrid Deep Learning Ensemble (UNSW-NB15)

## Overview

This project implements a **Zero-Day Network Intrusion Detection System (NIDS)** for detecting previously unseen cyberattacks using a hybrid deep learning + statistical anomaly detection approach.

The model is trained on the **UNSW-NB15** network traffic dataset and combines:

- **Autoencoder Neural Network** for learning normal traffic patterns
- **Mahalanobis Distance Analysis** for statistical deviation measurement
- **Reconstruction Error Scoring** for anomaly quantification
- **Logistic Regression Ensemble** for final anomaly classification

This architecture is designed specifically for **zero-day attack detection**, where attack signatures are unknown during training.

---

## Problem Statement

Traditional signature-based IDS systems fail to detect **zero-day attacks** because:

- They rely on known attack signatures
- Cannot generalize to unseen threats
- Require constant rule/database updates

This project addresses the problem using **unsupervised / anomaly-based detection**, where the model learns **normal traffic behavior** and flags deviations as suspicious.

---

## Dataset Used

### UNSW-NB15 Dataset

The **UNSW-NB15** dataset is a modern benchmark dataset for network intrusion detection developed by the Australian Centre for Cyber Security.

It contains:

- Realistic modern network traffic
- Normal and malicious packets
- Multiple attack categories
- 49 network flow features

### Attack Categories Included

- Fuzzers  
- Analysis  
- Backdoor  
- DoS  
- Exploits  
- Generic  
- Reconnaissance  
- Shellcode  
- Worms  

---

## Model Architecture

```text
Input Features
   ↓
Preprocessing + Encoding
   ↓
Autoencoder
   ↓
Latent / Reconstruction Features
   ↓
Mahalanobis Distance Computation
   ↓
Reconstruction Error
   ↓
Logistic Regression Ensemble
   ↓
Final Anomaly Score
```

---

## Performance

| Metric | Value |
|--------|------|
| Accuracy | 79.81% |
| Precision | 94.96% |
| Recall | 74.28% |
| F1 Score | 83.36% |
| ROC-AUC | 0.883 |
| PR-AUC | 0.944 |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/hrishikesh-bhatia/MLSEMPROJECT
cd MLSEMPROJECT
```

---

### Create Virtual Environment

```bash
python -m venv venv
venv/Scripts/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Required Libraries

Main dependencies:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- torch
- scipy
- joblib

---

## Dataset Setup

Due to large file size, dataset is not included in repository.

Download UNSW-NB15 dataset and place it inside:

```text
completedata/
```

Required files:

```text
completedata/
├── UNSW_NB15_training-set.csv
└── UNSW_NB15_testing-set.csv
```

---

## Usage

### Train Model

```bash
python train_and_save.py
```

This will:

- Train autoencoder
- Compute Mahalanobis statistics
- Train ensemble classifier
- Save trained models in `saved_models/`

---

### Evaluate Model

```bash
python evaluate.py
```

This will:

- Load saved models
- Evaluate on testing set
- Generate performance metrics
- Save graphs/plots in `results/`

---

## Output Artifacts

### Saved Models

```text
saved_models/
├── autoencoder.pth
├── preprocessor.pkl
├── mahalanobis_stats.pkl
├── ensemble_clf.pkl
├── input_size.pkl
```

---

### Evaluation Results

```text
results/
├── metrics.txt
├── classification_report.csv
├── confusion_matrix.png
├── roc_curve.png
├── pr_curve.png
├── score_distribution.png
├── loss_curve.png
```

---

## Key Features

- Detects **Zero-Day / Unknown Attacks**
- Unsupervised Normal-Behavior Learning
- Deep Feature Extraction using Autoencoder
- Statistical Outlier Measurement via Mahalanobis Distance
- Ensemble Decision Layer for Robust Classification

---

## Future Improvements

- Real-time Packet Stream Detection
- Threshold Optimization / Dynamic Calibration
- Transformer-Based Sequence Modeling
- Deployment in Live IDS Environment

---

## Author

**Hrishikesh Bhatia**

---

## License

This project is for academic / educational purposes.
