import os
import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)
from scipy.spatial.distance import mahalanobis

from utils.autoencoder import autoencoder

# ==============================
# CONFIG
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = "completedata/UNSW_NB15_testing-set.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# LOAD SAVED ARTIFACTS
# ==============================
print("Loading saved models...")

preprocessor = joblib.load("saved_models/preprocessor.pkl")
stats = joblib.load("saved_models/mahalanobis_stats.pkl")
ensemble_clf = joblib.load("saved_models/ensemble_clf.pkl")
input_size = joblib.load("saved_models/input_size.pkl")

model = autoencoder(input_size).to(DEVICE)
model.load_state_dict(torch.load("saved_models/autoencoder.pth", map_location=DEVICE))
model.eval()

# ==============================
# LOAD DATA
# ==============================
print("Loading evaluation dataset...")

df = pd.read_csv(DATASET_PATH)

X = df.drop(['id', 'label', 'attack_cat'], axis=1)
y = (df['attack_cat'] != 'Normal').astype(int)

# ==============================
# FEATURE ENGINEERING
# ==============================
print("Applying preprocessing...")

skewed_cols = ['dur', 'sbytes', 'dbytes', 'sloss', 'dloss']
for col in skewed_cols:
    X[col] = np.log1p(X[col])

X_proc = preprocessor.transform(X)

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_scores(model, X_np):
    with torch.no_grad():
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)

        en_out, de_out, latent, recon = model(X_tensor)

        recon_error = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()

        return (
            X_np,
            [x.cpu().numpy() for x in en_out],
            [x.cpu().numpy() for x in de_out],
            recon_error
        )

print("Extracting deep anomaly features...")

inp, en, de, recon_err = extract_scores(model, X_proc)

# ==============================
# BUILD ENSEMBLE FEATURES
# ==============================
ensemble_features = []

mean, inv_cov = stats['input']
ensemble_features.append([
    mahalanobis(x, mean, inv_cov)
    for x in inp
])

for i, feat in enumerate(en):
    mean, inv_cov = stats[f'en_{i}']
    ensemble_features.append([
        mahalanobis(x, mean, inv_cov)
        for x in feat
    ])

for i, feat in enumerate(de):
    mean, inv_cov = stats[f'de_{i}']
    ensemble_features.append([
        mahalanobis(x, mean, inv_cov)
        for x in feat
    ])

ensemble_features.append(recon_err)

ensemble_X = np.array(ensemble_features).T

# ==============================
# PREDICT
# ==============================
print("Generating predictions...")

preds = ensemble_clf.predict(ensemble_X)
probs = ensemble_clf.predict_proba(ensemble_X)[:, 1]

# ==============================
# METRICS
# ==============================
print("Computing metrics...")

acc = accuracy_score(y, preds)
prec = precision_score(y, preds)
rec = recall_score(y, preds)
f1 = f1_score(y, preds)
roc_auc = roc_auc_score(y, probs)

precision_curve, recall_curve, _ = precision_recall_curve(y, probs)
pr_auc = auc(recall_curve, precision_curve)

cm = confusion_matrix(y, preds)

# ==============================
# SAVE METRICS
# ==============================
with open(f"{RESULTS_DIR}/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"PR-AUC: {pr_auc:.4f}\n")

report_df = pd.DataFrame(
    classification_report(y, preds, output_dict=True)
).transpose()

report_df.to_csv(f"{RESULTS_DIR}/classification_report.csv")

# ==============================
# CONFUSION MATRIX
# ==============================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
plt.close()

# ==============================
# ROC CURVE
# ==============================
fpr, tpr, _ = roc_curve(y, probs)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(f"{RESULTS_DIR}/roc_curve.png")
plt.close()

# ==============================
# PR CURVE
# ==============================
plt.figure(figsize=(6,5))
plt.plot(recall_curve, precision_curve)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig(f"{RESULTS_DIR}/pr_curve.png")
plt.close()

# ==============================
# SCORE DISTRIBUTION
# ==============================
plt.figure(figsize=(8,5))
plt.hist(probs[y==0], bins=50, alpha=0.6, label="Normal")
plt.hist(probs[y==1], bins=50, alpha=0.6, label="Attack")
plt.legend()
plt.title("Predicted Score Distribution")
plt.savefig(f"{RESULTS_DIR}/score_distribution.png")
plt.close()

print("\n🚀 Evaluation Complete — Results Saved in /results")