import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

from utils.autoencoder import autoencoder

# ==============================
# CONFIG
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 512
LR = 0.001

# ==============================
# LOAD DATA
# ==============================
print("Loading dataset...")
df = pd.read_csv("completedata/UNSW_NB15_training-set.csv")

X = df.drop(['id', 'label', 'attack_cat'], axis=1)
y = (df['attack_cat'] != 'Normal').astype(int)

# ==============================
# FEATURE ENGINEERING
# ==============================
print("Feature engineering...")

skewed_cols = ['dur', 'sbytes', 'dbytes', 'sloss', 'dloss']
for col in skewed_cols:
    X[col] = np.log1p(X[col])

cat_cols = ['proto', 'service', 'state']
num_cols = [c for c in X.columns if c not in cat_cols]

# ==============================
# TRAIN / VALIDATION SPLIT
# ==============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ONLY NORMAL FOR AE TRAINING
X_train_normal = X_train[y_train == 0]

# ==============================
# PREPROCESSING
# ==============================
print("Preprocessing...")

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

X_train_normal_proc = preprocessor.fit_transform(X_train_normal)
X_val_proc = preprocessor.transform(X_val)

input_size = X_train_normal_proc.shape[1]

# ==============================
# TRAIN AUTOENCODER
# ==============================
print("Training Autoencoder...")

model = autoencoder(input_size).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_tensor = torch.tensor(X_train_normal_proc, dtype=torch.float32).to(DEVICE)

for epoch in range(EPOCHS):
    model.train()

    permutation = torch.randperm(train_tensor.size(0))

    epoch_loss = 0

    for i in range(0, train_tensor.size(0), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch = train_tensor[indices]

        optimizer.zero_grad()

        _, _, _, recon = model(batch)

        loss = criterion(recon, batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f}")

# ==============================
# FEATURE EXTRACTION FUNCTION
# ==============================
def extract_scores(model, X_np):
    model.eval()

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

# ==============================
# TRAIN NORMAL FEATURE STATS
# ==============================
print("Computing Mahalanobis stats...")

train_input, train_en, train_de, _ = extract_scores(model, X_train_normal_proc)

def compute_stats(features):
    mean = np.mean(features, axis=0)
    cov = np.cov(features.T) + np.eye(features.shape[1]) * 1e-6
    inv_cov = inv(cov)
    return mean, inv_cov

stats = {}

stats['input'] = compute_stats(train_input)

for i, feat in enumerate(train_en):
    stats[f'en_{i}'] = compute_stats(feat)

for i, feat in enumerate(train_de):
    stats[f'de_{i}'] = compute_stats(feat)

# ==============================
# BUILD VALIDATION ENSEMBLE FEATURES
# ==============================
print("Building ensemble features...")

val_input, val_en, val_de, val_recon = extract_scores(model, X_val_proc)

ensemble_features = []

# Input Mahalanobis
mean, inv_cov = stats['input']
ensemble_features.append([
    mahalanobis(x, mean, inv_cov)
    for x in val_input
])

# Encoder Mahalanobis
for i, feat in enumerate(val_en):
    mean, inv_cov = stats[f'en_{i}']
    ensemble_features.append([
        mahalanobis(x, mean, inv_cov)
        for x in feat
    ])

# Decoder Mahalanobis
for i, feat in enumerate(val_de):
    mean, inv_cov = stats[f'de_{i}']
    ensemble_features.append([
        mahalanobis(x, mean, inv_cov)
        for x in feat
    ])

# Reconstruction Error
ensemble_features.append(val_recon)

ensemble_X = np.array(ensemble_features).T

# ==============================
# TRAIN ENSEMBLE CLASSIFIER
# ==============================
print("Training Logistic Regression Ensemble...")

ensemble_clf = LogisticRegression(max_iter=1000)
ensemble_clf.fit(ensemble_X, y_val)

# ==============================
# EVALUATE
# ==============================
preds = ensemble_clf.predict(ensemble_X)
probs = ensemble_clf.predict_proba(ensemble_X)[:, 1]

print("\n--- Validation Results ---")
print(classification_report(y_val, preds))
print("ROC-AUC:", roc_auc_score(y_val, probs))

# ==============================
# SAVE EVERYTHING
# ==============================
print("Saving models...")

torch.save(model.state_dict(), "saved_models/autoencoder.pth")
joblib.dump(preprocessor, "saved_models/preprocessor.pkl")
joblib.dump(stats, "saved_models/mahalanobis_stats.pkl")
joblib.dump(ensemble_clf, "saved_models/ensemble_clf.pkl")
joblib.dump(input_size, "saved_models/input_size.pkl")
print("🚀 DONE — Paper-based Zero-Day Detector Ready!")