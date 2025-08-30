import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, log_loss
)
from xgboost import XGBClassifier

# === SETTINGS ===
feature_dir = r"D:/dl/MIDOG_2025_Track_2/HIBOU_L_and_Barlow_Concatenated_Features/"
csv_path = r"D:\dl\MIDOG_2025_Track_2\MIDOG25_Atypical_Classification_Train_Set.csv"

# === LOAD CSV ===
df = pd.read_csv(csv_path)
df["feature_path"] = df["image_id"].apply(lambda x: os.path.join(feature_dir, x.replace(".png", "_hibou_bt.pt")))
df = df[df["feature_path"].apply(os.path.exists)]  # Only keep valid paths

# === LOAD FEATURES ===
X = torch.stack([torch.load(p).squeeze(0) for p in df["feature_path"]]).numpy()
y = df["majority"].map({"NMF": 1, "AMF": 0}).values

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Reduce to 1D feature using LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)

import joblib
lda_save_path = r"D:/dl/MIDOG_2025_Track_2/models/v6/v6_lda_transformer.pkl"
# === Save transformer ===
joblib.dump(lda, lda_save_path)
print(f"LDA transformer saved to: {lda_save_path}")

# Then split train/test on LDA features
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X_lda, y, df.index, test_size=0.2, stratify=y, random_state=42
)

# === DEFINE CATBOOST (FOCAL LOSS) AND GRID SEARCH ===
from catboost import CatBoostClassifier

# You can tune focal loss’ gamma/alpha by enumerating different strings
param_grid = {
    "loss_function": [
        "Focal:focal_gamma=1;focal_alpha=0.25",
        "Focal:focal_gamma=2;focal_alpha=0.25",
        "Focal:focal_gamma=2;focal_alpha=0.5",
    ],
    "iterations": [300, 600, 1000],
    "depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "l2_leaf_reg": [1, 3, 5],
    "subsample": [0.8, 1.0],        # row subsampling
    "rsm": [0.8, 1.0],              # column subsampling (like colsample_bytree)
    # Optionally handle slight class skew in addition to focal loss:
    # "scale_pos_weight": [1.0, 1.5, 2.0],
}

cb = CatBoostClassifier(
    random_seed=42,
    verbose=0,              # silence per-iter logs during GridSearchCV
    eval_metric="Logloss"   # keep a standard eval metric; training uses Focal
)

clf = GridSearchCV(
    estimator=cb,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="balanced_accuracy",
    verbose=1,
    n_jobs=-1
)

clf.fit(X_train, y_train)

print("\nBest Parameters:", clf.best_params_)
best_model = clf.best_estimator_


# === EVALUATION ON TEST SET ===
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_f1 = f1_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_specificity = tn / (tn + fp)
test_npv = tn / (tn + fn)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_balanced_accuracy = 0.5 * (test_recall + test_specificity)
test_auc = roc_auc_score(y_test, y_test_proba)
test_loss = log_loss(y_test, y_test_proba)

print("\nTest Set Metrics:")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")

print(f"F1-Score: {test_f1:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"Specificity: {test_specificity:.4f}")
print(f"Negative Predictive Value: {test_npv:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Balanced Accuracy: {test_balanced_accuracy:.4f}")
print(f"AUC: {test_auc:.4f}")
print("Test Log Loss:", test_loss)


import joblib
# Refit the best model to the entire dataset
best_model.fit(X_lda, y)
joblib.dump(best_model, "models/v6/v6_classifier.pkl")
print("Best model saved as 'best_xgboost_model.pkl'")

best_model_ = clf.best_estimator_
# Cross-validation predictions for each fold
cv = StratifiedKFold(n_splits=5)
metrics = {
    "f1": [],
    "precision": [],
    "recall": [],
    "specificity": [],
    "npv": [],
    "accuracy": [],
    "balanced_accuracy": [],
    "auc": [],
    "log_loss": []
}

def calculate_specificity_npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return specificity, npv

for train_idx, val_idx in cv.split(X_train, y_train):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    best_model_.fit(X_fold_train, y_fold_train)
    y_pred = best_model_.predict(X_fold_val)
    y_proba = best_model_.predict_proba(X_fold_val)[:, 1]

    metrics["f1"].append(f1_score(y_fold_val, y_pred))
    metrics["precision"].append(precision_score(y_fold_val, y_pred))
    metrics["recall"].append(recall_score(y_fold_val, y_pred))
    specificity, npv = calculate_specificity_npv(y_fold_val, y_pred)
    metrics["specificity"].append(specificity)
    metrics["npv"].append(npv)
    metrics["accuracy"].append(accuracy_score(y_fold_val, y_pred))
    metrics["balanced_accuracy"].append(0.5 * (recall_score(y_fold_val, y_pred) + specificity))
    metrics["auc"].append(roc_auc_score(y_fold_val, y_proba))
    metrics["log_loss"].append(log_loss(y_fold_val, y_proba))

# Calculate average metrics
average_metrics = {metric: np.mean(scores) for metric, scores in metrics.items()}

print("Cross-Validation Metrics (Averaged over 5 folds):")
for metric, score in average_metrics.items():
    print(f"{metric.capitalize()}: {score:.4f}")

print("Best Params:", clf.best_params_)
print("Best Cross-Validation Balanced Accuracy:", clf.best_score_)





from collections import defaultdict

def evaluate_per_domain(df_full, X_all, y_all, y_pred_all, y_proba_all, domain_column):
    results = defaultdict(dict)
    df_full = df_full.reset_index(drop=True)

    print(f"\n===== Per-Domain Metrics by: {domain_column} =====")
    for domain in df_full[domain_column].unique():
        indices = df_full[df_full[domain_column] == domain].index

        y_true_d = y_all[indices]
        y_pred_d = y_pred_all[indices]
        y_proba_d = y_proba_all[indices]

        if len(np.unique(y_true_d)) < 2:
            print(f"\nDomain '{domain}' skipped (only one class present)")
            continue

        tn, fp, fn, tp = confusion_matrix(y_true_d, y_pred_d).ravel()
        f1 = f1_score(y_true_d, y_pred_d)
        precision = precision_score(y_true_d, y_pred_d)
        recall = recall_score(y_true_d, y_pred_d)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = accuracy_score(y_true_d, y_pred_d)
        balanced_acc = 0.5 * (recall + specificity)
        auc = roc_auc_score(y_true_d, y_proba_d)
        loss = log_loss(y_true_d, y_proba_d)

        print(f"\nDomain: {domain}")
        print(f"  N: {len(indices)}")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"  F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  Specificity: {specificity:.4f}, NPV: {npv:.4f}, Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}, AUC: {auc:.4f}")
        print(f"  Log Loss: {loss:.4f}")

        results[domain] = {
            "n_samples": len(indices),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "npv": npv,
            "accuracy": accuracy,
            "balanced_acc": balanced_acc,
            "auc": auc,
            "log_loss": loss
        }

    path = r"D:\dl\MIDOG_2025_Track_2\models\v6"
    filename = f"{domain_column}_metrics.csv"
    full_path = os.path.join(path, filename)

    pd.DataFrame.from_dict(results, orient="index").to_csv(full_path)

    return results


df_test = df.loc[test_idx].copy()
_ = evaluate_per_domain(df_test, X_test, y_test, y_test_pred, y_test_proba, domain_column="Tumor")
_ = evaluate_per_domain(df_test, X_test, y_test, y_test_pred, y_test_proba, domain_column="Scanner")
_ = evaluate_per_domain(df_test, X_test, y_test, y_test_pred, y_test_proba, domain_column="Origin")
_ = evaluate_per_domain(df_test, X_test, y_test, y_test_pred, y_test_proba, domain_column="Species")
