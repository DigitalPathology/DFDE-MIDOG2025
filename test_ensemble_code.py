import os
import json
import joblib
import numpy as np
import pandas as pd
import torch

from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, log_loss, confusion_matrix
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# =========================
# ===== USER SETTINGS =====
# =========================
feature_dir = r"D:/dl/MIDOG_2025_Track_2/HIBOU_L_and_Barlow_Concatenated_Features/"
csv_path    = r"D:\dl\MIDOG_2025_Track_2\MIDOG25_Atypical_Classification_Train_Set.csv"

# LDA transformers saved during training (set to None if not used)
lda_xgb_path = r"D:/dl/MIDOG_2025_Track_2/models/v5/v5_lda_transformer.pkl"
lda_cb_path  = r"D:/dl/MIDOG_2025_Track_2/models/v6/v6_lda_transformer.pkl"

# Model files
xgb_paths = [
    r"D:\dl\MIDOG_2025_Track_2\models\v5\v5_classifier.pkl",
]
cb_paths = [
    r"D:\dl\MIDOG_2025_Track_2\models\v6\v6_classifier.pkl",
]

# Ensemble weights (will be normalized)
W_XGB = 0.493
W_CB  = 0.507

# Output
OUT_CSV = r"D:\dl\MIDOG_2025_Track_2\ensemble_test_predictions.csv"
THRESH  = 0.5
RANDOM_STATE = 42


# =========================
# ======  HELPERS    ======
# =========================
def find_first_exists(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None

def load_catboost(path: str):
    if path.lower().endswith(".cbm"):
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(path)
        return model
    return joblib.load(path)

def load_xgb(path: str):
    return joblib.load(path)

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def npv_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0

def evaluate_all(y_true, y_proba, thr=0.5) -> dict:
    y_pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Specificity": specificity_score(y_true, y_pred),
        "Negative Predictive Value": npv_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": 0.5 * (recall_score(y_true, y_pred) + specificity_score(y_true, y_pred)),
        "AUC": roc_auc_score(y_true, y_proba),
        "Log Loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "Threshold": thr
    }


# =========================
# ========  MAIN  =========
# =========================
if __name__ == "__main__":
    # 1) LOAD CSV + feature paths
    df = pd.read_csv(csv_path)
    df["feature_path"] = df["image_id"].apply(
        lambda x: os.path.join(feature_dir, x.replace(".png", "_hibou_bt.pt"))
    )
    df = df[df["feature_path"].apply(os.path.exists)].reset_index(drop=True)

    # Labels: NMF->1, AMF->0
    y = df["majority"].map({"NMF": 1, "AMF": 0}).values

    # 2) LOAD RAW FEATURES (order = df order)
    feats = []
    for p in df["feature_path"]:
        t = torch.load(p)
        if t.dim() > 1:  # expect (1, D)
            t = t.squeeze(0)
        feats.append(t.numpy())
    X_raw = np.stack(feats, axis=0)

    # 3) RECREATE EXACT TRAIN/TEST INDICES
    #    To get the *same* test set independent of feature space,
    #    split on indices with the same y/stratify/random_state:
    idx_all = np.arange(len(df))
    idx_train, idx_test, y_train_tmp, y_test = train_test_split(
        idx_all, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    test_ids = df.loc[idx_test, "image_id"].to_list()

    # 4) PREP FEATURE VIEWS FOR EACH MODEL
    # XGBoost view
    if lda_xgb_path and os.path.isfile(lda_xgb_path):
        lda_xgb: LinearDiscriminantAnalysis = joblib.load(lda_xgb_path)
        X_xgb = lda_xgb.transform(X_raw)
        print(f"[INFO] Loaded XGB LDA: {lda_xgb_path} | X_xgb shape: {X_xgb.shape}")
    else:
        raise FileNotFoundError(
            f"LDA for XGBoost not found at {lda_xgb_path}. "
            f"Please set 'lda_xgb_path' to the LDA saved during XGB training."
        )

    # CatBoost view
    if lda_cb_path:
        if os.path.isfile(lda_cb_path):
            lda_cb: LinearDiscriminantAnalysis = joblib.load(lda_cb_path)
            X_cb = lda_cb.transform(X_raw)
            print(f"[INFO] Loaded CatBoost LDA: {lda_cb_path} | X_cb shape: {X_cb.shape}")
        else:
            raise FileNotFoundError(
                f"LDA for CatBoost not found at {lda_cb_path}. "
                f"Either provide the correct path or set 'lda_cb_path = None' if CatBoost used raw features."
            )
    else:
        X_cb = X_raw
        print(f"[INFO] CatBoost uses RAW features | X_cb shape: {X_cb.shape}")

    # Slice TEST partitions
    X_xgb_test = X_xgb[idx_test]
    X_cb_test  = X_cb[idx_test]

    # 5) LOAD MODELS
    xgb_path = find_first_exists(xgb_paths)
    cb_path  = find_first_exists(cb_paths)
    if xgb_path is None:
        raise FileNotFoundError(f"Could not find XGBoost model in: {xgb_paths}")
    if cb_path is None:
        raise FileNotFoundError(f"Could not find CatBoost model in: {cb_paths}")

    print(f"[INFO] Loading XGBoost: {xgb_path}")
    xgb_model = load_xgb(xgb_path)
    print(f"[INFO] Loading CatBoost: {cb_path}")
    cb_model  = load_catboost(cb_path)

    # 6) PREDICT PROBABILITIES
    p_xgb = xgb_model.predict_proba(X_xgb_test)[:, 1]
    p_cb  = cb_model.predict_proba(X_cb_test)[:, 1]

    # 7) SOFT-VOTING ENSEMBLE
    w_sum = float(W_XGB + W_CB)
    wx, wc = W_XGB / w_sum, W_CB / w_sum
    p_ens = wx * p_xgb + wc * p_cb

    # 8) METRICS
    metrics_xgb = evaluate_all(y_test, p_xgb, thr=THRESH)
    metrics_cb  = evaluate_all(y_test, p_cb,  thr=THRESH)
    metrics_ens = evaluate_all(y_test, p_ens, thr=0.707)

    print("\n=== Weights ===")
    print(f"W_XGB={wx:.4f}, W_CB={wc:.4f}")

    print("\n=== XGBoost (TEST) ===")
    print(json.dumps(metrics_xgb, indent=2))

    print("\n=== CatBoost (TEST) ===")
    print(json.dumps(metrics_cb, indent=2))

    print("\n=== Ensemble (Soft Vote, TEST) ===")
    print(json.dumps(metrics_ens, indent=2))

    # 9) SAVE PER-SLIDE OUTPUT
    out_df = pd.DataFrame({
        "image_id": test_ids,
        "y_true": y_test,
        "p_xgb": p_xgb,
        "p_cb": p_cb,
        "p_ensemble": p_ens,
        "y_pred_ens": (p_ens >= 0.707).astype(int)
    })
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Saved predictions to: {OUT_CSV}")

    # ==== THRESHOLD SWEEP & PLOTS (add below your current save) ====
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        precision_recall_curve, roc_curve, auc, average_precision_score,
        confusion_matrix, f1_score
    )


    # Helper metrics at a threshold
    def _binary_metrics(y_true, y_proba, thr):
        y_pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0  # sensitivity
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn)
        bal = 0.5 * (rec + spec)
        f1 = f1_score(y_true, y_pred)
        return prec, rec, spec, npv, acc, bal, f1


    # 1) Threshold sweep
    thr_grid = np.linspace(0.0, 1.0, 1001)  # fine grid
    rows = []
    for thr in thr_grid:
        prec, rec, spec, npv, acc, bal, f1 = _binary_metrics(y_test, p_ens, thr)
        rows.append({
            "threshold": thr, "precision": prec, "recall": rec, "specificity": spec,
            "npv": npv, "accuracy": acc, "balanced_accuracy": bal, "f1": f1,
            "youdenJ": rec + spec - 1.0
        })
    sweep_df = pd.DataFrame(rows)
    sweep_csv = os.path.splitext(OUT_CSV)[0].replace(".csv", "") + "_threshold_sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"[OK] Saved threshold sweep to: {sweep_csv}")

    # 2) Pick best thresholds
    best_bal = sweep_df.iloc[sweep_df["balanced_accuracy"].idxmax()]
    best_j = sweep_df.iloc[sweep_df["youdenJ"].idxmax()]
    best_f1 = sweep_df.iloc[sweep_df["f1"].idxmax()]

    print("\n=== Suggested thresholds ===")
    print(f"Max Balanced Acc: thr={best_bal.threshold:.3f} | bal_acc={best_bal.balanced_accuracy:.4f} | "
          f"recall={best_bal.recall:.4f} | spec={best_bal.specificity:.4f} | F1={best_bal.f1:.4f}")
    print(f"Max Youden's J : thr={best_j.threshold:.3f} | J={best_j.youdenJ:.4f} | "
          f"recall={best_j.recall:.4f} | spec={best_j.specificity:.4f} | F1={best_j.f1:.4f}")
    print(f"Max F1         : thr={best_f1.threshold:.3f} | F1={best_f1.f1:.4f} | "
          f"recall={best_f1.recall:.4f} | spec={best_f1.specificity:.4f} | bal_acc={best_f1.balanced_accuracy:.4f}")

    # 3) PR curve (precision vs recall)
    precisions, recalls, _ = precision_recall_curve(y_test, p_ens)
    ap = average_precision_score(y_test, p_ens)

    plt.figure()
    plt.plot(recalls, precisions, label=f"Ensemble PR (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Ensemble)")
    plt.legend()
    pr_path = os.path.splitext(OUT_CSV)[0].replace(".csv", "") + "_PR.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved PR curve to: {pr_path}")

    # 4) ROC curve (TPR vs FPR)
    fpr, tpr, _ = roc_curve(y_test, p_ens)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"Ensemble ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve (Ensemble)")
    plt.legend()
    roc_path = os.path.splitext(OUT_CSV)[0].replace(".csv", "") + "_ROC.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved ROC curve to: {roc_path}")

    # 5) Balanced accuracy vs threshold
    plt.figure()
    plt.plot(sweep_df["threshold"], sweep_df["balanced_accuracy"], label="Balanced Accuracy")
    plt.axvline(best_bal.threshold, linestyle="--", label=f"Best: {best_bal.threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Balanced Accuracy")
    plt.title("Balanced Accuracy vs Threshold (Ensemble)")
    plt.legend()
    ba_path = os.path.splitext(OUT_CSV)[0].replace(".csv", "") + "_BalAcc_vs_Thr.png"
    plt.savefig(ba_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Balanced Accuracy vs Threshold to: {ba_path}")
