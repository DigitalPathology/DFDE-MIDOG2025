# DFDE-MIDOG2025

**Dual-Fusion Double-Ensemble (DFDE)** for atypical mitosis classification — built for **MIDOG 2025 Track 2**.

The pipeline:
1) **Feature extraction** → HIBOU-L + Barlow Twins (concatenated)
2) **Training** → LDA → CatBoost (Focal Loss) with CV + per-domain eval
3) **Evaluation / Ensembling** → XGBoost + CatBoost soft-vote + threshold sweep & curves

---

## 📦 Repository

DFDE-MIDOG2025/
├─ extract_features_concat.py # Step 0: Precompute & save HIBOU-L + Barlow Twins features
├─ train_pipeline.py # Step 1: Train LDA + CatBoost (Focal), CV & per-domain metrics
├─ test_ensemble_code.py # Step 2: Recreate split, load models, soft-vote, sweep thresholds
└─ README.md


> **Notes**
> - The code expects **concatenated features** saved as `.pt` tensors per image with the naming pattern:
>   `image_id.png  →  image_id_hibou_bt.pt`
> - Paths in scripts are currently **Windows-style** (e.g., `D:\...`). Adjust for your environment.

---

## ⚙️ Requirements

- Python ≥ 3.10
- PyTorch
- scikit-learn
- xgboost
- catboost
- pandas, numpy, joblib, matplotlib

```bash
pip install torch scikit-learn xgboost catboost pandas numpy joblib matplotlib

🔧 Data & Folders

CSV: MIDOG25_Atypical_Classification_Train_Set.csv

    Must contain columns:

      image_id (e.g., xxx.png)

      majority labels with values AMF or NMF

      (Optional) domain columns used in per-domain eval: Tumor, Scanner, Origin, Species

Images: only needed if your extractor reads original PNGs.

Features: saved to a directory (see User Settings in scripts).

🚀 Usage
0) Extract & Concatenate Features (NEW)

Run your new extractor to compute HIBOU-L + Barlow Twins and save concatenated tensors:

python extract_features_concat.py \
  --images_dir <PATH_TO_IMAGES> \
  --csv_path <PATH_TO_CSV> \
  --out_dir <FEATURE_OUTPUT_DIR>

Output expectation

  One file per image: <image_id>_hibou_bt.pt

  Tensor shape: (D,) or (1, D) (both are handled; (1,D) is squeezed).

Make sure the filenames match the pattern used in train_pipeline.py / test_ensemble_code.py.

1) Train (LDA → CatBoost with Focal Loss)

Open train_pipeline.py and set the User Settings at the top:

feature_dir = r"D:/dl/MIDOG_2025_Track_2/HIBOU_L_and_Barlow_Concatenated_Features/"
csv_path    = r"D:\dl\MIDOG_2025_Track_2\MIDOG25_Atypical_Classification_Train_Set.csv"

Then run:
python train_pipeline.py

What it does

  Loads concatenated features
  
  Maps labels: NMF → 1, AMF → 0
  
  LDA (n_components=1), then CatBoost + Focal Loss via GridSearchCV (StratifiedKFold)
  
  Saves:
  
  LDA: models/v6/v6_lda_transformer.pkl
  
  CatBoost: models/v6/v6_classifier.pkl
  
  Prints test metrics and saves per-domain CSVs in models/v6/:
  
  Tumor_metrics.csv, Scanner_metrics.csv, Origin_metrics.csv, Species_metrics.csv

Reported metrics

TP/FP/TN/FN, F1, Precision, Recall, Specificity, NPV, Accuracy, Balanced Accuracy, AUC, Log Loss

2) Ensemble Evaluation (Soft-Vote + Threshold Sweep)

Open test_ensemble_code.py and set:

feature_dir = r"D:/dl/MIDOG_2025_Track_2/HIBOU_L_and_Barlow_Concatenated_Features/"
csv_path    = r"D:\dl\MIDOG_2025_Track_2\MIDOG25_Atypical_Classification_Train_Set.csv"

lda_xgb_path = r"D:/dl/MIDOG_2025_Track_2/models/v5/v5_lda_transformer.pkl"
lda_cb_path  = r"D:/dl/MIDOG_2025_Track_2/models/v6/v6_lda_transformer.pkl"

xgb_paths = [r"D:\dl\MIDOG_2025_Track_2\models\v5\v5_classifier.pkl"]
cb_paths  = [r"D:\dl\MIDOG_2025_Track_2\models\v6\v6_classifier.pkl"]

OUT_CSV   = r"D:\dl\MIDOG_2025_Track_2\ensemble_test_predictions.csv"

Run:

python test_ensemble_code.py

What it does

  Recreates the same stratified split used during training (by splitting on indices with the same random_state)
  
  Loads XGBoost + CatBoost and their own LDA transformers
  
  Soft-voting (default normalized weights: W_XGB=0.493, W_CB=0.507)
  
  Saves per-slide predictions to:
  
  ensemble_test_predictions.csv
  
  Performs threshold sweep across [0,1] and saves:
  
  *_threshold_sweep.csv
  
  Exports plots:
  
  *_PR.png (Precision–Recall)
  
  *_ROC.png (ROC)
  
  *_BalAcc_vs_Thr.png (Balanced Accuracy vs Threshold)

Suggested thresholds (printed)

  Max Balanced Accuracy
  
  Max Youden’s J
  
  Max F1

🧩 Design Highlights

  Dual-fusion features: HIBOU-L + Barlow Twins concatenation for richer representations
  
  Dimensionality reduction: LDA → compact 1-D discriminative space per model view
  
  Focal Loss in CatBoost: robustness to class imbalance / hard examples
  
  Soft-voting ensemble: complementary decision surfaces (XGBoost + CatBoost)
  
  Per-domain evaluation: fairness & robustness by Tumor/Scanner/Origin/Species

🔁 Reproducibility

  Fixed random_state=42 and stratified splits
  
  LDA transformers saved and reused per model family
  
  Feature naming convention enforces correct alignment between CSV and .pt files

📄 Citation

If you use this repository in academic work:

@misc{DFDE-MIDOG2025,
  title   = {Dual-Fusion Double-Ensemble (DFDE) for MIDOG 2025 Track 2},
  author  = {Cayir, Sercan and Kukuk, Suha Berk},
  year    = {2025},
  url     = {https://github.com/DigitalPathology/DFDE-MIDOG2025}
}

