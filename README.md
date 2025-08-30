# DFDE-MIDOG2025

**Dual-Fusion Double-Ensemble (DFDE)** for atypical mitosis classification — built for **MIDOG 2025 Track 2**.

The pipeline:
1. **Feature extraction** → HIBOU-L + Barlow Twins (concatenated)  
2. **Training** → LDA → XGBoost **and/or** CatBoost (Focal Loss) with CV + per-domain eval  
3. **Evaluation / Ensembling** → Soft-vote ensemble of XGBoost + CatBoost with threshold sweep & curves  

---

## 📦 Repository

```
DFDE-MIDOG2025/
├─ extract_features_concat.py   # Step 0: Precompute & save HIBOU-L + Barlow Twins features
├─ train_catboost.py            # Step 1a: Train LDA + CatBoost (Focal), CV & per-domain metrics
├─ train_xgboost.py             # Step 1b: Train LDA + XGBoost, CV & per-domain metrics
├─ test_ensemble_code.py        # Step 2: Recreate split, load models, soft-vote, sweep thresholds
└─ README.md
```

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
```

---

## 🔧 Data & Folders  

**CSV**: `MIDOG25_Atypical_Classification_Train_Set.csv`  

Must contain columns:  
- `image_id` (e.g., `xxx.png`)  
- `majority` labels with values **AMF** or **NMF**  
- (Optional) domain columns used in per-domain eval: **Tumor, Scanner, Origin, Species**  

**Images**: only needed if your extractor reads original PNGs.  
**Features**: saved to a directory (see *User Settings* in scripts).  

---

## 🚀 Usage

### 0) Extract & Concatenate Features

```bash
python extract_features_concat.py   --images_dir <PATH_TO_IMAGES>   --csv_path <PATH_TO_CSV>   --out_dir <FEATURE_OUTPUT_DIR>
```

Output: `<image_id>_hibou_bt.pt` tensors saved in the feature directory.  

---

### 1a) Train CatBoost (LDA + Focal Loss)

```bash
python train_catboost.py
```

**What it does**
- Loads features → LDA (n_components=1) → CatBoost (Focal Loss)  
- Performs **GridSearchCV** with StratifiedKFold  
- Saves:  
  - `models/v6/v6_lda_transformer.pkl`  
  - `models/v6/v6_classifier.pkl`  
- Reports metrics: TP/FP/TN/FN, F1, Precision, Recall, Specificity, NPV, Accuracy, Balanced Accuracy, AUC, Log Loss  
- Exports **per-domain CSV metrics** (Tumor, Scanner, Origin, Species)  

---

### 1b) Train XGBoost (LDA + Gradient Boosting)

```bash
python train_xgboost.py
```

**What it does**
- Loads features → LDA (n_components=1) → XGBoost classifier  
- Performs hyperparameter tuning with GridSearchCV  
- Saves:  
  - `models/v5/v5_lda_transformer.pkl`  
  - `models/v5/v5_classifier.pkl`  
- Reports metrics (same as CatBoost)  
- Exports per-domain evaluation  

---

### 2) Ensemble Evaluation (Soft-Vote + Threshold Sweep)

```bash
python test_ensemble_code.py
```

**What it does**
- Recreates the same stratified split as training (via `random_state=42`)  
- Loads both models + their LDA transformers  
- Performs **soft-voting ensemble** (default weights: `W_XGB=0.493`, `W_CB=0.507`)  
- Outputs:  
  - `ensemble_test_predictions.csv` (per-slide probabilities & predictions)  
  - `*_threshold_sweep.csv` (metrics across thresholds)  
  - Curves: `*_PR.png`, `*_ROC.png`, `*_BalAcc_vs_Thr.png`  
- Prints suggested thresholds for: Balanced Accuracy, Youden’s J, and F1  

---

## 🧩 Design Highlights

- **Dual-fusion features**: HIBOU-L + Barlow Twins  
- **Model diversity**: CatBoost (Focal Loss) + XGBoost  
- **Dimensionality reduction**: LDA for compact features  
- **Soft-voting ensemble**: robust predictions from complementary models  
- **Per-domain evaluation**: fairness across Tumor, Scanner, Origin, Species  

---

## 🔁 Reproducibility

- Fixed `random_state=42`  
- Stratified train/test splits  
- Saved transformers and models per version (v5 = XGBoost, v6 = CatBoost)  

---

## 📄 Citation

If you use this repository in academic work:

```bibtex
@misc{DFDE-MIDOG2025,
  title   = {Dual-Fusion Double-Ensemble (DFDE) for MIDOG 2025 Track 2},
  author  = {Cayir, Sercan and Kukuk, Suha Berk},
  year    = {2025},
  url     = {https://github.com/DigitalPathology/DFDE-MIDOG2025}
}
```
