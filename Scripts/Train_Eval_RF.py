# train_random_forest.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# =========== CONFIG ===========
CSV_PATH = "LA_climate_fire_labeled.csv"   
TARGET_COL = "fire_occurred"               
DATE_COLS = ["date"]                       # any non-feature date-like columns to drop from X
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 500
MAX_DEPTH = 15            
CLASS_WEIGHT = "balanced"  
OUT_DIR = "rf_outputs"
# ==============================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load data
    df = pd.read_csv(CSV_PATH, parse_dates=[c for c in DATE_COLS if c in pd.read_csv(CSV_PATH, nrows=0).columns])

    # 2) Separate y
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {CSV_PATH}.")
    y = df[TARGET_COL].astype(int)

    # 3) Build feature matrix X
    #    Start with numeric columns only, then drop target/date-like columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in [TARGET_COL] + [c for c in DATE_COLS if c in df.columns]:
        if col in numeric_cols:
            numeric_cols.remove(col)
    X = df[numeric_cols].copy()

    
    print(f"Using {len(numeric_cols)} features:\n{numeric_cols}\n")

    # 4) Train/test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 5) Handle any missing values
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # 6) Class imbalance (either 'balanced' or manual weights)
    if CLASS_WEIGHT == "balanced":
        class_weight = "balanced"
    else:
        # Manual weights example (majority: 1, minority: ratio)
        counts = y_train.value_counts()
        class_weight = {0: 1.0, 1: (counts[0] / counts[1]) * 2.0}

    # 7) Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=class_weight,
    )
    rf.fit(X_train_imp, y_train)

    # 8) Evaluate
    y_pred = rf.predict(X_test_imp)
    y_proba = rf.predict_proba(X_test_imp)[:, 1]

    # For data visualization
    save_metric_pngs(y_test.values if hasattr(y_test, "values") else y_test, y_proba, OUT_DIR, prefix="rf")


    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC:  {pr_auc:.3f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 9) Feature importance
    importances = pd.DataFrame({
        "feature": numeric_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    importances_path = os.path.join(OUT_DIR, "feature_importances.csv")
    importances.to_csv(importances_path, index=False)

    # 10) Save artifacts
    joblib.dump({"model": rf, "imputer": imputer, "features": numeric_cols}, os.path.join(OUT_DIR, "rf_model.joblib"))

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }, f, indent=2)

    print(f"\nSaved artifacts in ./{OUT_DIR}/")
    print(f"- Model: rf_model.joblib")
    print(f"- Metrics: metrics.json")
    print(f"- Feature importances: feature_importances.csv")

def save_metric_pngs(y_true, y_score, out_dir, prefix):
    """
    Saves two PNG images in out_dir:
      - <prefix>_pr_curve.png                (Precision–Recall curve with AP)
      - <prefix>_metrics_vs_threshold.png    (Precision, Recall, F1 vs. threshold)
    """
    font_size_labels = 14
    font_size_title = 16
    font_size_legend = 12

    # Precision–Recall curve
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6,4))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall", fontsize=font_size_labels)
    plt.ylabel("Precision", fontsize=font_size_labels)
    plt.ylim(0, 1)
    plt.title(f"{prefix.upper()} — Precision–Recall Curve", fontsize=font_size_title)
    plt.legend(fontsize=font_size_legend)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_pr_curve.png"), dpi=150)
    plt.close()

    # Metrics vs. threshold
    thresholds = np.linspace(0, 1, 501)
    P, R, F1 = [], [], []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        P.append(p); R.append(r); F1.append(f1)

    plt.figure(figsize=(6,4))
    plt.plot(thresholds, P, label="Precision")
    plt.plot(thresholds, R, label="Recall")
    plt.plot(thresholds, F1, label="F1-score")
    plt.xlabel("Threshold", fontsize=font_size_labels)
    plt.ylabel("Score", fontsize=font_size_labels)
    plt.ylim(0, 1)
    plt.title(f"{prefix.upper()} — Metrics vs Threshold", fontsize=font_size_title)
    plt.legend(fontsize=font_size_legend)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_metrics_vs_threshold.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
