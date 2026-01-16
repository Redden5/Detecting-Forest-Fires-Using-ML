# train_xgb_balanced.py
import os, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve
)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


# ========== CONFIG ==========
CSV_PATH = "LA_climate_fire_labeled.csv"
TARGET_COL = "fire_occurred"     # 0/1
DATE_COLS  = ["date"]
FEATURES   = ["TMAX","TMIN","TAVG","PRCP","PRCP_6mo_avg"]  # set None to auto-pick numeric
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUT_DIR = "xgb_outputs_balanced"

# Balancing options
BALANCE_METHOD = "undersample"   
NEG_TO_POS_RATIO = 5             # for undersample: keep at most this many negatives per positive

# Threshold target
TARGET_RECALL = 0.60             # aim for 0.6 recall on fires
MIN_PRECISION = 0.30             # don’t accept thresholds with precision below this
# ===========================

XGB_PARAMS = dict(
    n_estimators=1500,
    max_depth=4,
    learning_rate=0.06,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="auc"
)

def balance_train(X, y, method="undersample", ratio=5):
    if method == "none":
        return X, y

    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y  # nothing to balance

    if method == "undersample":
        keep_neg = min(len(neg_idx), ratio * len(pos_idx))
        neg_sample = np.random.RandomState(RANDOM_STATE).choice(neg_idx, size=keep_neg, replace=False)
        keep_idx = np.concatenate([pos_idx, neg_sample])
        return X.loc[keep_idx], y.loc[keep_idx]

    return X, y

def pick_threshold(proba, y_true, target_recall=TARGET_RECALL, min_precision=MIN_PRECISION):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    # thr length = len(rec)-1; align by trimming last rec/prec
    candidates = []
    for i in range(len(thr)):
        if rec[i] >= target_recall and prec[i] >= min_precision:
            candidates.append((thr[i], prec[i], rec[i], 2*prec[i]*rec[i]/(prec[i]+rec[i]+1e-9)))
    if not candidates:
        # fallback to best F1 overall
        f1s = []
        for i in range(len(thr)):
            f1 = 2*prec[i]*rec[i]/(prec[i]+rec[i]+1e-9)
            f1s.append((thr[i], prec[i], rec[i], f1))
        thr_best = max(f1s, key=lambda t: t[3])[0] if f1s else 0.5
        return float(thr_best)
    # choose the candidate with highest precision (avoid collapsing precision), tie-break by F1
    thr_best = sorted(candidates, key=lambda t: (t[1], t[3]), reverse=True)[0][0]
    return float(thr_best)

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

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    raw = pd.read_csv(CSV_PATH, parse_dates=[c for c in DATE_COLS if c in pd.read_csv(CSV_PATH, nrows=0).columns])
    y = raw[TARGET_COL].astype(int)

    if FEATURES is None:
        numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
        for c in [TARGET_COL] + [d for d in DATE_COLS if d in raw.columns]:
            if c in numeric_cols:
                numeric_cols.remove(c)
        feats = numeric_cols
    else:
        feats = FEATURES

    X = raw[feats].copy()
    print(f"Using {len(feats)} features: {feats}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    imp = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=feats, index=X_train.index)
    X_test_imp  = pd.DataFrame(imp.transform(X_test),  columns=feats, index=X_test.index)

    # Balance only the training set
    Xb, yb = balance_train(X_train_imp, y_train, method=BALANCE_METHOD, ratio=NEG_TO_POS_RATIO)
    print(f"Train size original: {len(X_train)} | balanced: {len(Xb)} (positives: {int(yb.sum())}, negatives: {int((yb==0).sum())})")

    # scale_pos_weight on the balanced set (still helpful)
    neg, pos = (yb == 0).sum(), (yb == 1).sum()
    spw = float(neg / pos) if pos > 0 else 1.0
    print(f"scale_pos_weight (balanced): {spw:.2f}")

    model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw, random_state=RANDOM_STATE)
    model.fit(Xb, yb)

    # Evaluate default threshold
    proba = model.predict_proba(X_test_imp)[:, 1]
    y_pred_05 = (proba >= 0.5).astype(int)

    # For data visualization
    save_metric_pngs(y_test.values if hasattr(y_test, "values") else y_test, proba, OUT_DIR, prefix="xgb")


    metrics = {}
    metrics["roc_auc_0p5"] = float(roc_auc_score(y_test, proba))
    metrics["pr_auc"] = float(average_precision_score(y_test, proba))
    metrics["cm_0p5"] = confusion_matrix(y_test, y_pred_05).tolist()
    metrics["report_0p5"] = classification_report(y_test, y_pred_05, output_dict=True)

    print(f"\nDefault(0.5) ROC-AUC: {metrics['roc_auc_0p5']:.3f} | PR-AUC: {metrics['pr_auc']:.3f}")
    print("Confusion @0.5:\n", np.array(metrics["cm_0p5"]))
    print("Report @0.5:\n", classification_report(y_test, y_pred_05, digits=3))

    # Threshold tuning (highest precision ≥ MIN_PRECISION that meets recall target)
    thr = pick_threshold(proba, y_test, TARGET_RECALL, MIN_PRECISION)
    y_pred_tuned = (proba >= thr).astype(int)
    metrics["threshold_tuned"] = thr
    metrics["cm_tuned"] = confusion_matrix(y_test, y_pred_tuned).tolist()
    metrics["report_tuned"] = classification_report(y_test, y_pred_tuned, output_dict=True)

    print(f"\nTuned threshold (target recall ≥ {TARGET_RECALL}, min precision {MIN_PRECISION}): {thr:.3f}")
    print("Confusion (tuned):\n", np.array(metrics["cm_tuned"]))
    print("Report (tuned):\n", classification_report(y_test, y_pred_tuned, digits=3))

    # Save artifacts
    joblib.dump({"model": model, "imputer": imp, "features": feats}, os.path.join(OUT_DIR, "xgb_model_balanced.joblib"))
    with open(os.path.join(OUT_DIR, "metrics_balanced.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved in ./{OUT_DIR}/")
    print("- Model: xgb_model_balanced.joblib")
    print("- Metrics: metrics_balanced.json")    

if __name__ == "__main__":
    main()
