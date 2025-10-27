import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from inspect import signature

from typing import Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    recall_score,
    accuracy_score,
    precision_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

import torch

# TabPFN import can vary slightly across versions. Try both.
try:
    from tabpfn import TabPFNClassifier
except Exception:
    from tabpfn.scripts.tabpfn_classifier import TabPFNClassifier  # fallback for older layouts


def _split_xy(
    df: pd.DataFrame,
    y_col_name: str = "label"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Split features/labels, keep numeric features only, and fill NaNs."""
    if y_col_name not in df.columns:
        raise ValueError(f"y_col_name='{y_col_name}' not found in columns: {list(df.columns)}")

    y = df[y_col_name].values
    X = df.drop(columns=[y_col_name])

    # Only numeric features for safety
    X = X.select_dtypes(include=[np.number]).copy()

    # Fill NaNs if any
    X = X.fillna(0.0)

    # Ensure contiguous arrays
    return X, y


def _ensure_binary_labels(y: np.ndarray) -> np.ndarray:
    """
    Ensure labels are 0/1. If labels are strings or {0,1}—works as-is;
    if labels are {1,2} or any categorical, we map to {0,1}.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    if len(np.unique(y_enc)) != 2:
        raise ValueError(f"TabPFN script expects binary classification. Got classes: {np.unique(y)}")
    return y_enc


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png: str
) -> None:
    """Save confusion matrix figure (similar to your XGB flow)."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    # Show values
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def tabpfn_model_train(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    tabpfn_model_path: str,
    y_col_name: str = "label",
    n_ensembles: int = 8,
    device: Optional[str] = None,
    calibration: bool = True,
) -> TabPFNClassifier:
    """
    Train TabPFN on given train/valid DataFrames (feature maps with a label column).
    Saves the trained model to `tabpfn_model_path` and returns it.
    Also prints metrics and writes evaluation_results_tabpfn.csv and confusion_matrix_tabpfn.png.
    """

    # Device resolution
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare data
    X_train, y_train = _split_xy(train_data, y_col_name)
    X_valid, y_valid = _split_xy(valid_data, y_col_name)

    y_train = _ensure_binary_labels(y_train)
    y_valid = _ensure_binary_labels(y_valid)

    # TabPFN classifier
    # Resolve device automatically if "auto" or None
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build TabPFN classifier with version-tolerant kwargs
    init_sig = signature(TabPFNClassifier)
    kwargs = {}

    # Ensemble size param (different names across versions)
    if "n_estimators" in init_sig.parameters:
        kwargs["n_estimators"] = n_ensembles
    elif "N_estimators" in init_sig.parameters:
        kwargs["N_estimators"] = n_ensembles
    elif "N_ensemble_configurations" in init_sig.parameters:
        kwargs["N_ensemble_configurations"] = n_ensembles  # older API

    # Device (only if supported)
    if "device" in init_sig.parameters:
        kwargs["device"] = device

    # Low-memory options (only if supported)
    if "fit_mode" in init_sig.parameters:
        kwargs["fit_mode"] = "low_memory"
    if "memory_saving_mode" in init_sig.parameters:
        kwargs["memory_saving_mode"] = "auto"
    if "ignore_pretraining_limits" in init_sig.parameters:
        kwargs["ignore_pretraining_limits"] = True

    clf = TabPFNClassifier(**kwargs)
    print("TabPFN init kwargs:", kwargs)

    print("Using the following parameters for model training:")
    print(f"y_col_name: {y_col_name}")
    print(f"N_ensemble_configurations: {n_ensembles}")
    print(f"device: {device}")
    print(f"calibration: {calibration}")

    # Fit
    clf.fit(X_train.values, y_train)

    # Predict probabilities on validation
    proba_valid = clf.predict_proba(X_valid.values)[:, 1]
    y_pred_valid = (proba_valid >= 0.5).astype(int)

    # Optionally calibrate (simple Platt scaling via sklearn if desired)
    # Skipping explicit calibration here; TabPFN often performs well as-is.

    # Metrics
    # Guard AUC for single-class edge cases
    try:
        auc = roc_auc_score(y_valid, proba_valid)
    except ValueError:
        auc = float("nan")

    acc = accuracy_score(y_valid, y_pred_valid)
    prec = precision_score(y_valid, y_pred_valid, zero_division=0)
    rec = recall_score(y_valid, y_pred_valid, zero_division=0)
    f1 = f1_score(y_valid, y_pred_valid, zero_division=0)

    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Precision: {prec:.4f}")
    print(f"Validation Recall: {rec:.4f}")
    print(f"Validation F1: {f1:.4f}")

    # Save model (pickle)
    os.makedirs(os.path.dirname(tabpfn_model_path) or ".", exist_ok=True)
    with open(tabpfn_model_path, "wb") as f:
        pickle.dump(clf, f)

    # Save evaluation results
    eval_csv = os.path.join(os.path.dirname(tabpfn_model_path) or ".", "evaluation_results_tabpfn.csv")
    pd.DataFrame(
        {
            "auc": [auc],
            "accuracy": [acc],
            "precision": [prec],
            "recall": [rec],
            "f1": [f1],
        }
    ).to_csv(eval_csv, index=False)

    # Save confusion matrix figure
    cm_png = os.path.join(os.path.dirname(tabpfn_model_path) or ".", "confusion_matrix_tabpfn.png")
    plot_and_save_confusion_matrix(y_valid, y_pred_valid, cm_png)

    return clf


if __name__ == "__main__":
    # Example: train on pre-extracted feature maps saved by your pipeline
    # Adjust paths as needed.
    train_csv = "./output/xgb_train_featuremap.csv"   # training features (+label)
    valid_csv = "./output/valid_featuremap.csv"       # validation features (+label)
    out_model = "./output/tabpfn_model.pkl"           # saved TabPFN model

    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    _ = tabpfn_model_train(
        train_data=train_df,
        valid_data=valid_df,
        tabpfn_model_path=out_model,
        y_col_name="label",
        n_ensembles=8,       # reduce if you run out of VRAM/CPU
        device="auto",       # "cuda" / "cpu" / "auto"
        calibration=False,   # keep False unless you add an explicit calibrator
    )
