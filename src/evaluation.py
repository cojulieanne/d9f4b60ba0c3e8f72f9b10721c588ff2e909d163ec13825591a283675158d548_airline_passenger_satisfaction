
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, auc
from sklearn.exceptions import NotFittedError

import shap
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def evaluate_single_model(model, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a single model on binary classification metrics.

    Returns:
        dict: {accuracy, precision, recall, f1_score, roc_auc}
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except (AttributeError, NotFittedError):
        y_proba = None

    return model, {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else "N/A"
    }


def save_shap_summary_plot(model, X_test, output_path=project_root/"results"/"shap_summary.png"):
    """
    Generates and saves a SHAP beeswarm summary plot for feature contributions
    using TreeExplainer (optimized for tree-based models).

    Args:
        model (sklearn-compatible): Trained tree-based model (e.g., XGBoost, LightGBM).
        X_test (pd.DataFrame): Feature matrix used for SHAP value estimation.
        output_path (str): File path to save the plot image.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use TreeExplainer explicitly
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary (TreeExplainer)")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary saved to {output_path}")

def save_feature_importance_plot(model, X_test, output_path=project_root/"results"/"feature_importance.png"):
    """
    Plots and saves a bar chart of feature importances from a tree-based model.

    Args:
        model (sklearn-compatible): Trained model with .feature_importances_ attribute.
        X_test (pd.DataFrame): Feature matrix for reference column names.
        output_path (str): File path to save the plot image.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    importance = model.feature_importances_
    names = X_test.columns
    sorted_idx = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(names)), importance[sorted_idx])
    plt.yticks(range(len(names)), np.array(names)[sorted_idx])
    plt.title("XGBoost Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Feature importance plot saved to {output_path}")

def save_roc_curve_plot(model, X_test, y_test, output_path=project_root/"results"/"roc_curve.png"):
    """
    Plots and saves the ROC curve with AUC score.

    Args:
        model (sklearn-compatible): Trained model with predict_proba method.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series or array-like): True labels for test set.
        output_path (str): File path to save the plot image.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve saved to {output_path}")

def save_confusion_matrix_plot(model, X_test, y_test, output_path=project_root/"results"/"confusion_matrix.png"):
    """
    Plots and saves the confusion matrix.

    Args:
        model (sklearn-compatible): Trained classifier with predict method.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series or array-like): True labels for test set.
        output_path (str): File path to save the plot image.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")
