# plot_results.py
"""
Plotting script for benchmark results.

Generates visualizations for model evaluation on test data:
1. ROC Curves (all models)
2. Confusion Matrices
3. Model Comparison Bar Charts
4. Feature Importance (interpretable features)
5. Precision-Recall Curves
6. Performance Summary Table
7. Score Distributions (impairment probability by class)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, precision_recall_curve,
    accuracy_score, f1_score, roc_auc_score, classification_report
)
import joblib

import config
from models import MODEL_REGISTRY


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_predictions(model_id: str):
    """Load saved predictions for a model."""
    safe_name = model_id.replace("/", "__")
    pred_path = os.path.join(config.OUTPUT_DIR, "predictions", f"{safe_name}_predictions.npz")

    if not os.path.exists(pred_path):
        print(f"Warning: Predictions not found for {model_id}")
        return None

    data = np.load(pred_path, allow_pickle=True)

    # Convert to dict and handle both old and new naming conventions
    result = {key: data[key] for key in data.files}

    # Map new names to old names for compatibility
    if 'y_test_true' in result and 'y_test' not in result:
        result['y_test'] = result['y_test_true']
        result['y_val'] = result['y_val_true']
        result['y_train'] = result['y_train_true']
    if 'scores_test' in result and 'y_test_prob' not in result:
        result['y_test_prob'] = result['scores_test']
        result['y_val_prob'] = result['scores_val']
        result['y_train_prob'] = result['scores_train']

    return result


def load_classifier(model_id: str):
    """Load saved classifier for a model."""
    safe_name = model_id.replace("/", "__")
    clf_path = os.path.join(config.OUTPUT_DIR, "classifiers", f"{safe_name}_classifier.joblib")

    if not os.path.exists(clf_path):
        print(f"Warning: Classifier not found for {model_id}")
        return None

    return joblib.load(clf_path)


def load_feature_names(model_id: str):
    """Load feature names for a model."""
    safe_name = model_id.replace("/", "__")
    names_path = os.path.join(config.FEATURES_DIR, f"{safe_name}_feature_names.txt")

    if not os.path.exists(names_path):
        return None

    with open(names_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


# =============================================================================
# PLOT 1: ROC CURVES (All Models)
# =============================================================================
def plot_roc_curves(save_path: str = None):
    """
    Plot ROC curves for all models on test data.

    Shows:
    - ROC curve for each model
    - AUC values in legend
    - Diagonal reference line
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(MODEL_REGISTRY)))

    for i, entry in enumerate(MODEL_REGISTRY):
        model_id = entry["model_id"]
        model_name = entry["model_name"]

        preds = load_predictions(model_id)
        if preds is None:
            continue

        y_test = preds['y_test']
        y_prob = preds['y_test_prob']

        if y_prob is None:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})')

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison (Test Set)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# PLOT 2: CONFUSION MATRICES
# =============================================================================
def plot_confusion_matrices(save_path: str = None):
    """
    Plot confusion matrices for all models.

    Shows:
    - Grid of confusion matrices
    - Counts and percentages
    """
    n_models = len(MODEL_REGISTRY)
    n_cols = 2
    n_rows = (n_models + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for i, entry in enumerate(MODEL_REGISTRY):
        model_id = entry["model_id"]
        model_name = entry["model_name"]

        preds = load_predictions(model_id)
        if preds is None:
            continue

        y_test = preds['y_test']
        y_pred = preds['y_test_pred']

        cm = confusion_matrix(y_test, y_pred)

        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['NC (0)', 'Impaired (1)'],
                    yticklabels=['NC (0)', 'Impaired (1)'])
        axes[i].set_xlabel('Predicted', fontsize=10)
        axes[i].set_ylabel('Actual', fontsize=10)
        axes[i].set_title(f'{model_name}', fontsize=12)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Confusion Matrices - Test Set', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# PLOT 3: MODEL COMPARISON BAR CHART
# =============================================================================
def plot_model_comparison(save_path: str = None):
    """
    Plot bar chart comparing models on key metrics.

    Shows:
    - Accuracy, F1 Score, AUC for each model
    - Grouped bars
    """
    metrics_data = []

    for entry in MODEL_REGISTRY:
        model_id = entry["model_id"]
        model_name = entry["model_name"]

        preds = load_predictions(model_id)
        if preds is None:
            continue

        y_test = preds['y_test']
        y_pred = preds['y_test_pred']
        y_prob = preds['y_test_prob']

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob) if y_prob is not None else 0

        metrics_data.append({
            'Model': model_name,
            'Accuracy': acc,
            'F1 Score': f1,
            'AUC': auc_score
        })

    df = pd.DataFrame(metrics_data)

    # Melt for grouped bar chart
    df_melted = df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.25

    metrics = ['Accuracy', 'F1 Score', 'AUC']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = df[metric].values
        bars = ax.bar(x + i * width, values, width, label=metric, color=color, alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison - Test Set Performance', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(df['Model'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# PLOT 4: FEATURE IMPORTANCE (Interpretable Features)
# =============================================================================
def plot_feature_importance(save_path: str = None):
    """
    Plot feature importance for interpretable features across models.

    Shows:
    - Bar chart of feature weights for top 10 interpretable features
    - One subplot per model
    """
    n_models = len(MODEL_REGISTRY)
    n_cols = 2
    n_rows = (n_models + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for i, entry in enumerate(MODEL_REGISTRY):
        model_id = entry["model_id"]
        model_name = entry["model_name"]

        clf = load_classifier(model_id)
        feature_names = load_feature_names(model_id)

        if clf is None or feature_names is None:
            continue

        if not hasattr(clf, 'coef_'):
            axes[i].text(0.5, 0.5, 'No coefficients available',
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{model_name}')
            continue

        # Get interpretable feature weights (first 10)
        coefs = clf.coef_[0][:10]
        names = feature_names[:10]

        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        coefs_sorted = coefs[sorted_idx]
        names_sorted = [names[j] for j in sorted_idx]

        # Color by sign
        colors = ['#e74c3c' if c > 0 else '#3498db' for c in coefs_sorted]

        bars = axes[i].barh(range(len(coefs_sorted)), coefs_sorted, color=colors, alpha=0.8)
        axes[i].set_yticks(range(len(names_sorted)))
        axes[i].set_yticklabels(names_sorted)
        axes[i].set_xlabel('Coefficient Weight')
        axes[i].set_title(f'{model_name}')
        axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[i].invert_yaxis()

        # Add legend
        axes[i].plot([], [], 's', color='#e74c3c', label='Positive (-> Impaired)')
        axes[i].plot([], [], 's', color='#3498db', label='Negative (-> NC)')
        axes[i].legend(loc='lower right', fontsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Importance - Interpretable Features', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# PLOT 5: PRECISION-RECALL CURVES
# =============================================================================
def plot_precision_recall_curves(save_path: str = None):
    """
    Plot Precision-Recall curves for all models.

    Important for imbalanced datasets.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(MODEL_REGISTRY)))

    for i, entry in enumerate(MODEL_REGISTRY):
        model_id = entry["model_id"]
        model_name = entry["model_name"]

        preds = load_predictions(model_id)
        if preds is None:
            continue

        y_test = preds['y_test']
        y_prob = preds['y_test_prob']

        if y_prob is None:
            continue

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)

        ax.plot(recall, precision, color=colors[i], lw=2,
                label=f'{model_name} (AP = {pr_auc:.3f})')

    # Baseline (random classifier)
    baseline = np.sum(preds['y_test']) / len(preds['y_test'])
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1,
               label=f'Baseline (AP = {baseline:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Model Comparison (Test Set)', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# PLOT 6: PERFORMANCE SUMMARY TABLE
# =============================================================================
def create_summary_table(save_path: str = None):
    """
    Create a summary table of all metrics.
    """
    summary_data = []

    for entry in MODEL_REGISTRY:
        model_id = entry["model_id"]
        model_name = entry["model_name"]

        preds = load_predictions(model_id)
        if preds is None:
            continue

        y_test = preds['y_test']
        y_pred = preds['y_test_pred']
        y_prob = preds['y_test_prob']

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        auc_score = roc_auc_score(y_test, y_prob) if y_prob is not None else 0

        summary_data.append({
            'Model': model_name,
            'Accuracy': f'{acc:.3f}',
            'F1 Score': f'{f1:.3f}',
            'AUC': f'{auc_score:.3f}',
            'Sensitivity': f'{sensitivity:.3f}',
            'Specificity': f'{specificity:.3f}',
            'PPV': f'{ppv:.3f}',
            'NPV': f'{npv:.3f}',
        })

    df = pd.DataFrame(summary_data)

    # Create figure for table
    fig, ax = plt.subplots(figsize=(14, 2 + 0.5 * len(df)))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title('Performance Summary - Test Set', fontsize=14, pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

        # Also save as CSV
        csv_path = save_path.replace('.png', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    return fig, df


# =============================================================================
# PLOT 7: SCORE DISTRIBUTION
# =============================================================================
def plot_score_distributions(save_path: str = None):
    """
    Plot distribution of impairment scores (0-1) for each model.

    Shows:
    - Score distribution for NC (Normal Control) samples
    - Score distribution for Impaired samples
    - Overlap indicates classification difficulty
    """
    n_models = len(MODEL_REGISTRY)
    n_cols = 2
    n_rows = (n_models + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for i, entry in enumerate(MODEL_REGISTRY):
        model_id = entry["model_id"]
        model_name = entry["model_name"]

        preds = load_predictions(model_id)
        if preds is None:
            continue

        y_test = preds['y_test']
        scores = preds.get('y_test_prob', preds.get('scores_test'))

        if scores is None:
            continue

        # Separate scores by class
        scores_nc = scores[y_test == 0]
        scores_impaired = scores[y_test == 1]

        # Plot distributions
        axes[i].hist(scores_nc, bins=20, alpha=0.6, color='#3498db',
                     label=f'NC (n={len(scores_nc)})', density=True)
        axes[i].hist(scores_impaired, bins=20, alpha=0.6, color='#e74c3c',
                     label=f'Impaired (n={len(scores_impaired)})', density=True)

        # Add threshold line
        axes[i].axvline(x=0.5, color='black', linestyle='--', lw=2, label='Threshold (0.5)')

        # Add mean lines
        axes[i].axvline(x=np.mean(scores_nc), color='#3498db', linestyle='-', lw=2, alpha=0.8)
        axes[i].axvline(x=np.mean(scores_impaired), color='#e74c3c', linestyle='-', lw=2, alpha=0.8)

        axes[i].set_xlabel('Impairment Score (0=NC, 1=Impaired)', fontsize=10)
        axes[i].set_ylabel('Density', fontsize=10)
        axes[i].set_title(f'{model_name}', fontsize=12)
        axes[i].legend(loc='upper center', fontsize=9)
        axes[i].set_xlim([0, 1])

        # Add text with mean scores
        axes[i].text(0.02, 0.95, f'NC mean: {np.mean(scores_nc):.3f}',
                    transform=axes[i].transAxes, fontsize=9, verticalalignment='top',
                    color='#3498db', fontweight='bold')
        axes[i].text(0.02, 0.88, f'Impaired mean: {np.mean(scores_impaired):.3f}',
                    transform=axes[i].transAxes, fontsize=9, verticalalignment='top',
                    color='#e74c3c', fontweight='bold')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Impairment Score Distributions by Class (Test Set)', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# MAIN: GENERATE ALL PLOTS
# =============================================================================
def generate_all_plots():
    """Generate all plots and save to results/plots/ directory."""

    # Create plots directory
    plots_dir = os.path.join(config.OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print("GENERATING EVALUATION PLOTS")
    print("=" * 70)

    # 1. ROC Curves
    print("\n1. Generating ROC Curves...")
    plot_roc_curves(os.path.join(plots_dir, "roc_curves.png"))

    # 2. Confusion Matrices
    print("\n2. Generating Confusion Matrices...")
    plot_confusion_matrices(os.path.join(plots_dir, "confusion_matrices.png"))

    # 3. Model Comparison
    print("\n3. Generating Model Comparison...")
    plot_model_comparison(os.path.join(plots_dir, "model_comparison.png"))

    # 4. Feature Importance
    print("\n4. Generating Feature Importance...")
    plot_feature_importance(os.path.join(plots_dir, "feature_importance.png"))

    # 5. Precision-Recall Curves
    print("\n5. Generating Precision-Recall Curves...")
    plot_precision_recall_curves(os.path.join(plots_dir, "precision_recall_curves.png"))

    # 6. Summary Table
    print("\n6. Generating Summary Table...")
    create_summary_table(os.path.join(plots_dir, "summary_table.png"))

    # 7. Score Distributions
    print("\n7. Generating Score Distributions...")
    plot_score_distributions(os.path.join(plots_dir, "score_distributions.png"))

    print("\n" + "=" * 70)
    print(f"All plots saved to: {plots_dir}")
    print("=" * 70)

    # Show plots
    plt.show()


if __name__ == "__main__":
    generate_all_plots()
