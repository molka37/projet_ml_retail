# utils.py
# Fonctions utilitaires partagées entre preprocessing.py, train_model.py et predict.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve
)


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str, width: int = 40) -> None:
    """Affiche un titre de section formaté dans le terminal."""
    print(f"\n{'=' * width}")
    print(f"{title}")
    print(f"{'=' * width}")


def print_df_info(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Résumé rapide d'un DataFrame : shape, types, NaN, doublons."""
    print_section(f"Info — {label}")
    print(f"  Shape      : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"  NaN total  : {df.isnull().sum().sum()}")
    print(f"  Doublons   : {df.duplicated().sum()}")
    print(f"  Types      :")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"    {str(dtype):<12} → {count} colonnes")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE QUALITÉ DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un DataFrame trié par taux de valeurs manquantes décroissant.
    Colonnes : count, percent.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    report = pd.DataFrame({
        'count'  : missing,
        'percent': (missing / len(df) * 100).round(2)
    }).sort_values('percent', ascending=False)
    return report


def outlier_report(df: pd.DataFrame, cols: list | None = None,
                   z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Détecte les outliers via le Z-score sur les colonnes numériques.

    Parameters
    ----------
    df       : DataFrame source
    cols     : liste de colonnes à analyser (None = toutes les numériques)
    z_thresh : seuil Z-score (défaut 3.0)

    Returns
    -------
    DataFrame avec count et percent d'outliers par colonne
    """
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()

    results = []
    for col in cols:
        series = df[col].dropna()
        z = np.abs((series - series.mean()) / series.std())
        n_out = (z > z_thresh).sum()
        results.append({
            'feature': col,
            'count'  : n_out,
            'percent': round(n_out / len(series) * 100, 2)
        })

    return (pd.DataFrame(results)
              .sort_values('percent', ascending=False)
              .reset_index(drop=True))


def correlation_matrix(df: pd.DataFrame, target: str = 'Churn',
                       threshold: float = 0.8) -> list:
    """
    Affiche la heatmap de corrélation et retourne les paires fortement corrélées.

    Parameters
    ----------
    threshold : corrélation absolue au-delà de laquelle on signale la paire

    Returns
    -------
    Liste de tuples (feat1, feat2, corr) pour |corr| > threshold
    """
    num_df = df.select_dtypes(include=np.number)
    corr   = num_df.corr()

    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                annot=False, linewidths=0.3, vmin=-1, vmax=1)
    plt.title("Matrice de corrélation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Paires à forte corrélation (hors diagonale)
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                high_corr.append((corr.columns[i], corr.columns[j], round(val, 4)))

    if high_corr:
        print(f"\n  Paires avec |corr| ≥ {threshold} :")
        for f1, f2, v in sorted(high_corr, key=lambda x: -abs(x[2])):
            print(f"    {f1:<35} ↔ {f2:<35} : {v:+.4f}")

    return high_corr


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred,
                          labels: list | None = None,
                          save_path: str | None = None) -> None:
    """Affiche la matrice de confusion avec annotations."""
    if labels is None:
        labels = ['Fidèle (0)', 'Churn (1)']

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor='gray')
    plt.ylabel('Réel', fontweight='bold')
    plt.xlabel('Prédit', fontweight='bold')
    plt.title('Matrice de confusion', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Sauvegardée → {save_path}")
    plt.show()


def plot_roc_curve(y_true, y_proba,
                   save_path: str | None = None) -> float:
    """Trace la courbe ROC et retourne l'AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2,
             label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.fill_between(fpr, tpr, alpha=0.08, color='steelblue')
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel('Taux Faux Positifs (FPR)', fontweight='bold')
    plt.ylabel('Taux Vrais Positifs (TPR)', fontweight='bold')
    plt.title('Courbe ROC', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Sauvegardée → {save_path}")
    plt.show()
    return roc_auc


def plot_precision_recall(y_true, y_proba,
                          save_path: str | None = None) -> None:
    """Trace la courbe Précision-Rappel."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.fill_between(recall, precision, alpha=0.08, color='darkorange')
    plt.xlabel('Rappel (Recall)', fontweight='bold')
    plt.ylabel('Précision', fontweight='bold')
    plt.title('Courbe Précision-Rappel', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Sauvegardée → {save_path}")
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 20,
                             save_path: str | None = None) -> pd.Series:
    """
    Trace le bar chart des importances de features et retourne la Series triée.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, top_n * 0.4 + 1))
    colors = ['#d73027' if i >= len(importances) - 5 else '#4575b4'
              for i in range(len(importances))]
    importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_xlabel('Importance (Gini)', fontweight='bold')
    ax.set_title(f'Top {top_n} features les plus importantes',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Sauvegardée → {save_path}")
    plt.show()
    return importances.sort_values(ascending=False)


def plot_churn_distribution(y: pd.Series,
                             save_path: str | None = None) -> None:
    """Pie chart + bar chart de la distribution Churn / Fidèle."""
    counts = y.value_counts()
    labels = ['Fidèle (0)', 'Churn (1)']
    colors = ['#4575b4', '#d73027']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor='white', width=0.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')
    axes[0].set_title('Distribution absolue', fontweight='bold')
    axes[0].set_ylabel('Nombre de clients')
    axes[0].grid(axis='y', alpha=0.3)

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Distribution relative', fontweight='bold')

    plt.suptitle('Distribution de la variable Churn', fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Sauvegardée → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES DIVERS
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    """Crée les répertoires s'ils n'existent pas."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Sauvegarde une figure matplotlib et affiche le chemin."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  Figure sauvegardée → {path}")
if __name__ == "__main__":
    print("Test utils OK")