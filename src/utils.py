import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc



def plot_pca(X, save_path=None):
    pca = PCA()
    pca.fit(X)

    var_cum = pca.explained_variance_ratio_.cumsum()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(var_cum) + 1), var_cum, marker='o')
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquée cumulée")
    plt.title("ACP — Variance cumulée")
    plt.axhline(y=0.80, color='orange', linestyle='--', label='80%')
    plt.axhline(y=0.90, color='red',    linestyle='--', label='90%')
    plt.axhline(y=0.95, color='purple', linestyle='--', label='95%')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    print("80% variance :", (var_cum >= 0.80).argmax() + 1)
    print("90% variance :", (var_cum >= 0.90).argmax() + 1)
    print("95% variance :", (var_cum >= 0.95).argmax() + 1)

    plt.close()



def print_df_info(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Affiche les informations générales du DataFrame (shape, NaN, doublons, types)."""
    print(f"\n{'='*50}")
    print(f"  Info — {label}")
    print(f"{'='*50}")
    print(f"  Shape      : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"  NaN total  : {df.isnull().sum().sum()}")
    print(f"  Doublons   : {df.duplicated().sum()}")
    print("  Types      :")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"    {str(dtype):<12} -> {count} colonnes")



def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne un rapport des valeurs manquantes trié par pourcentage décroissant."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    report = pd.DataFrame({
        'count'  : missing,
        'percent': (missing / len(df) * 100).round(2)
    }).sort_values('percent', ascending=False)
    return report

def correlation_matrix(df: pd.DataFrame, target: str = 'Churn',
                       threshold: float = 0.8,
                       save_path: str = None) -> list:

    num_df = df.select_dtypes(include=np.number)
    corr   = num_df.corr()

    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                annot=False, linewidths=0.3, vmin=-1, vmax=1)
    plt.title("Matrice de corrélation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[corr]  Heatmap sauvegardée → {save_path}")

    plt.show()

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
    else:
        print(f"\n  Aucune paire avec |corr| ≥ {threshold}")

    return high_corr



def compute_vif(X: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print("[vif] statsmodels non installé. Lancer : pip install statsmodels")
        return pd.DataFrame()

    X_num = X.select_dtypes(include=np.number).dropna()

    vif_data = pd.DataFrame()
    vif_data["feature"] = X_num.columns
    vif_data["VIF"]     = [
        variance_inflation_factor(X_num.values, i)
        for i in range(X_num.shape[1])
    ]
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

    high_vif = vif_data[vif_data["VIF"] > threshold]
    if not high_vif.empty:
        print(f"\n  Features avec VIF > {threshold} (multicolinéarité sévère) :")
        for _, row in high_vif.iterrows():
            print(f"    {row['feature']:<35} VIF = {row['VIF']:.2f}")
    else:
        print(f"\n  Aucune feature avec VIF > {threshold}")

    return vif_data



def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fidèle (0)', 'Churn (1)'],
                yticklabels=['Fidèle (0)', 'Churn (1)'])
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.close()



def plot_roc_curve(y_true, y_proba, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='steelblue', label=f"AUC = {roc_auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbe ROC")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.close()


def plot_feature_importance(model, feature_names, save_path=None, top_n=15):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, 5))
    plt.barh(range(top_n), importances[indices][::-1], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} features les plus importantes")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.close()