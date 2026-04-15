import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

def plot_pca(X, save_path=None):
    pca = PCA()
    pca.fit(X)

    var_cum = pca.explained_variance_ratio_.cumsum()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(var_cum) + 1), var_cum, marker='o')
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquée cumulée")
    plt.title("ACP")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    print("80% variance :", (var_cum >= 0.80).argmax() + 1)
    print("90% variance :", (var_cum >= 0.90).argmax() + 1)
    print("95% variance :", (var_cum >= 0.95).argmax() + 1)

    plt.close()

def print_df_info(df: pd.DataFrame, label: str = "DataFrame") -> None:
    print(f"\nInfo - {label}")
    print(f"Shape      : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"NaN total  : {df.isnull().sum().sum()}")
    print(f"Doublons   : {df.duplicated().sum()}")
    print("Types      :")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {str(dtype):<12} -> {count} colonnes")

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    report = pd.DataFrame({
        'count'  : missing,
        'percent': (missing / len(df) * 100).round(2)
    }).sort_values('percent', ascending=False)
    return report



def correlation_matrix(df: pd.DataFrame, target: str = 'Churn',
                       threshold: float = 0.8) -> list:
    
    num_df = df.select_dtypes(include=np.number)
    corr   = num_df.corr()

    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                annot=False, linewidths=0.3, vmin=-1, vmax=1)
    plt.title("Matrice de corrélation", fontsize=14, fontweight='bold')
    plt.tight_layout()
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

    return high_corr



