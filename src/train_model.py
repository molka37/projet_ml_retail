import os
import joblib
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_pca

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
optuna.logging.set_verbosity(optuna.logging.WARNING)

LEAKAGE_COLS = [
    'Recency', 'MonetaryPerDay', 'TenureRatio',
    'ChurnRiskCategory', 'LoyaltyLevel', 'SpendingCategory',
    'AccountStatus_Closed', 'AccountStatus_Pending', 'AccountStatus_Suspended',
    'CustomerType_Nouveau', 'CustomerType_Occasionnel',
    'CustomerType_Perdu', 'CustomerType_Régulier',
    'RFMSegment_Dormants', 'RFMSegment_Fidèles', 'RFMSegment_Potentiels',
    'PreferredMonth', 'CustomerTenureDays', 'FirstPurchaseDaysAgo',
    'UniqueDescriptions', 'UniqueInvoices',
]

REG_FEATURES = [
    "Recency",
    "Frequency",
    "TotalQuantity",
    "AvgDaysBetweenPurchases",
    "UniqueProducts",
    "TotalTransactions",
    "SupportTicketsCount",
    "SatisfactionScore",
    "Age"
]
REG_TARGET = "MonetaryTotal"

N_PCA_COMPONENTS = 26


def main():
    print("\n1. Chargement des fichiers train/test")

    tt_dir = os.path.join(BASE_DIR, "data", "train_test")
    reports_dir = os.path.join(BASE_DIR, "reports")
    models_dir = os.path.join(BASE_DIR, "models")
    raw_path = os.path.join(BASE_DIR, "data", "raw", "data.csv")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    X_train = pd.read_csv(os.path.join(tt_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(tt_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(tt_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(tt_dir, "y_test.csv")).squeeze()

    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"Churn train : {y_train.mean():.2%} positifs")
    print(f"Churn test  : {y_test.mean():.2%} positifs")

    print("\n2. Suppression des colonnes data leakage")
    cols_dropped = [c for c in LEAKAGE_COLS if c in X_train.columns]
    print(f"Colonnes supprimées ({len(cols_dropped)}) :")
    for c in cols_dropped:
        print(f"  - {c}")

    X_train = X_train.drop(columns=cols_dropped, errors="ignore")
    X_test = X_test.drop(columns=cols_dropped, errors="ignore")
    print(f"Features restantes : {X_train.shape[1]}")

    print("\n3. Normalisation (StandardScaler) — Classification")
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler sauvegardé : {scaler_path}")

    print("\n4. ACP — Analyse de la variance")
    plot_pca(X_train_s, os.path.join(reports_dir, "pca_variance.png"))
    print("Courbe ACP sauvegardée dans reports/pca_variance.png")

    print("\n5. Classification — Recherche Optuna")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)
        score = cross_val_score(model, X_train_s, y_train, cv=3, scoring="f1").mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print(f"Meilleurs paramètres : {study.best_params}")
    print(f"Meilleur F1 CV-3 : {study.best_value:.4f}")

    print("\n6. Entraînement du modèle final de classification")
    best_params = {
        **study.best_params,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train_s, y_train)
    print("Modèle de classification entraîné")

    print("\n7. Évaluation finale sur X_test")
    y_pred = best_model.predict(X_test_s)
    y_proba = best_model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
    print(classification_report(y_test, y_pred, target_names=["Fidèle (0)", "Churn (1)"]))

    print("\n8. Validation croisée finale (5 folds)")
    cv_scores = cross_val_score(best_model, X_train_s, y_train, cv=5, scoring="f1")
    print(f"F1 par fold : {[round(float(s), 4) for s in cv_scores]}")
    print(f"F1 moyen    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\n9. Importance des features (top 15)")
    feat_imp = pd.Series(
        best_model.feature_importances_,
        index=X_train_s.columns
    ).sort_values(ascending=False)

    for feat, imp in feat_imp.head(15).items():
        print(f"  {feat:<35} {imp:.4f}")

    print(f"\n10. Clustering — PCA {N_PCA_COMPONENTS} composantes + KMeans")
    pca_cluster = PCA(n_components=min(N_PCA_COMPONENTS, X_train_s.shape[1]), random_state=42)
    X_train_pca = pca_cluster.fit_transform(X_train_s)

    var_explained = pca_cluster.explained_variance_ratio_.sum()
    print(f"  Variance expliquée : {var_explained:.1%}")

    norms = np.linalg.norm(X_train_pca, axis=1)
    Q1, Q3 = np.percentile(norms, 25), np.percentile(norms, 75)
    IQR = Q3 - Q1
    seuil = Q3 + 3 * IQR
    mask = norms <= seuil

    n_outliers = (~mask).sum()
    print(f"  Outliers extrêmes retirés avant clustering : {n_outliers} clients")
    X_train_pca_clean = X_train_pca[mask]

    inertias = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_train_pca_clean)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(list(k_range), inertias, marker='o')
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Inertie")
    plt.title("Méthode du coude — KMeans")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "kmeans_elbow.png"), dpi=150)
    plt.close()

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train_pca_clean)
    clusters = kmeans.predict(X_train_pca)

    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    print("  Effectif par cluster :")
    for cid, cnt in cluster_counts.items():
        pct = cnt / len(clusters) * 100
        print(f"    Cluster {cid} : {cnt:4d} clients ({pct:.1f}%)")

    pca_2d = PCA(n_components=2, random_state=42)
    X_train_2d = pca_2d.fit_transform(X_train_s)

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        X_train_2d[:, 0], X_train_2d[:, 1],
        c=clusters, cmap="tab10", alpha=0.6, s=15
    )
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clustering KMeans (visualisation PCA 2D)")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "kmeans_clusters.png"), dpi=150)
    plt.close()

    print("\n11. Régression — prédiction MonetaryTotal")
    df_raw_full = pd.read_csv(raw_path)

    valid_reg_features = [c for c in REG_FEATURES if c in df_raw_full.columns]
    if REG_TARGET not in df_raw_full.columns:
        raise ValueError(f"Colonne cible de régression absente : {REG_TARGET}")

    df_reg = df_raw_full[valid_reg_features + [REG_TARGET]].copy()

    for col in valid_reg_features:
        if pd.api.types.is_numeric_dtype(df_reg[col]):
            if col == "SupportTicketsCount":
                df_reg[col] = df_reg[col].replace([-1, 999], np.nan)
            elif col == "SatisfactionScore":
                df_reg[col] = df_reg[col].replace([-1, 99], np.nan)
            df_reg[col] = df_reg[col].fillna(df_reg[col].median())

    Q1 = df_reg[REG_TARGET].quantile(0.25)
    Q3 = df_reg[REG_TARGET].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR

    n_before = len(df_reg)
    df_reg = df_reg[(df_reg[REG_TARGET] >= lower) & (df_reg[REG_TARGET] <= upper)]
    n_after = len(df_reg)

    print(f"  Outliers MonetaryTotal supprimés : {n_before - n_after} lignes")
    print(f"  Intervalle conservé : [{lower:.0f}, {upper:.0f}] £")

    X_reg = df_reg[valid_reg_features]
    y_reg = df_reg[REG_TARGET]

    split_idx = int(len(df_reg) * 0.8)
    X_reg_train = X_reg.iloc[:split_idx]
    X_reg_test = X_reg.iloc[split_idx:]
    y_reg_train = y_reg.iloc[:split_idx]
    y_reg_test = y_reg.iloc[split_idx:]

    scaler_reg = StandardScaler()
    X_reg_train_s = scaler_reg.fit_transform(X_reg_train)
    X_reg_test_s = scaler_reg.transform(X_reg_test)

    reg_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg_model.fit(X_reg_train_s, y_reg_train)

    y_reg_pred = reg_model.predict(X_reg_test_s)

    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    r2 = r2_score(y_reg_test, y_reg_pred)

    print(f"  MAE  : {mae:.2f} £")
    print(f"  RMSE : {rmse:.2f} £")
    print(f"  R²   : {r2:.4f}")

    print("\n12. Sauvegarde de tous les artefacts")
    artifacts = {
        "model.pkl": best_model,
        "scaler.pkl": scaler,
        "features.pkl": X_train_s.columns.tolist(),
        "kmeans.pkl": kmeans,
        "pca_cluster.pkl": pca_cluster,
        "reg_model.pkl": reg_model,
        "scaler_reg.pkl": scaler_reg,
        "reg_features.pkl": valid_reg_features,
    }

    for filename, obj in artifacts.items():
        path = os.path.join(models_dir, filename)
        joblib.dump(obj, path)
        print(f"  {filename:<22} -> {path}")

    print("\nRésumé final")
    print("=" * 55)
    print(f"  Classification Accuracy : {acc:.4f}")
    print(f"  Classification AUC-ROC  : {auc:.4f}")
    print(f"  Classification F1 CV-5  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Régression MAE          : {mae:.2f} £")
    print(f"  Régression RMSE         : {rmse:.2f} £")
    print(f"  Régression R²           : {r2:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()