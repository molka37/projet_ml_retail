import os
import joblib
import optuna
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import plot_pca, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_PCA_COMPONENTS = 10  
                    
LEAKAGE_COLS = [
    'Recency',
    'ChurnRiskCategory',
    'LoyaltyLevel',
    'SpendingCategory',
    'AccountStatus_Closed',
    'AccountStatus_Pending',
    'AccountStatus_Suspended',
    'CustomerType_Nouveau',
    'CustomerType_Occasionnel',
    'CustomerType_Perdu',
    'CustomerType_Régulier',
    'RFMSegment_Dormants',
    'RFMSegment_Fidèles',
    'RFMSegment_Potentiels',
    'PreferredMonth',
    'CustomerTenureDays',
    'FirstPurchaseDaysAgo',
    'UniqueDescriptions',
    'UniqueInvoices',
]
REG_FEATURES = [
    'Frequency',
    'TotalQuantity',
    'AvgDaysBetweenPurchases',
    'UniqueProducts',
    'TotalTransactions',
    'SupportTicketsCount',
    'SatisfactionScore',
    'Age'
]
REG_TARGET = 'MonetaryTotal'


def sep(title=""):
    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 60}")


def main():
    tt_dir      = os.path.join(BASE_DIR, "data", "train_test")
    reports_dir = os.path.join(BASE_DIR, "reports")
    models_dir  = os.path.join(BASE_DIR, "models")
    raw_path    = os.path.join(BASE_DIR, "data", "raw", "data.csv")

    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    sep("1. Chargement des fichiers train/test")

    X_train = pd.read_csv(os.path.join(tt_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(tt_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(tt_dir, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(tt_dir, "y_test.csv")).squeeze()

    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Churn train : {y_train.mean():.2%} positifs")
    print(f"  Churn test  : {y_test.mean():.2%} positifs")

    sep("2. Suppression des colonnes data leakage")

    non_num    = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    all_drop   = sorted(set(LEAKAGE_COLS + non_num))
    cols_dropped = [c for c in all_drop if c in X_train.columns]

    print(f"  Supprimées ({len(cols_dropped)}) :")
    for c in cols_dropped:
        print(f"    - {c}")

    X_train = X_train.drop(columns=cols_dropped, errors='ignore')
    X_test  = X_test.drop(columns=cols_dropped,  errors='ignore')
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"\n  Features restantes : {X_train.shape[1]}")


    sep("3. Normalisation StandardScaler")

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
    print(f"  Scaler sauvegardé → {scaler_path}")
    print(f"  Fitté sur {X_train_s.shape[1]} features")

    from utils import compute_vif

    sep("Analyse VIF — Multicolinéarité")

    vif_df = compute_vif(X_train)

    high_vif = vif_df[vif_df["VIF"] > 10]

    if not high_vif.empty:
        print("\n⚠ Features à forte multicolinéarité :")
        print(high_vif.head(10))
    else:
        print("\n✅ Pas de multicolinéarité sévère (VIF < 10)")

    sep("4. ACP — Analyse en Composantes Principales")

    pca_path = os.path.join(reports_dir, "pca_variance.png")
    plot_pca(X_train_s, save_path=pca_path)
    print(f"  Courbe ACP sauvegardée → {pca_path}")

    sep("5. Classification — Recherche Optuna (RandomForest)")

    def objective(trial):
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators', 50, 300),
            'max_depth'        : trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features'     : trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'class_weight'     : 'balanced',
            'random_state'     : 42,
            'n_jobs'           : -1,
        }
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X_train_s, y_train, cv=3, scoring='f1').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print(f"\n  Meilleurs paramètres : {study.best_params}")
    print(f"  Meilleur F1 CV-3     : {study.best_value:.4f}")

    sep("6. Entraînement du modèle final de classification")

    best_params = {
        **study.best_params,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs'      : -1
    }

    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train_s, y_train)
    print("  Modèle entraîné ✅")

    sep("7. Évaluation finale sur X_test")

    y_pred  = best_model.predict(X_test_s)
    y_proba = best_model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Fidèle (0)', 'Churn (1)'])}")

    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(reports_dir, "confusion_matrix.png")
    )
    plot_roc_curve(
        y_test, y_proba,
        save_path=os.path.join(reports_dir, "roc_curve.png")
    )
    plot_feature_importance(
        best_model,
        list(X_train_s.columns),
        save_path=os.path.join(reports_dir, "feature_importance_plot.png"),
        top_n=15
    )
    print("  Graphiques sauvegardés : confusion_matrix.png, roc_curve.png, feature_importance_plot.png")

    sep("8. Validation croisée finale (5 folds)")

    cv_scores = cross_val_score(best_model, X_train_s, y_train, cv=5, scoring='f1')
    print(f"  F1 par fold : {[round(float(s), 4) for s in cv_scores]}")
    print(f"  F1 moyen    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    sep("9. Importance des features (top 15)")

    feat_imp = pd.Series(
        best_model.feature_importances_,
        index=X_train_s.columns
    ).sort_values(ascending=False)

    for feat, imp in feat_imp.head(15).items():
        print(f"  {feat:<35} {imp:.4f}  {'█' * int(imp * 200)}")

    feat_imp.to_csv(os.path.join(reports_dir, "feature_importance.csv"))

    sep(f"10. Clustering — PCA {N_PCA_COMPONENTS} composantes + KMeans")

    n_components = min(N_PCA_COMPONENTS, X_train_s.shape[1])
    pca_cluster  = PCA(n_components=n_components, random_state=42)
    X_train_pca  = pca_cluster.fit_transform(X_train_s)

    var_explained = pca_cluster.explained_variance_ratio_.sum()
    print(f"  Variance expliquée : {var_explained:.1%}")

    norms        = np.linalg.norm(X_train_pca, axis=1)
    Q1, Q3       = np.percentile(norms, 25), np.percentile(norms, 75)
    IQR          = Q3 - Q1
    mask         = norms <= (Q3 + 1.5 * IQR)  # CORRECTION : 3→1.5
    n_outliers   = (~mask).sum()
    print(f"  Outliers extrêmes retirés avant clustering : {n_outliers} clients")

    X_train_pca_clean = X_train_pca[mask]

    inertias = []
    k_range  = range(2, 9)
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
    elbow_path = os.path.join(reports_dir, "kmeans_elbow.png")
    plt.savefig(elbow_path, dpi=150)
    plt.close()
    print(f"  Courbe du coude sauvegardée → {elbow_path}")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train_pca_clean)   # fit sur données propres
    clusters = kmeans.predict(X_train_pca)  # predict sur tous les points

    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    print("  Effectif par cluster :")
    for cid, cnt in cluster_counts.items():
        print(f"    Cluster {cid} : {cnt:4d} clients ({cnt / len(clusters):.1%})")
    X_train_cluster = X_train_s.copy()
    X_train_cluster['cluster'] = clusters
    X_train_cluster['Churn']   = y_train.values

    cluster_summary = X_train_cluster.groupby('cluster').agg(
        Churn_rate         = ('Churn',                  'mean'),
        Frequency_mean     = ('Frequency',               'mean'),
        MonetaryTotal_mean = ('MonetaryTotal',            'mean'),
        AvgDays_mean       = ('AvgDaysBetweenPurchases',  'mean'),
        n_clients          = ('Churn',                   'count')
    ).round(3)

    print("\n  Profil des clusters :")
    print(cluster_summary.to_string())
    cluster_summary.to_csv(os.path.join(reports_dir, "cluster_profiles.csv"))
    print(f"  cluster_profiles.csv → {reports_dir}")
    pca_2d       = PCA(n_components=2, random_state=42)
    X_train_2d   = pca_2d.fit_transform(X_train_s)

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
    clust_path = os.path.join(reports_dir, "kmeans_clusters.png")
    plt.savefig(clust_path, dpi=150)
    plt.close()
    print(f"  Clusters sauvegardés → {clust_path}")
    sep("11. Régression — prédiction MonetaryTotal")

    df_raw = pd.read_csv(raw_path)
    df_reg = df_raw.copy()

    if 'SatisfactionScore' in df_reg.columns:
        df_reg['SatisfactionScore'] = df_reg['SatisfactionScore'].replace([-1, 99], np.nan)
    if 'SupportTicketsCount' in df_reg.columns:
        df_reg['SupportTicketsCount'] = df_reg['SupportTicketsCount'].replace([-1, 999], np.nan)

    missing_reg = [c for c in REG_FEATURES if c not in df_reg.columns]
    if missing_reg:
        print(f"  ⚠ Features absentes : {missing_reg} → remplacées par médiane/0")

    for col in REG_FEATURES:
        if col not in df_reg.columns:
            df_reg[col] = 0.0
        df_reg[col] = pd.to_numeric(df_reg[col], errors='coerce')
        df_reg[col] = df_reg[col].fillna(df_reg[col].median())

    Q1_r, Q3_r = df_reg[REG_TARGET].quantile([0.01, 0.99])
    df_reg = df_reg[(df_reg[REG_TARGET] >= Q1_r) & (df_reg[REG_TARGET] <= Q3_r)]
    print(f"  Outliers MonetaryTotal retirés — {len(df_reg)} lignes conservées")

    X_reg = df_reg[REG_FEATURES]
    y_reg = df_reg[REG_TARGET]

    X_reg_tr, X_reg_te, y_reg_tr, y_reg_te = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    scaler_reg   = StandardScaler()
    X_reg_tr_s   = scaler_reg.fit_transform(X_reg_tr)
    X_reg_te_s   = scaler_reg.transform(X_reg_te)

    reg_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg_model.fit(X_reg_tr_s, y_reg_tr)

    y_reg_pred = reg_model.predict(X_reg_te_s)
    mae  = mean_absolute_error(y_reg_te, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_te, y_reg_pred))
    r2   = r2_score(y_reg_te, y_reg_pred)

# Calcul pourcentages
    mean_monetary = y_reg_te.mean()
    mae_pct  = (mae  / mean_monetary) * 100
    rmse_pct = (rmse / mean_monetary) * 100

    print(f"  MAE  : {mae:.2f} £  ({mae_pct:.1f}% de la dépense moyenne)")
    print(f"  RMSE : {rmse:.2f} £  ({rmse_pct:.1f}% de la dépense moyenne)")
    print(f"  R²   : {r2:.4f}")
    print(f"  Dépense moyenne test : {mean_monetary:.2f} £")

    sep("12. Sauvegarde des artefacts")

    artifacts = {
        "model.pkl"       : best_model,
        "features.pkl"    : X_train_s.columns.tolist(),
        "kmeans.pkl"      : kmeans,
        "pca_cluster.pkl" : pca_cluster,
        "reg_model.pkl"   : reg_model,
        "scaler_reg.pkl"  : scaler_reg,
        "reg_features.pkl": REG_FEATURES,
    }

    for name, obj in artifacts.items():
        path = os.path.join(models_dir, name)
        joblib.dump(obj, path)
        print(f"  {name:<25} → {path}")

    print(f"  scaler.pkl               → {scaler_path} (déjà sauvegardé)")


    sep("RÉSUMÉ FINAL")
    print(f"  Classification Accuracy : {acc:.4f}")
    print(f"  Classification AUC-ROC  : {auc:.4f}")
    print(f"  Classification F1 CV-5  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Clustering              : 3 segments identifiés")
    print(f"  Régression MAE          : {mae:.2f} £  ({mae_pct:.1f}%)")
    print(f"  Régression RMSE         : {rmse:.2f} £  ({rmse_pct:.1f}%)")
    print(f"  Régression R²           : {r2:.4f}")


if __name__ == "__main__":
    main()