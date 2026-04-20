import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    plot_pca,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
)
from sklearn.ensemble import RandomForestClassifier
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
    r2_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PCA_COMPONENTS = 19
REG_TARGET = "MonetaryTotal"

LEAKAGE_COLS = [
    "Recency",
    "ChurnRiskCategory",
    "LoyaltyLevel",
    "SpendingCategory",
    "AccountStatus_Closed",
    "AccountStatus_Pending",
    "AccountStatus_Suspended",
    "CustomerType_Nouveau",
    "CustomerType_Occasionnel",
    "CustomerType_Perdu",
    "CustomerType_Régulier",
    "RFMSegment_Dormants",
    "RFMSegment_Fidèles",
    "RFMSegment_Potentiels",
    "PreferredMonth",
    "CustomerTenureDays",
    "FirstPurchaseDaysAgo",
    "UniqueDescriptions",
    "UniqueInvoices",
]

HIGH_VIF_DROP = [
    "UniqueInvoices",
    "UniqueDescriptions",
    "CustomerTenureDays",
    "FirstPurchaseDaysAgo",
    "Recency",
]


def sep(title=""):
    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 60}")


def find_best_threshold(y_true, y_proba):
    best_thr = 0.50
    best_f1 = -1
    best_metrics = None

    for thr in np.arange(0.30, 0.71, 0.02):
        y_pred_thr = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred_thr)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_metrics = {
                "f1": f1,
                "precision": precision_score(y_true, y_pred_thr, zero_division=0),
                "recall": recall_score(y_true, y_pred_thr, zero_division=0),
            }

    return best_thr, best_metrics


def add_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    base_cols = [
        "Frequency",
        "TotalQuantity",
        "UniqueProducts",
        "AvgDaysBetweenPurchases",
        "TotalTransactions",
        "SupportTicketsCount",
        "SatisfactionScore",
        "Age",
        "AvgQuantityPerTransaction",
        "NegativeQuantityCount",
        "CancelledTransactions",
    ]

    for col in base_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "SatisfactionScore" in df.columns:
        df["SatisfactionScore"] = df["SatisfactionScore"].replace([-1, 99], np.nan)

    if "SupportTicketsCount" in df.columns:
        df["SupportTicketsCount"] = df["SupportTicketsCount"].replace([-1, 999], np.nan)

    df["QtyPerProduct"] = df["TotalQuantity"] / (df["UniqueProducts"] + 1)
    df["TransactionsPerProduct"] = df["TotalTransactions"] / (df["UniqueProducts"] + 1)
    df["SpendPerTransaction"] = df["TotalQuantity"] / (df["TotalTransactions"] + 1)

    df["EngagementScore"] = (
        df["Frequency"] * df["TotalTransactions"] / (df["AvgDaysBetweenPurchases"] + 1)
    )
    df["PurchaseIntensity"] = (
        df["Frequency"] / (df["AvgDaysBetweenPurchases"] + 1)
    )
    df["CustomerValueScore"] = (
        df["Frequency"] * df["TotalQuantity"] * df["TotalTransactions"]
    ) / (df["AvgDaysBetweenPurchases"] + 1)
    df["ClientActivityScore"] = (
        df["Frequency"] * df["TotalQuantity"]
    ) / (df["AvgDaysBetweenPurchases"] + 1)
    df["StabilityScore"] = (
        df["Frequency"] / (df["AvgDaysBetweenPurchases"] + 1)
    )
    df["ConsistencyScore"] = (
        df["Frequency"] / (df["AvgDaysBetweenPurchases"] + 1)
    ) * df["TotalTransactions"]

    df["MonetaryPerProduct"] = df["TotalQuantity"] / (df["UniqueProducts"] + 1)
    df["MonetaryPerDay"] = df["TotalQuantity"] / (df["AvgDaysBetweenPurchases"] + 1)
    df["TransactionsFrequencyScore"] = df["TotalTransactions"] * df["Frequency"]
    df["CustomerEngagement"] = (
        df["Frequency"] * df["TotalTransactions"] * df["SatisfactionScore"]
    ) / (df["AvgDaysBetweenPurchases"] + 1)

    df["LogFrequency"] = np.log1p(df["Frequency"].clip(lower=0))
    df["LogTotalQuantity"] = np.log1p(df["TotalQuantity"].clip(lower=0))
    df["FreqXRecencyProxy"] = df["Frequency"] / (df["AvgDaysBetweenPurchases"] + 1)

    return df


def main():
    tt_dir = os.path.join(BASE_DIR, "data", "train_test")
    reports_dir = os.path.join(BASE_DIR, "reports")
    models_dir = os.path.join(BASE_DIR, "models")
    raw_path = os.path.join(BASE_DIR, "data", "raw", "data.csv")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    sep("1. Chargement des fichiers train/test")

    X_train = pd.read_csv(os.path.join(tt_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(tt_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(tt_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(tt_dir, "y_test.csv")).squeeze()

    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Churn train : {y_train.mean():.2%} positifs")
    print(f"  Churn test  : {y_test.mean():.2%} positifs")

    sep("2. Suppression des colonnes leakage + redondantes")

    non_num = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    all_drop = sorted(set(LEAKAGE_COLS + HIGH_VIF_DROP + non_num))
    cols_dropped = [c for c in all_drop if c in X_train.columns]

    print(f"  Supprimées ({len(cols_dropped)}) :")
    for c in cols_dropped:
        print(f"    - {c}")

    X_train = X_train.drop(columns=cols_dropped, errors="ignore")
    X_test = X_test.drop(columns=cols_dropped, errors="ignore")
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"\n  Features restantes : {X_train.shape[1]}")

    sep("3. Normalisation StandardScaler")

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler sauvegardé → {scaler_path}")
    print(f"  Fitté sur {X_train_s.shape[1]} features")

    sep("4. ACP — Analyse en Composantes Principales")

    pca_path = os.path.join(reports_dir, "pca_variance.png")
    plot_pca(X_train_s, save_path=pca_path)
    print(f"  Courbe ACP sauvegardée → {pca_path}")

    sep("5. Classification — Paramètres RandomForest")

    clf_params = {
        "n_estimators": 350,
        "max_depth": 14,
        "min_samples_split": 8,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "class_weight": {0: 1, 1: 2},
        "random_state": 42,
        "n_jobs": -1,
    }

    print(f"\n  Paramètres : {clf_params}")

    sep("6. Entraînement classification")

    clf = RandomForestClassifier(**clf_params)
    clf.fit(X_train_s, y_train)
    print("  Modèle entraîné ✅")

    sep("7. Évaluation classification")

    y_proba = clf.predict_proba(X_test_s)[:, 1]
    y_pred_05 = (y_proba >= 0.50).astype(int)

    acc_05 = accuracy_score(y_test, y_pred_05)
    auc = roc_auc_score(y_test, y_proba)

    print("  Avec seuil = 0.50")
    print(f"  Accuracy : {acc_05:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(classification_report(y_test, y_pred_05, target_names=["Fidèle (0)", "Churn (1)"]))

    best_thr, best_metrics = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_thr).astype(int)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Meilleur seuil F1 : {best_thr:.2f}")
    print(f"  Precision : {best_metrics['precision']:.4f}")
    print(f"  Recall    : {best_metrics['recall']:.4f}")
    print(f"  F1        : {best_metrics['f1']:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Fidèle (0)', 'Churn (1)'])}")

    plot_confusion_matrix(
        y_test,
        y_pred,
        save_path=os.path.join(reports_dir, "confusion_matrix.png"),
    )
    plot_roc_curve(
        y_test,
        y_proba,
        save_path=os.path.join(reports_dir, "roc_curve.png"),
    )
    plot_feature_importance(
        clf,
        list(X_train_s.columns),
        save_path=os.path.join(reports_dir, "feature_importance_plot.png"),
        top_n=15,
    )
    print("  Graphiques sauvegardés : confusion_matrix.png, roc_curve.png, feature_importance_plot.png")

    sep("8. Validation croisée classification")

    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=5, scoring="f1")
    print(f"  F1 par fold : {[round(float(s), 4) for s in cv_scores]}")
    print(f"  F1 moyen    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    sep("9. Clustering — PCA + KMeans")

    n_components = min(N_PCA_COMPONENTS, X_train_s.shape[1])
    pca_cluster = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca_cluster.fit_transform(X_train_s)

    var_explained = pca_cluster.explained_variance_ratio_.sum()
    print(f"  Variance expliquée : {var_explained:.1%}")

    norms = np.linalg.norm(X_train_pca, axis=1)
    q1, q3 = np.percentile(norms, 25), np.percentile(norms, 75)
    iqr = q3 - q1
    mask = norms <= (q3 + 1.5 * iqr)
    n_outliers = (~mask).sum()
    print(f"  Outliers extrêmes retirés avant clustering : {n_outliers} clients")

    X_train_pca_clean = X_train_pca[mask]

    k_range = range(2, 9)
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_train_pca_clean)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(list(k_range), inertias, marker="o")
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Inertie")
    plt.title("Méthode du coude — KMeans")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    elbow_path = os.path.join(reports_dir, "kmeans_elbow.png")
    plt.savefig(elbow_path, dpi=150)
    plt.close()
    print(f"  Courbe du coude sauvegardée → {elbow_path}")

    n_clusters = 3
    print(f"\n  Clustering avec k={n_clusters}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_train_pca_clean)
    clusters = kmeans.predict(X_train_pca)

    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    print("  Effectif par cluster :")
    for cid, cnt in cluster_counts.items():
        print(f"    Cluster {cid} : {cnt:4d} clients ({cnt / len(clusters):.1%})")

    sep("10. Régression — prédiction MonetaryTotal")

    df_raw = pd.read_csv(raw_path)
    df_reg = df_raw.copy()

    if REG_TARGET not in df_reg.columns:
        raise ValueError(f"Colonne cible manquante : {REG_TARGET}")

    df_reg[REG_TARGET] = pd.to_numeric(df_reg[REG_TARGET], errors="coerce")
    df_reg = df_reg.dropna(subset=[REG_TARGET]).copy()
    df_reg = df_reg[df_reg[REG_TARGET] > 0].copy()

    upper_cap = df_reg[REG_TARGET].quantile(0.985)
    df_reg = df_reg[df_reg[REG_TARGET] <= upper_cap].copy()

    print(f"  Lignes conservées : {len(df_reg)}")

    df_reg = add_regression_features(df_reg)

    reg_features = [
        "Frequency",
        "TotalQuantity",
        "UniqueProducts",
        "AvgDaysBetweenPurchases",
        "TotalTransactions",
        "SupportTicketsCount",
        "SatisfactionScore",
        "Age",
        "AvgQuantityPerTransaction",
        "NegativeQuantityCount",
        "CancelledTransactions",
        "QtyPerProduct",
        "TransactionsPerProduct",
        "SpendPerTransaction",
        "EngagementScore",
        "PurchaseIntensity",
        "CustomerValueScore",
        "ClientActivityScore",
        "StabilityScore",
        "ConsistencyScore",
        "MonetaryPerProduct",
        "MonetaryPerDay",
        "TransactionsFrequencyScore",
        "CustomerEngagement",
        "LogFrequency",
        "LogTotalQuantity",
        "FreqXRecencyProxy",
    ]

    missing_reg = [c for c in reg_features if c not in df_reg.columns]
    if missing_reg:
        raise ValueError(f"Colonnes manquantes pour la régression : {missing_reg}")

    X_reg = df_reg[reg_features].copy()
    y_reg_raw = df_reg[REG_TARGET].copy()

    X_train_full, X_test_reg, y_train_full_raw, y_test_reg_raw = train_test_split(
        X_reg,
        y_reg_raw,
        test_size=0.2,
        random_state=42,
    )

    X_train_reg, X_val_reg, y_train_reg_raw, y_val_reg_raw = train_test_split(
        X_train_full,
        y_train_full_raw,
        test_size=0.15,
        random_state=42,
    )

    reg_impute_values = {}
    for col in reg_features:
        median_val = X_train_reg[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        reg_impute_values[col] = float(median_val)

    X_train_reg = X_train_reg.fillna(reg_impute_values)
    X_val_reg = X_val_reg.fillna(reg_impute_values)
    X_test_reg = X_test_reg.fillna(reg_impute_values)

    q_low, q_high = y_train_reg_raw.quantile([0.005, 0.995])
    mask_out = (y_train_reg_raw >= q_low) & (y_train_reg_raw <= q_high)

    X_train_reg = X_train_reg.loc[mask_out].copy()
    y_train_reg_raw = y_train_reg_raw.loc[mask_out].copy()

    print(f"  Train après filtre outliers : {len(X_train_reg)}")

    split_value = y_train_reg_raw.quantile(0.80)
    y_regime_train = (y_train_reg_raw > split_value).astype(int)
    y_regime_val = (y_val_reg_raw > split_value).astype(int)
    y_regime_test = (y_test_reg_raw > split_value).astype(int)

    regime_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    regime_clf.fit(X_train_reg, y_regime_train)

    y_train_log = np.log1p(y_train_reg_raw)
    y_val_log = np.log1p(y_val_reg_raw)
    y_test_log = np.log1p(y_test_reg_raw)

    small_mask_train = y_regime_train == 0
    large_mask_train = y_regime_train == 1

    X_train_small = X_train_reg.loc[small_mask_train]
    y_train_small_log = y_train_log.loc[small_mask_train]
    y_train_small_raw = y_train_reg_raw.loc[small_mask_train]

    X_train_large = X_train_reg.loc[large_mask_train]
    y_train_large_log = y_train_log.loc[large_mask_train]
    y_train_large_raw = y_train_reg_raw.loc[large_mask_train]

    small_mask_val = y_regime_val == 0
    large_mask_val = y_regime_val == 1

    X_val_small = X_val_reg.loc[small_mask_val]
    y_val_small_log = y_val_log.loc[small_mask_val]

    X_val_large = X_val_reg.loc[large_mask_val]
    y_val_large_log = y_val_log.loc[large_mask_val]

    sample_weights_small = 1 + (y_train_small_raw / np.mean(y_train_small_raw))
    sample_weights_large = 1 + (y_train_large_raw / np.mean(y_train_large_raw))

    xgb_small = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=2500,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.4,
        reg_lambda=5.0,
        gamma=0.1,
        early_stopping_rounds=120,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    xgb_large = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=3000,
        max_depth=10,
        learning_rate=0.008,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=1,
        reg_alpha=0.3,
        reg_lambda=4.0,
        gamma=0.05,
        early_stopping_rounds=150,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    eval_small = [(X_val_reg, y_val_log)] if len(X_val_small) == 0 else [(X_val_small, y_val_small_log)]
    eval_large = [(X_val_reg, y_val_log)] if len(X_val_large) == 0 else [(X_val_large, y_val_large_log)]

    xgb_small.fit(
        X_train_small,
        y_train_small_log,
        sample_weight=sample_weights_small,
        eval_set=eval_small,
        verbose=False,
    )

    xgb_large.fit(
        X_train_large,
        y_train_large_log,
        sample_weight=sample_weights_large,
        eval_set=eval_large,
        verbose=False,
    )

    regime_pred_test = regime_clf.predict(X_test_reg)

    pred_small_log = xgb_small.predict(X_test_reg)
    pred_large_log = xgb_large.predict(X_test_reg)

    final_pred_log = np.where(regime_pred_test == 1, pred_large_log, pred_small_log)
    y_pred = np.expm1(final_pred_log)
    y_test_real = y_test_reg_raw.values

    upper_pred_cap = np.percentile(y_train_reg_raw, 99.2)
    y_pred = np.clip(y_pred, 0, upper_pred_cap)

    mae = mean_absolute_error(y_test_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
    r2 = r2_score(y_test_real, y_pred)

    mean_monetary = y_test_real.mean()
    mae_pct = (mae / mean_monetary) * 100
    rmse_pct = (rmse / mean_monetary) * 100

    print(f"  Split value régime : {split_value:.2f}")
    print(f"  MAE  : {mae:.2f} £  ({mae_pct:.1f}%)")
    print(f"  RMSE : {rmse:.2f} £  ({rmse_pct:.1f}%)")
    print(f"  R²   : {r2:.4f}")
    print(f"  Dépense moyenne test : {mean_monetary:.2f} £")

    sep("11. Sauvegarde des artefacts")

    artifacts = {
        "model.pkl": clf,
        "features.pkl": X_train_s.columns.tolist(),
        "kmeans.pkl": kmeans,
        "pca_cluster.pkl": pca_cluster,
        "best_threshold.pkl": best_thr,
        "reg_features.pkl": reg_features,
        "reg_impute_values.pkl": reg_impute_values,
        "xgb_small.pkl": xgb_small,
        "xgb_large.pkl": xgb_large,
        "regime_clf.pkl": regime_clf,
        "reg_split_value.pkl": split_value,
        "reg_upper_pred_cap.pkl": float(upper_pred_cap),
    }

    for name, obj in artifacts.items():
        path = os.path.join(models_dir, name)
        joblib.dump(obj, path)
        print(f"  {name:<25} → {path}")

    print(f"  scaler.pkl                → {scaler_path} (déjà sauvegardé)")

    sep("RÉSUMÉ FINAL")
    print(f"  Classification Accuracy : {acc:.4f}")
    print(f"  Classification AUC-ROC  : {auc:.4f}")
    print(f"  Classification F1 CV-5  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Meilleur seuil          : {best_thr:.2f}")
    print("  Clustering              : 3 segments identifiés")
    print(f"  Régression MAE          : {mae:.2f} £  ({mae_pct:.1f}%)")
    print(f"  Régression RMSE         : {rmse:.2f} £  ({rmse_pct:.1f}%)")
    print(f"  Régression R²           : {r2:.4f}")


if __name__ == "__main__":
    main()