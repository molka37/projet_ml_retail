import os
import joblib
import warnings
import numpy as np
import pandas as pd

from preprocessing import transform_for_inference

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.pkl")
CM_PATH = os.path.join(MODELS_DIR, "country_means.pkl")
IMPUTE_PATH = os.path.join(MODELS_DIR, "impute_values.pkl")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans.pkl")
PCA_CLUSTER_PATH = os.path.join(MODELS_DIR, "pca_cluster.pkl")
REG_FEATURES_PATH = os.path.join(MODELS_DIR, "reg_features.pkl")
REG_IMPUTE_PATH = os.path.join(MODELS_DIR, "reg_impute_values.pkl")
KNN_AGE_PATH = os.path.join(MODELS_DIR, "knn_age.pkl")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "best_threshold.pkl")

XGB_SMALL_PATH = os.path.join(MODELS_DIR, "xgb_small.pkl")
XGB_LARGE_PATH = os.path.join(MODELS_DIR, "xgb_large.pkl")
REGIME_CLF_PATH = os.path.join(MODELS_DIR, "regime_clf.pkl")
REG_SPLIT_VALUE_PATH = os.path.join(MODELS_DIR, "reg_split_value.pkl")
REG_UPPER_PRED_CAP_PATH = os.path.join(MODELS_DIR, "reg_upper_pred_cap.pkl")

_model = None
_scaler = None
_feature_names = None
_country_means = None
_impute_values = None
_kmeans = None
_pca_cluster = None
_reg_features = None
_reg_impute_values = None
_knn_age = None
_best_threshold = 0.5

_xgb_small = None
_xgb_large = None
_regime_clf = None
_reg_split_value = None
_reg_upper_pred_cap = None


def load_artifacts():
    global _model, _scaler, _feature_names, _country_means, _impute_values
    global _kmeans, _pca_cluster, _reg_features, _reg_impute_values
    global _knn_age, _best_threshold
    global _xgb_small, _xgb_large, _regime_clf, _reg_split_value, _reg_upper_pred_cap

    if _model is not None:
        return

    required = {
        "model.pkl": MODEL_PATH,
        "scaler.pkl": SCALER_PATH,
        "features.pkl": FEATURES_PATH,
        "country_means.pkl": CM_PATH,
        "impute_values.pkl": IMPUTE_PATH,
        "kmeans.pkl": KMEANS_PATH,
        "pca_cluster.pkl": PCA_CLUSTER_PATH,
        "reg_features.pkl": REG_FEATURES_PATH,
        "reg_impute_values.pkl": REG_IMPUTE_PATH,
        "knn_age.pkl": KNN_AGE_PATH,
        "xgb_small.pkl": XGB_SMALL_PATH,
        "xgb_large.pkl": XGB_LARGE_PATH,
        "regime_clf.pkl": REGIME_CLF_PATH,
        "reg_split_value.pkl": REG_SPLIT_VALUE_PATH,
        "reg_upper_pred_cap.pkl": REG_UPPER_PRED_CAP_PATH,
    }

    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\nArtefact manquant : {name}\n"
                f"Exécute d'abord :\n"
                f"  python src/preprocessing.py\n"
                f"  python src/train_model.py"
            )

    _model = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    _feature_names = joblib.load(FEATURES_PATH)
    _country_means = joblib.load(CM_PATH)
    _impute_values = joblib.load(IMPUTE_PATH)
    _kmeans = joblib.load(KMEANS_PATH)
    _pca_cluster = joblib.load(PCA_CLUSTER_PATH)
    _reg_features = joblib.load(REG_FEATURES_PATH)
    _reg_impute_values = joblib.load(REG_IMPUTE_PATH)
    _knn_age = joblib.load(KNN_AGE_PATH)

    _xgb_small = joblib.load(XGB_SMALL_PATH)
    _xgb_large = joblib.load(XGB_LARGE_PATH)
    _regime_clf = joblib.load(REGIME_CLF_PATH)
    _reg_split_value = joblib.load(REG_SPLIT_VALUE_PATH)
    _reg_upper_pred_cap = joblib.load(REG_UPPER_PRED_CAP_PATH)

    if os.path.exists(THRESHOLD_PATH):
        _best_threshold = joblib.load(THRESHOLD_PATH)

    print(f"[load] Artefacts chargés — {len(_feature_names)} features clf | {len(_reg_features)} features reg")
    print(f"[load] Seuil de classification utilisé : {_best_threshold:.2f}")
    print(f"[load] Split value régime régression : {_reg_split_value:.2f}")


def risk_label(p: float) -> str:
    if p < 0.20:
        return "Faible"
    elif p < 0.40:
        return "Moyen"
    elif p < 0.70:
        return "Eleve"
    else:
        return "Critique"


def cluster_label(c: int) -> str:
    return f"Cluster {c}"


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


def predict(df_raw: pd.DataFrame) -> pd.DataFrame:
    load_artifacts()

    X_clf = transform_for_inference(
        df_raw,
        _country_means,
        _impute_values,
        _feature_names,
        knn_age=_knn_age,
    )

    X_clf_scaled = pd.DataFrame(
        _scaler.transform(X_clf),
        columns=_feature_names,
        index=X_clf.index,
    )

    probas = _model.predict_proba(X_clf_scaled)[:, 1]
    preds = (probas >= _best_threshold).astype(int)

    X_pca = _pca_cluster.transform(X_clf_scaled)
    clusters = _kmeans.predict(X_pca)

    df_reg = add_regression_features(df_raw.copy())

    for col in _reg_features:
        if col not in df_reg.columns:
            df_reg[col] = _reg_impute_values.get(col, 0.0)
        else:
            df_reg[col] = pd.to_numeric(df_reg[col], errors="coerce")
            df_reg[col] = df_reg[col].fillna(_reg_impute_values.get(col, 0.0))

    X_reg = df_reg[_reg_features].copy()

    regime_pred = _regime_clf.predict(X_reg)

    pred_small_log = _xgb_small.predict(X_reg)
    pred_large_log = _xgb_large.predict(X_reg)

    final_pred_log = np.where(regime_pred == 1, pred_large_log, pred_small_log)
    monetary_pred = np.expm1(final_pred_log)
    monetary_pred = np.clip(monetary_pred, 0, _reg_upper_pred_cap)

    if "CustomerID" in df_raw.columns:
        results = df_raw[["CustomerID"]].copy().reset_index(drop=True)
    else:
        results = pd.DataFrame(index=range(len(df_raw)))

    results["churn_pred"] = preds
    results["churn_proba"] = np.round(probas, 4)
    results["risk_level"] = [risk_label(p) for p in probas]
    results["cluster"] = clusters
    results["cluster_label"] = [cluster_label(c) for c in clusters]
    results["regime_pred"] = regime_pred
    results["monetary_pred"] = np.round(monetary_pred, 2)

    return results


if __name__ == "__main__":
    raw_path = os.path.join(BASE_DIR, "data", "raw", "data.csv")

    print("=" * 60)
    print("  PREDICT — Test sur 10 clients aléatoires")
    print("=" * 60)

    if not os.path.exists(raw_path):
        print(f"Fichier introuvable : {raw_path}")
        raise SystemExit(1)

    df_raw = pd.read_csv(raw_path)
    sample = df_raw.sample(10, random_state=99).reset_index(drop=True)

    results = predict(sample)

    print("\nRésultats :")
    print(results.to_string(index=False))

    print(f"\n  Churners prédits    : {results['churn_pred'].sum()} / {len(results)}")
    print(f"  Probabilité moyenne : {results['churn_proba'].mean():.2%}")
    print(f"  Niveaux de risque   : {results['risk_level'].value_counts().to_dict()}")
    print(f"  Clusters            : {results['cluster_label'].value_counts().to_dict()}")
    print(f"  Régimes régression  : {results['regime_pred'].value_counts().to_dict()}")
    print(f"  Dépense prévue moy. : £{results['monetary_pred'].mean():.2f}")