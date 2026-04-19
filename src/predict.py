import os
import joblib
import warnings
import numpy as np
import pandas as pd

from preprocessing import transform_for_inference

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH        = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH       = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH     = os.path.join(MODELS_DIR, "features.pkl")
CM_PATH           = os.path.join(MODELS_DIR, "country_means.pkl")
IMPUTE_PATH       = os.path.join(MODELS_DIR, "impute_values.pkl")
KMEANS_PATH       = os.path.join(MODELS_DIR, "kmeans.pkl")
PCA_CLUSTER_PATH  = os.path.join(MODELS_DIR, "pca_cluster.pkl")
REG_MODEL_PATH    = os.path.join(MODELS_DIR, "reg_model.pkl")
SCALER_REG_PATH   = os.path.join(MODELS_DIR, "scaler_reg.pkl")
REG_FEATURES_PATH = os.path.join(MODELS_DIR, "reg_features.pkl")

_model         = None
_scaler        = None
_feature_names = None
_country_means = None
_impute_values = None
_kmeans        = None
_pca_cluster   = None
_reg_model     = None
_scaler_reg    = None
_reg_features  = None


def load_artifacts():
    global _model, _scaler, _feature_names, _country_means, _impute_values
    global _kmeans, _pca_cluster, _reg_model, _scaler_reg, _reg_features

    if _model is not None:
        return

    required = {
        "model.pkl"        : MODEL_PATH,
        "scaler.pkl"       : SCALER_PATH,
        "features.pkl"     : FEATURES_PATH,
        "country_means.pkl": CM_PATH,
        "impute_values.pkl": IMPUTE_PATH,
        "kmeans.pkl"       : KMEANS_PATH,
        "pca_cluster.pkl"  : PCA_CLUSTER_PATH,
        "reg_model.pkl"    : REG_MODEL_PATH,
        "scaler_reg.pkl"   : SCALER_REG_PATH,
        "reg_features.pkl" : REG_FEATURES_PATH,
    }

    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\nArtefact manquant : {name}\n"
                f"Exécute d'abord :\n"
                f"  python src/preprocessing.py\n"
                f"  python src/train_model.py"
            )

    _model         = joblib.load(MODEL_PATH)
    _scaler        = joblib.load(SCALER_PATH)
    _feature_names = joblib.load(FEATURES_PATH)
    _country_means = joblib.load(CM_PATH)
    _impute_values = joblib.load(IMPUTE_PATH)
    _kmeans        = joblib.load(KMEANS_PATH)
    _pca_cluster   = joblib.load(PCA_CLUSTER_PATH)
    _reg_model     = joblib.load(REG_MODEL_PATH)
    _scaler_reg    = joblib.load(SCALER_REG_PATH)
    _reg_features  = joblib.load(REG_FEATURES_PATH)

    print(f"[load] Artefacts charges — "
          f"{len(_feature_names)} features clf | {len(_reg_features)} features reg")


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


def predict(df_raw: pd.DataFrame) -> pd.DataFrame:
    
    load_artifacts()

    X_clf = transform_for_inference(
        df_raw, _country_means, _impute_values, _feature_names
    )

    X_clf_scaled = pd.DataFrame(
        _scaler.transform(X_clf),
        columns=_feature_names,
        index=X_clf.index
    )

    preds  = _model.predict(X_clf_scaled)
    probas = _model.predict_proba(X_clf_scaled)[:, 1]

   
    X_clf_aligned = X_clf_scaled.reindex(columns=_feature_names, fill_value=0)
    X_pca    = _pca_cluster.transform(X_clf_aligned)
    clusters = _kmeans.predict(X_pca)


    df_reg = df_raw.copy()

    if "SupportTicketsCount" in df_reg.columns:
        df_reg["SupportTicketsCount"] = df_reg["SupportTicketsCount"].replace([-1, 999], np.nan)
    if "SatisfactionScore" in df_reg.columns:
        df_reg["SatisfactionScore"] = df_reg["SatisfactionScore"].replace([-1, 99], np.nan)

    for col in _reg_features:
        if col not in df_reg.columns:
            df_reg[col] = _impute_values.get(col, 0.0)
        else:
            df_reg[col] = pd.to_numeric(df_reg[col], errors="coerce")
            df_reg[col] = df_reg[col].fillna(_impute_values.get(col, 0.0))

    X_reg         = df_reg[_reg_features].copy()
    X_reg_scaled  = _scaler_reg.transform(X_reg)
    monetary_pred = np.maximum(0, _reg_model.predict(X_reg_scaled))


    if "CustomerID" in df_raw.columns:
        results = df_raw[["CustomerID"]].copy().reset_index(drop=True)
    else:
        results = pd.DataFrame(index=range(len(df_raw)))

    results["churn_pred"]    = preds
    results["churn_proba"]   = np.round(probas, 4)
    results["risk_level"]    = [risk_label(p) for p in probas]
    results["cluster"]       = clusters
    results["cluster_label"] = [cluster_label(c) for c in clusters]
    results["monetary_pred"] = np.round(monetary_pred, 2)

    return results


if __name__ == "__main__":
    raw_path = os.path.join(BASE_DIR, "data", "raw", "data.csv")

    print("=" * 60)
    print("  PREDICT — Test sur 10 clients aleatoires")
    print("=" * 60)

    if not os.path.exists(raw_path):
        print(f"Fichier introuvable : {raw_path}")
        raise SystemExit(1)

    df_raw = pd.read_csv(raw_path)
    sample = df_raw.sample(10, random_state=99).reset_index(drop=True)

    results = predict(sample)

    print("\nResultats :")
    print(results.to_string(index=False))

    print(f"\n  Churners predits    : {results['churn_pred'].sum()} / {len(results)}")
    print(f"  Probabilite moyenne : {results['churn_proba'].mean():.2%}")
    print(f"  Niveaux de risque   : {results['risk_level'].value_counts().to_dict()}")
    print(f"  Clusters            : {results['cluster_label'].value_counts().to_dict()}")
    print(f"  Depense prevue moy. : £{results['monetary_pred'].mean():.2f}")