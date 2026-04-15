import os
import joblib
import warnings
import numpy as np
import pandas as pd

from preprocessing import transform_for_inference

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH         = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH        = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH      = os.path.join(MODELS_DIR, "features.pkl")
CM_PATH            = os.path.join(MODELS_DIR, "country_means.pkl")
IMPUTE_PATH        = os.path.join(MODELS_DIR, "impute_values.pkl")
KMEANS_PATH        = os.path.join(MODELS_DIR, "kmeans.pkl")
PCA_CLUSTER_PATH   = os.path.join(MODELS_DIR, "pca_cluster.pkl")   # ← nouveau
REG_MODEL_PATH     = os.path.join(MODELS_DIR, "reg_model.pkl")
SCALER_REG_PATH    = os.path.join(MODELS_DIR, "scaler_reg.pkl")
REG_FEATURES_PATH  = os.path.join(MODELS_DIR, "reg_features.pkl")

# Variables globales — chargées une seule fois (lazy loading)
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
    """Charge tous les artefacts depuis models/ au premier appel."""
    global _model, _scaler, _feature_names, _country_means, _impute_values
    global _kmeans, _pca_cluster, _reg_model, _scaler_reg, _reg_features

    if _model is not None:
        return  # Déjà chargés

    required = {
        "model.pkl":         MODEL_PATH,
        "scaler.pkl":        SCALER_PATH,
        "features.pkl":      FEATURES_PATH,
        "country_means.pkl": CM_PATH,
        "impute_values.pkl": IMPUTE_PATH,
        "kmeans.pkl":        KMEANS_PATH,
        "pca_cluster.pkl":   PCA_CLUSTER_PATH,
        "reg_model.pkl":     REG_MODEL_PATH,
        "scaler_reg.pkl":    SCALER_REG_PATH,
        "reg_features.pkl":  REG_FEATURES_PATH,
    }

    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artefact manquant : {name}\n"
                f"Lance d'abord : python src/preprocessing.py  puis  python src/train_model.py"
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

    print(f"[load] Artefacts chargés — {len(_feature_names)} features clf | "
          f"{len(_reg_features)} features reg | "
          f"PCA {_pca_cluster.n_components_}D → KMeans")


def risk_label(p: float) -> str:
    """Probabilité → niveau de risque churn."""
    if p < 0.20:
        return "Faible"
    elif p < 0.40:
        return "Moyen"
    elif p < 0.70:
        return "Élevé"
    else:
        return "Critique"


def cluster_label(c: int) -> str:
    """Numéro de cluster → nom métier (k=3)."""
    labels = {
        0: "Clients fidèles",
        1: "Clients à risque",
        2: "Clients dormants",
    }
    return labels.get(c, f"Cluster {c}")


def predict(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne pour chaque client :
      - churn_pred, churn_proba, risk_level   (Classification)
      - cluster, cluster_label                (Clustering : PCA → KMeans)
      - monetary_pred                         (Régression MonetaryTotal)

    Paramètres
    ----------
    df_raw : DataFrame avec les colonnes brutes (formulaire ou CSV)

    Retourne
    --------
    DataFrame avec toutes les prédictions
    """
    load_artifacts()

    # ── 1. CLASSIFICATION ─────────────────────────────────────────────
    X_clf = transform_for_inference(df_raw, _country_means, _impute_values, _feature_names)

    missing = [c for c in _feature_names if c not in X_clf.columns]
    if missing:
        raise ValueError(f"Features manquantes pour la classification : {missing}")

    X_clf_scaled = pd.DataFrame(
        _scaler.transform(X_clf),
        columns=_feature_names,
        index=X_clf.index
    )

    preds  = _model.predict(X_clf_scaled)
    probas = _model.predict_proba(X_clf_scaled)[:, 1]

    # ── 2. CLUSTERING : PCA réduite → KMeans ──────────────────────────
    # On applique la même PCA que celle entraînée dans train_model.py
    # pour avoir des clusters équilibrés
    X_pca    = _pca_cluster.transform(X_clf_scaled)
    clusters = _kmeans.predict(X_pca)

    # ── 3. RÉGRESSION ─────────────────────────────────────────────────
    df_reg = df_raw.copy()

    # Nettoyage valeurs aberrantes
    if "SupportTicketsCount" in df_reg.columns:
        df_reg["SupportTicketsCount"] = df_reg["SupportTicketsCount"].replace([-1, 999], np.nan)
    if "SatisfactionScore" in df_reg.columns:
        df_reg["SatisfactionScore"] = df_reg["SatisfactionScore"].replace([-1, 99], np.nan)

    # Imputation — toutes les features regression avec medianes du train
    # _impute_values contient maintenant toutes les features reg (Recency, Frequency, etc.)
    for col in _reg_features:
        if col in df_reg.columns:
            mediane = _impute_values.get(col, df_reg[col].median() if df_reg[col].notna().any() else 0)
            df_reg[col] = df_reg[col].fillna(mediane)
        else:
            # Colonne absente (formulaire web) : utiliser mediane du train
            df_reg[col] = _impute_values.get(col, 0)

    X_reg         = df_reg[_reg_features].copy()
    X_reg_scaled  = _scaler_reg.transform(X_reg)
    monetary_pred = _reg_model.predict(X_reg_scaled)

    # ── 4. Assemblage des résultats ────────────────────────────────────
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
    print("  PREDICT — Test sur 10 clients aléatoires")
    print("=" * 60)

    if not os.path.exists(raw_path):
        print(f"Fichier introuvable : {raw_path}")
        exit(1)

    df_raw = pd.read_csv(raw_path)
    sample = df_raw.sample(10, random_state=99).reset_index(drop=True)

    results = predict(sample)

    print("\nRésultats :")
    print(results.to_string(index=False))

    print(f"\n  Churners prédits    : {results['churn_pred'].sum()} / {len(results)}")
    print(f"  Probabilité moyenne : {results['churn_proba'].mean():.2%}")
    print(f"  Risque              : {results['risk_level'].value_counts().to_dict()}")
    print(f"  Clusters            : {results['cluster_label'].value_counts().to_dict()}")
    print(f"  Dépense prévue moy. : £{results['monetary_pred'].mean():.2f}")
