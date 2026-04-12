# train_model.py
# Entraînement Random Forest — charge les fichiers NON scalés de preprocessing.py
# Le scaling est fait ICI, après suppression des leakage cols
# → scaler.pkl correspond exactement aux 57 features finales du modèle

import os
import joblib
import optuna
import pandas as pd
import numpy as np

from utils import print_section

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Doit être identique à predict.py
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


def main():

    # ─────────────────────────────────────────────
    # 1. CHARGEMENT (fichiers NON scalés)
    # ─────────────────────────────────────────────
    print_section("1. Chargement des fichiers train/test")

    tt_dir  = os.path.join(BASE_DIR, "data", "train_test")
    X_train = pd.read_csv(os.path.join(tt_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(tt_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(tt_dir, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(tt_dir, "y_test.csv")).squeeze()

    print(f"   X_train : {X_train.shape}")
    print(f"   X_test  : {X_test.shape}")
    print(f"   Churn train : {y_train.mean():.2%} positifs")
    print(f"   Churn test  : {y_test.mean():.2%} positifs")

    # ─────────────────────────────────────────────
    # 2. SUPPRESSION DATA LEAKAGE
    # ─────────────────────────────────────────────
    print_section("2. Suppression des colonnes data leakage")

    cols_dropped = [c for c in LEAKAGE_COLS if c in X_train.columns]
    print(f"   Supprimées ({len(cols_dropped)}) :")
    for c in cols_dropped:
        print(f"     - {c}")

    X_train = X_train.drop(columns=cols_dropped, errors='ignore')
    X_test  = X_test.drop(columns=cols_dropped,  errors='ignore')
    print(f"\n   Features restantes : {X_train.shape[1]}")

    # ─────────────────────────────────────────────
    # 3. NORMALISATION — après suppression leakage
    #    scaler fitté sur exactement les bonnes features
    #    → cohérent avec predict.py
    # ─────────────────────────────────────────────
    print_section("3. Normalisation (StandardScaler)")

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )

    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"   Scaler sauvegardé → {scaler_path}")
    print(f"   (fitté sur {X_train_s.shape[1]} features)")

    # ─────────────────────────────────────────────
    # 4. OPTIMISATION OPTUNA
    # ─────────────────────────────────────────────
    print_section("4. Optuna – Recherche des meilleurs hyperparamètres")

    def objective(trial):
        params = {
            'n_estimators'      : trial.suggest_int('n_estimators', 50, 300),
            'max_depth'         : trial.suggest_int('max_depth', 5, 20),
            'min_samples_split' : trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf'  : trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features'      : trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'class_weight'      : 'balanced',
            'random_state'      : 42,
            'n_jobs'            : -1,
        }
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X_train_s, y_train, cv=3, scoring='f1').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print(f"\n   Meilleurs paramètres : {study.best_params}")
    print(f"   Meilleur F1 (CV-3)   : {study.best_value:.4f}")

    # ─────────────────────────────────────────────
    # 5. ENTRAÎNEMENT DU MODÈLE FINAL
    # ─────────────────────────────────────────────
    print_section("5. Entraînement du modèle final")

    best_params = {**study.best_params, 'class_weight': 'balanced',
                   'random_state': 42, 'n_jobs': -1}
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train_s, y_train)
    print("   Modèle entraîné ✅")

    # ─────────────────────────────────────────────
    # 6. ÉVALUATION SUR X_TEST
    # ─────────────────────────────────────────────
    print_section("6. Évaluation finale sur X_test")

    y_pred  = best_model.predict(X_test_s)
    y_proba = best_model.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n   Accuracy  : {acc:.4f}")
    print(f"   AUC-ROC   : {auc:.4f}")
    print(f"\n   Matrice de confusion :")
    print(f"   TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"   FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Fidèle (0)','Churn (1)'])}")

    # ─────────────────────────────────────────────
    # 7. VALIDATION CROISÉE (5 folds)
    # ─────────────────────────────────────────────
    print_section("7. Validation croisée finale (5 folds)")

    cv_scores = cross_val_score(best_model, X_train_s, y_train, cv=5, scoring='f1')
    print(f"   F1 par fold : {[round(float(s), 4) for s in cv_scores]}")
    print(f"   F1 moyen    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ─────────────────────────────────────────────
    # 8. IMPORTANCE DES FEATURES (top 15)
    # ─────────────────────────────────────────────
    print_section("8. Importance des features (top 15)")

    feat_imp = pd.Series(
        best_model.feature_importances_, index=X_train_s.columns
    ).sort_values(ascending=False)

    for feat, imp in feat_imp.head(15).items():
        print(f"   {feat:<35} {imp:.4f}  {'█' * int(imp * 200)}")

    # ─────────────────────────────────────────────
    # 9. SAUVEGARDE
    # ─────────────────────────────────────────────
    print_section("9. Sauvegarde des artefacts")

    models_dir    = os.path.join(BASE_DIR, "models")
    model_path    = os.path.join(models_dir, "model.pkl")
    features_path = os.path.join(models_dir, "features.pkl")

    joblib.dump(best_model,                model_path)
    joblib.dump(X_train_s.columns.tolist(), features_path)

    print(f"   model.pkl    → {model_path}")
    print(f"   features.pkl → {features_path}")
    print(f"   scaler.pkl   → {scaler_path}")

    print(f"\n{'='*50}")
    print(f"  Accuracy : {acc:.4f}  |  AUC-ROC : {auc:.4f}")
    print(f"  F1 moyen (CV-5) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()