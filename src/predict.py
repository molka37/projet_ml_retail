import os
import joblib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(BASE_DIR, "models")
MODEL_PATH    = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.pkl")
CM_PATH       = os.path.join(MODELS_DIR, "country_means.pkl")

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


def load_artifacts() -> tuple:
    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    country_means = joblib.load(CM_PATH)
    print(f"[load] Artefacts chargés — {len(feature_names)} features attendues")
    return model, scaler, feature_names, country_means


def preprocess_input(
    df_raw: pd.DataFrame,
    country_means: dict,
    feature_names: list
) -> pd.DataFrame:
    """
    Reproduit exactement preprocessing.py + suppression leakage de train_model.py.
    Retourne un DataFrame aligné sur feature_names, prêt pour scaler.transform().
    """
    df = df_raw.copy()

    df = df.drop(columns=[c for c in
                           ['NewsletterSubscribed', 'LastLoginIP', 'CustomerID', 'Churn']
                           if c in df.columns])

    if 'SatisfactionScore' in df.columns:
        df['SatisfactionScore'] = df['SatisfactionScore'].replace([-1, 99], np.nan)
    if 'SupportTicketsCount' in df.columns:
        df['SupportTicketsCount'] = df['SupportTicketsCount'].replace([-1, 999], np.nan)

    impute = {
        'Age': 46.0, 'AvgDaysBetweenPurchases': 14.0,
        'SatisfactionScore': 3.0, 'SupportTicketsCount': 2.0,
    }
    for col, val in impute.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(
            df['RegistrationDate'], dayfirst=True, errors='coerce'
        )
        df['RegYear']    = df['RegistrationDate'].dt.year.fillna(2010).astype(int)
        df['RegMonth']   = df['RegistrationDate'].dt.month.fillna(1).astype(int)
        df['RegDay']     = df['RegistrationDate'].dt.day.fillna(1).astype(int)
        df['RegWeekday'] = df['RegistrationDate'].dt.weekday.fillna(0).astype(int)
        df = df.drop(columns=['RegistrationDate'])

    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)
    if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)
    if 'UniqueProducts' in df.columns and 'Frequency' in df.columns:
        df['DiversityPerTrans'] = df['UniqueProducts'] / (df['Frequency'] + 1)

    ordinal_mappings = {
        'AgeCategory'      : ['18-24','25-34','35-44','45-54','55-64','65+','Inconnu'],
        'SpendingCategory' : ['Low','Medium','High','VIP'],
        'LoyaltyLevel'     : ['Nouveau','Jeune','Établi','Ancien','Inconnu'],
        'ChurnRiskCategory': ['Faible','Moyen','Élevé','Critique'],
        'BasketSizeCategory': ['Petit','Moyen','Grand','Inconnu'],
        'PreferredTimeOfDay': ['Matin','Midi','Après-midi','Soir','Nuit'],
    }
    for col, order in ordinal_mappings.items():
        if col in df.columns:
            mapping = {v: i for i, v in enumerate(order)}
            df[col] = df[col].map(mapping).fillna(-1).astype(int)

    if 'Country' in df.columns:
        df['Country_encoded'] = df['Country'].map(country_means).fillna(0.33)
        df = df.drop(columns=['Country'])

    onehot_cols = [c for c in ['RFMSegment','CustomerType','FavoriteSeason',
                                'Region','WeekendPreference','ProductDiversity',
                                'Gender','AccountStatus'] if c in df.columns]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    df = df.reindex(columns=feature_names, fill_value=0)

    return df


def predict(df_raw: pd.DataFrame) -> pd.DataFrame:
 
    model, scaler, feature_names, country_means = load_artifacts()

    X = preprocess_input(df_raw, country_means, feature_names)

    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)

    preds  = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)[:, 1]

    def risk_label(p):
        if p < 0.25:   return 'Faible'
        elif p < 0.50: return 'Moyen'
        elif p < 0.75: return 'Élevé'
        else:          return 'Critique'

    results = df_raw[['CustomerID']].copy().reset_index(drop=True) \
              if 'CustomerID' in df_raw.columns \
              else pd.DataFrame(index=range(len(df_raw)))

    results['churn_pred']  = preds
    results['churn_proba'] = np.round(probas, 4)
    results['risk_level']  = [risk_label(p) for p in probas]

    return results


if __name__ == "__main__":

    raw_path = os.path.join(BASE_DIR, "data", "raw", "data.csv")

    print("=" * 50)
    print("  PREDICT — Test sur 10 clients aléatoires")
    print("=" * 50)

    df_raw  = pd.read_csv(raw_path)
    sample  = df_raw.sample(10, random_state=99).reset_index(drop=True)
    results = predict(sample)

    print("\nRésultats :")
    print(results.to_string(index=False))

    print(f"\n  Clients prédits churners : {results['churn_pred'].sum()} / {len(results)}")
    print(f"  Probabilité moyenne      : {results['churn_proba'].mean():.2%}")
    print(f"  Distribution risque      : {results['risk_level'].value_counts().to_dict()}")