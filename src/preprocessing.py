import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=[c for c in ['NewsletterSubscribed', 'LastLoginIP', 'CustomerID']
                          if c in df.columns])

    if 'SatisfactionScore' in df.columns:
        df['SatisfactionScore'] = df['SatisfactionScore'].replace([-1, 99], np.nan)

    if 'SupportTicketsCount' in df.columns:
        df['SupportTicketsCount'] = df['SupportTicketsCount'].replace([-1, 999], np.nan)

    neg = (df['MonetaryTotal'] < 0).sum() if 'MonetaryTotal' in df.columns else 0
    if neg:
        print(f"[clean] MonetaryTotal négatif : {neg} cas (conservés)")

    # NOTE : l'imputation des NaN numeriques est faite APRES le split
    # dans le bloc __main__ pour eviter le data leakage (train -> test)

    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(
            df['RegistrationDate'],
            format='mixed',
            dayfirst=True,
            errors='coerce'
        )
        df['RegYear'] = df['RegistrationDate'].dt.year
        df['RegMonth'] = df['RegistrationDate'].dt.month
        df['RegDay'] = df['RegistrationDate'].dt.day
        df['RegWeekday'] = df['RegistrationDate'].dt.weekday
        df = df.drop(columns=['RegistrationDate'])

    print(f"[clean] Terminé → {df.shape[1]} colonnes | NaN : {df.isnull().sum().sum()}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)

    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)

    if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)

    if 'UniqueProducts' in df.columns and 'Frequency' in df.columns:
        df['DiversityPerTrans'] = df['UniqueProducts'] / (df['Frequency'] + 1)

    print(f"[feat]  {df.shape[1]} colonnes au total")
    return df


def fit_country_encoder(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    temp = X_train.copy()
    temp['Churn'] = y_train.values
    return temp.groupby('Country')['Churn'].mean().to_dict() if 'Country' in temp.columns else {}


def encode_data(df: pd.DataFrame, country_means: dict | None = None) -> pd.DataFrame:
    df = df.copy()

    ordinal_mappings = {
        'AgeCategory': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
        'SpendingCategory': ['Low', 'Medium', 'High', 'VIP'],
        'LoyaltyLevel': ['Nouveau', 'Jeune', 'Établi', 'Ancien', 'Inconnu'],
        'ChurnRiskCategory': ['Faible', 'Moyen', 'Élevé', 'Critique'],
        'BasketSizeCategory': ['Petit', 'Moyen', 'Grand', 'Inconnu'],
        'PreferredTimeOfDay': ['Matin', 'Midi', 'Après-midi', 'Soir', 'Nuit'],
    }

    for col, order in ordinal_mappings.items():
        if col in df.columns:
            mapping = {v: i for i, v in enumerate(order)}
            df[col] = df[col].map(mapping).fillna(-1).astype(int)

    if 'Country' in df.columns:
        default_country = np.mean(list(country_means.values())) if country_means else 0.33
        df['Country_encoded'] = df['Country'].map(country_means).fillna(default_country)
        df = df.drop(columns=['Country'])

    onehot_cols = [c for c in [
        'RFMSegment', 'CustomerType', 'FavoriteSeason',
        'Region', 'WeekendPreference', 'ProductDiversity',
        'Gender', 'AccountStatus'
    ] if c in df.columns]

    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    print(f"[enc]   Encodage terminé → {df.shape[1]} colonnes")
    return df


def transform_for_inference(
    df_raw: pd.DataFrame,
    country_means: dict,
    impute_values: dict,
    feature_names: list
) -> pd.DataFrame:
    df = df_raw.copy()

    df = df.drop(columns=[c for c in ['NewsletterSubscribed', 'LastLoginIP', 'CustomerID', 'Churn']
                          if c in df.columns])

    if 'SatisfactionScore' in df.columns:
        df['SatisfactionScore'] = df['SatisfactionScore'].replace([-1, 99], np.nan)

    if 'SupportTicketsCount' in df.columns:
        df['SupportTicketsCount'] = df['SupportTicketsCount'].replace([-1, 999], np.nan)

    for col, val in impute_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(
            df['RegistrationDate'],
            format='mixed',
            dayfirst=True,
            errors='coerce'
        )
        df['RegYear'] = df['RegistrationDate'].dt.year.fillna(2010).astype(int)
        df['RegMonth'] = df['RegistrationDate'].dt.month.fillna(1).astype(int)
        df['RegDay'] = df['RegistrationDate'].dt.day.fillna(1).astype(int)
        df['RegWeekday'] = df['RegistrationDate'].dt.weekday.fillna(0).astype(int)
        df = df.drop(columns=['RegistrationDate'])

    df = feature_engineering(df)
    df = encode_data(df, country_means=country_means)

    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], errors='ignore')
    df = df.reindex(columns=feature_names, fill_value=0)

    return df


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(BASE_DIR, "data", "raw", "data.csv")
    tt_dir = os.path.join(BASE_DIR, "data", "train_test")
    proc_dir = os.path.join(BASE_DIR, "data", "processed")
    models_dir = os.path.join(BASE_DIR, "models")

    os.makedirs(tt_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = load_data(raw_path)
    df = clean_data(df)
    df = feature_engineering(df)

    y = df['Churn']
    X = df.drop(columns=['Churn'])

    # Split D'ABORD — puis on calcule les medianes sur X_train uniquement
    # Evite le data leakage : le test set ne doit pas influencer l'imputation
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Imputation sur X_train uniquement (pas sur tout df)
    # Inclut toutes les features utilisees par la regression
    def safe_median(df, col, default):
        return float(df[col].median()) if col in df.columns and df[col].notna().any() else default

    impute_values = {
        'Age':                     safe_median(X_train_raw, 'Age', 46.0),
        'AvgDaysBetweenPurchases': safe_median(X_train_raw, 'AvgDaysBetweenPurchases', 14.0),
        'SatisfactionScore':       safe_median(X_train_raw, 'SatisfactionScore', 3.0),
        'SupportTicketsCount':     safe_median(X_train_raw, 'SupportTicketsCount', 2.0),
        # Features regression — necessaires pour predict.py quand colonne absente
        'Recency':                 safe_median(X_train_raw, 'Recency', 30.0),
        'Frequency':               safe_median(X_train_raw, 'Frequency', 4.0),
        'TotalQuantity':           safe_median(X_train_raw, 'TotalQuantity', 50.0),
        'UniqueProducts':          safe_median(X_train_raw, 'UniqueProducts', 20.0),
        'TotalTransactions':       safe_median(X_train_raw, 'TotalTransactions', 10.0),
    }

    # Appliquer l'imputation sur X_train et X_test avec les memes valeurs
    IMPUTE_COLS = ['Age', 'AvgDaysBetweenPurchases', 'SatisfactionScore', 'SupportTicketsCount']
    for col in IMPUTE_COLS:
        if col in X_train_raw.columns:
            X_train_raw[col] = X_train_raw[col].fillna(impute_values[col])
            X_test_raw[col]  = X_test_raw[col].fillna(impute_values[col])

    print(f"[impute] Valeurs calculees sur X_train : {impute_values}")

    country_means = fit_country_encoder(X_train_raw, y_train)

    X_train_enc = encode_data(X_train_raw, country_means=country_means)
    X_test_enc = encode_data(X_test_raw, country_means=country_means)

    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    X_train_enc.to_csv(os.path.join(tt_dir, "X_train.csv"), index=False)
    X_test_enc.to_csv(os.path.join(tt_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(tt_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(tt_dir, "y_test.csv"), index=False)

    cleaned = X_train_enc.copy()
    cleaned['Churn'] = y_train.values
    cleaned.to_csv(os.path.join(proc_dir, "cleaned_data.csv"), index=False)

    joblib.dump(country_means, os.path.join(models_dir, "country_means.pkl"))
    print(f"[save]  country_means → {models_dir}/country_means.pkl")

    joblib.dump(impute_values, os.path.join(models_dir, "impute_values.pkl"))
    print(f"[save]  impute_values → {models_dir}/impute_values.pkl")

    print(f"\n{'=' * 50}")
    print(f"  X_train : {X_train_enc.shape}  (non scalé)")
    print(f"  X_test  : {X_test_enc.shape}   (non scalé)")
    print(f"  Churn   : {y_train.mean():.1%} positifs (train)")
    print(f"{'=' * 50}")