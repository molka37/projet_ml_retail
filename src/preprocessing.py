import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


LEAKAGE_COLS = [
    'Recency',
    'ChurnRiskCategory',
    'LoyaltyLevel',
    'SpendingCategory',
    'AccountStatus_Closed', 'AccountStatus_Pending', 'AccountStatus_Suspended',
    'CustomerType_Nouveau', 'CustomerType_Occasionnel',
    'CustomerType_Perdu', 'CustomerType_Régulier',
    'RFMSegment_Dormants', 'RFMSegment_Fidèles', 'RFMSegment_Potentiels',
    'PreferredMonth',
    'CustomerTenureDays',
    'FirstPurchaseDaysAgo',
    'UniqueDescriptions',  
    'UniqueInvoices',     
]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'LastLoginIP' in df.columns:
        df['IP_IsPrivate'] = df['LastLoginIP'].str.match(
            r'^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)'
        ).astype(int)
        print(f"[clean] LastLoginIP → IP_IsPrivate extrait")

    df = df.drop(columns=[c for c in
                           ['NewsletterSubscribed', 'LastLoginIP', 'CustomerID']
                           if c in df.columns])

    if 'SatisfactionScore' in df.columns:
        df['SatisfactionScore'] = df['SatisfactionScore'].replace([-1, 99], np.nan)
    if 'SupportTicketsCount' in df.columns:
        df['SupportTicketsCount'] = df['SupportTicketsCount'].replace([-1, 999], np.nan)

    neg = (df['MonetaryTotal'] < 0).sum() if 'MonetaryTotal' in df.columns else 0
    if neg:
        print(f"[clean] MonetaryTotal négatif : {neg} cas (conservés — retours client)")

    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(
            df['RegistrationDate'],
            format='mixed',
            dayfirst=True,
            errors='coerce'
        )
        df['RegYear']    = df['RegistrationDate'].dt.year
        df['RegMonth']   = df['RegistrationDate'].dt.month
        df['RegDay']     = df['RegistrationDate'].dt.day
        df['RegWeekday'] = df['RegistrationDate'].dt.weekday
        df = df.drop(columns=['RegistrationDate'])

    print(f"[clean] Terminé → {df.shape[1]} colonnes | NaN : {df.isnull().sum().sum()}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)

    if 'UniqueProducts' in df.columns and 'Frequency' in df.columns:
        df['DiversityPerTrans'] = df['UniqueProducts'] / (df['Frequency'] + 1)

    print(f"[feat]  {df.shape[1]} colonnes après feature engineering")
    return df


def fit_country_encoder(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    if 'Country' not in X_train.columns:
        return {}
    temp = X_train[['Country']].copy()
    temp['Churn'] = y_train.values
    return temp.groupby('Country')['Churn'].mean().to_dict()



def encode_data(df: pd.DataFrame, country_means: dict | None = None) -> pd.DataFrame:
    df = df.copy()

    ordinal_mappings = {
        'AgeCategory'       : ['18-24','25-34','35-44','45-54','55-64','65+','Inconnu'],
        'SpendingCategory'  : ['Low','Medium','High','VIP'],
        'LoyaltyLevel'      : ['Nouveau','Jeune','Établi','Ancien','Inconnu'],
        'ChurnRiskCategory' : ['Faible','Moyen','Élevé','Critique'],
        'BasketSizeCategory': ['Petit','Moyen','Grand','Inconnu'],
        'PreferredTimeOfDay': ['Matin','Midi','Après-midi','Soir','Nuit'],
    }

    for col, order in ordinal_mappings.items():
        if col in df.columns:
            mapping = {v: i for i, v in enumerate(order)}
            df[col] = df[col].map(mapping).fillna(-1).astype(int)

    if 'Country' in df.columns:
        default = np.mean(list(country_means.values())) if country_means else 0.33
        df['Country_encoded'] = df['Country'].map(country_means).fillna(default)
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

    if 'LastLoginIP' in df.columns:
        df['IP_IsPrivate'] = df['LastLoginIP'].str.match(
            r'^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)'
        ).astype(int)

    df = df.drop(columns=[c for c in
                           ['NewsletterSubscribed', 'LastLoginIP', 'CustomerID', 'Churn']
                           if c in df.columns])

    if 'SatisfactionScore' in df.columns:
        df['SatisfactionScore'] = df['SatisfactionScore'].replace([-1, 99], np.nan)
    if 'SupportTicketsCount' in df.columns:
        df['SupportTicketsCount'] = df['SupportTicketsCount'].replace([-1, 999], np.nan)

    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(
            df['RegistrationDate'], format='mixed', dayfirst=True, errors='coerce'
        )
        df['RegYear']    = df['RegistrationDate'].dt.year.fillna(2010).astype(int)
        df['RegMonth']   = df['RegistrationDate'].dt.month.fillna(1).astype(int)
        df['RegDay']     = df['RegistrationDate'].dt.day.fillna(1).astype(int)
        df['RegWeekday'] = df['RegistrationDate'].dt.weekday.fillna(0).astype(int)
        df = df.drop(columns=['RegistrationDate'])

    df = feature_engineering(df)

    for col, val in impute_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    df = encode_data(df, country_means=country_means)
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], errors='ignore')
    df = df.reindex(columns=feature_names, fill_value=0)

    return df


if __name__ == "__main__":

    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path   = os.path.join(BASE_DIR, "data", "raw", "data.csv")
    tt_dir     = os.path.join(BASE_DIR, "data", "train_test")
    proc_dir   = os.path.join(BASE_DIR, "data", "processed")
    models_dir = os.path.join(BASE_DIR, "models")
    reports_dir = os.path.join(BASE_DIR, "reports")

    os.makedirs(tt_dir,      exist_ok=True)
    os.makedirs(proc_dir,    exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    df = load_data(raw_path)

    df = clean_data(df)

    y = df['Churn']
    X = df.drop(columns=['Churn'])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_raw = feature_engineering(X_train_raw)
    X_test_raw  = feature_engineering(X_test_raw)

    numeric_cols = X_train_raw.select_dtypes(include=[np.number]).columns

    impute_values = {
        col: float(X_train_raw[col].median())
        for col in numeric_cols
        if X_train_raw[col].notna().any()
    }

    for col, val in impute_values.items():
        if col in X_train_raw.columns:
            X_train_raw[col] = X_train_raw[col].fillna(val)
        if col in X_test_raw.columns:
            X_test_raw[col]  = X_test_raw[col].fillna(val)

    print(f"[impute] {len(impute_values)} colonnes imputées (médiane train)")

    country_means = fit_country_encoder(X_train_raw, y_train)

    X_train_enc = encode_data(X_train_raw, country_means=country_means)
    X_test_enc  = encode_data(X_test_raw,  country_means=country_means)
    X_test_enc  = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)


    from utils import compute_vif

    print("\nCalcul du VIF (multicolinéarité)...")
    vif_df = compute_vif(X_train_enc)

    vif_df.to_csv(os.path.join(reports_dir, "vif_report.csv"), index=False)
    print(f"[save]  vif_report.csv → {reports_dir}")
    high_vif = vif_df[vif_df["VIF"] > 10]

    if not high_vif.empty:
        print("\n⚠ Features à forte multicolinéarité :")
        print(high_vif.head(10))
    else:
        print("\n✅ Pas de multicolinéarité critique (VIF < 10)")
    X_train_enc.to_csv(os.path.join(tt_dir, "X_train.csv"), index=False)
    X_test_enc.to_csv( os.path.join(tt_dir, "X_test.csv"),  index=False)
    y_train.to_csv(    os.path.join(tt_dir, "y_train.csv"),  index=False)
    y_test.to_csv(     os.path.join(tt_dir, "y_test.csv"),   index=False)
    print(f"[save]  Splits sauvegardés → {tt_dir}")

    cleaned = X_train_enc.copy()
    cleaned['Churn'] = y_train.values
    cleaned.to_csv(os.path.join(proc_dir, "cleaned_data.csv"), index=False)

    from utils import missing_report, print_df_info
    print_df_info(df, label="Dataset après nettoyage")
    report = missing_report(df)
    if not report.empty:
        report.to_csv(os.path.join(reports_dir, "missing_report.csv"))
        print(f"[save]  missing_report.csv → {reports_dir}")

    joblib.dump(country_means, os.path.join(models_dir, "country_means.pkl"))
    joblib.dump(impute_values, os.path.join(models_dir, "impute_values.pkl"))
    print(f"[save]  country_means + impute_values → {models_dir}")

    print(f"\n{'='*55}")
    print(f"  X_train : {X_train_enc.shape}  (non scalé)")
    print(f"  X_test  : {X_test_enc.shape}   (non scalé)")
    print(f"  Churn   : {y_train.mean():.1%} positifs (train)")
    print(f"{'='*55}")