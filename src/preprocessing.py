import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=[c for c in ['NewsletterSubscribed', 'LastLoginIP', 'CustomerID']
                           if c in df.columns])

    df['SatisfactionScore']   = df['SatisfactionScore'].replace([-1, 99], np.nan)
    df['SupportTicketsCount'] = df['SupportTicketsCount'].replace([-1, 999], np.nan)

    neg = (df['MonetaryTotal'] < 0).sum()
    if neg:
        print(f"[clean] MonetaryTotal négatif : {neg} cas (conservés)")

    df['Age']                     = df['Age'].fillna(df['Age'].median())
    df['AvgDaysBetweenPurchases'] = df['AvgDaysBetweenPurchases'].fillna(
                                        df['AvgDaysBetweenPurchases'].median())
    df['SatisfactionScore']       = df['SatisfactionScore'].fillna(
                                        df['SatisfactionScore'].median())
    df['SupportTicketsCount']     = df['SupportTicketsCount'].fillna(
                                        df['SupportTicketsCount'].median())

    df['RegistrationDate'] = pd.to_datetime(
        df['RegistrationDate'], dayfirst=True, errors='coerce'
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
    df['MonetaryPerDay']   = df['MonetaryTotal'] / (df['Recency'] + 1)
    df['AvgBasketValue']   = df['MonetaryTotal'] / (df['Frequency'] + 1)
    df['TenureRatio']      = df['Recency'] / (df['CustomerTenureDays'] + 1)
    df['DiversityPerTrans']= df['UniqueProducts'] / (df['Frequency'] + 1)
    print(f"[feat]  {df.shape[1]} colonnes au total")
    return df


def encode_data(df: pd.DataFrame, target_col: str = 'Churn',
                country_means: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

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
        if country_means is None:
            if target_col not in df.columns:
                raise ValueError("country_means=None mais target_col absent du DataFrame.")
            country_means = df.groupby('Country')[target_col].mean().to_dict()
        global_mean = df[target_col].mean() if target_col in df.columns else 0.33
        df['Country_encoded'] = df['Country'].map(country_means).fillna(global_mean)
        df = df.drop(columns=['Country'])

    onehot_cols = [c for c in ['RFMSegment','CustomerType','FavoriteSeason',
                                'Region','WeekendPreference','ProductDiversity',
                                'Gender','AccountStatus'] if c in df.columns]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    print(f"[enc]   Encodage terminé → {df.shape[1]} colonnes")
    return df, country_means



if __name__ == "__main__":

    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path   = os.path.join(BASE_DIR, "data", "raw", "data.csv")
    tt_dir     = os.path.join(BASE_DIR, "data", "train_test")
    proc_dir   = os.path.join(BASE_DIR, "data", "processed")
    models_dir = os.path.join(BASE_DIR, "models")

    df = load_data(raw_path)
    df = clean_data(df)
    df = feature_engineering(df)

    y = df['Churn']
    X = df.drop(columns=['Churn'])
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = X_train_raw.copy()
    train_df['Churn'] = y_train.values
    X_train_enc, country_means = encode_data(train_df, target_col='Churn')
    X_train_enc = X_train_enc.drop(columns=['Churn'])

    test_df = X_test_raw.copy()
    test_df['Churn'] = y_test.values
    X_test_enc, _ = encode_data(test_df, target_col='Churn', country_means=country_means)
    X_test_enc = X_test_enc.drop(columns=['Churn'])

    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    X_train_enc.to_csv(os.path.join(tt_dir, "X_train.csv"), index=False)
    X_test_enc.to_csv(os.path.join(tt_dir,  "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(tt_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(tt_dir,  "y_test.csv"),  index=False)

    cleaned = X_train_enc.copy()
    cleaned['Churn'] = y_train.values
    cleaned.to_csv(os.path.join(proc_dir, "cleaned_data.csv"), index=False)

    joblib.dump(country_means, os.path.join(models_dir, "country_means.pkl"))
    print(f"[save]  country_means → {models_dir}/country_means.pkl")

    print(f"\n{'='*50}")
    print(f"  X_train : {X_train_enc.shape}  (non scalé)")
    print(f"  X_test  : {X_test_enc.shape}   (non scalé)")
    print(f"  Churn   : {y_train.mean():.1%} positifs (train)")
    print(f"{'='*50}")