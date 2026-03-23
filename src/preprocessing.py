#prédire si un client va churn (partir) ou non
import pandas as pd

def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AvgDaysBetweenPurchases'] = df['AvgDaysBetweenPurchases'].fillna(
        df['AvgDaysBetweenPurchases'].median()
    )

    df = df.drop(columns=['NewsletterSubscribed', 'LastLoginIP'])

    df['RegistrationDate'] = pd.to_datetime(
        df['RegistrationDate'],
        dayfirst=True,
        errors='coerce'
    )

    df['RegYear'] = df['RegistrationDate'].dt.year
    df['RegMonth'] = df['RegistrationDate'].dt.month
    df = df.drop(columns=['RegistrationDate'])

    df['SatisfactionScore'] = df['SatisfactionScore'].replace(
        99, df['SatisfactionScore'].median()
    )
    df['SupportTicketsCount'] = df['SupportTicketsCount'].replace(
        999, df['SupportTicketsCount'].median()
    )

    return df


def feature_engineering(df):
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)
    return df


def encode_data(df):
    return pd.get_dummies(df, drop_first=True)


if __name__ == "__main__":
    df = load_data("data/raw/data.csv")
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_data(df)

    print(df.head())
    print(df.shape)
    df.to_csv("data/processed/cleaned_data.csv", index=False)