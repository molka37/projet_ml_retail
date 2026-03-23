import joblib
from preprocessing import load_data, clean_data, feature_engineering, encode_data


def main():

    # Load + preprocessing
    df = load_data("data/raw/data.csv")
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_data(df)

    # Remove leakage (comme train)
    cols_to_drop = [col for col in df.columns if (
        col.startswith('ChurnRiskCategory') or
        col.startswith('AccountStatus') or
        col.startswith('CustomerType') or
        col.startswith('RFMSegment') or
        col.startswith('SpendingCategory')
    )]
    cols_to_drop.append('Churn')

    X = df.drop(columns=cols_to_drop)

    # Load model
    model = joblib.load("models/model.pkl")

    # Predict
    predictions = model.predict(X)

    print("Predictions:", predictions[:10])


if __name__ == "__main__":
    main()