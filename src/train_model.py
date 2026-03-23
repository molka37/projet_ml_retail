import joblib
import optuna

from preprocessing import load_data, clean_data, feature_engineering, encode_data
from utils import print_section

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():

    # =========================
    # LOAD + PREPROCESSING
    # =========================
    df = load_data("data/raw/data.csv")
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_data(df)

    # =========================
    # REMOVE DATA LEAKAGE
    # =========================
    cols_to_drop = [col for col in df.columns if (
        col.startswith('ChurnRiskCategory') or
        col.startswith('AccountStatus') or
        col.startswith('CustomerType') or
        col.startswith('RFMSegment') or
        col.startswith('SpendingCategory')
    )]
    cols_to_drop.append('Churn')

    X = df.drop(columns=cols_to_drop)
    y = df['Churn']

    # =========================
    # TRAIN / TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # SAVE TRAIN / TEST (PDF)
    # =========================
    X_train.to_csv("data/train_test/X_train.csv", index=False)
    X_test.to_csv("data/train_test/X_test.csv", index=False)
    y_train.to_csv("data/train_test/y_train.csv", index=False)
    y_test.to_csv("data/train_test/y_test.csv", index=False)

    # =========================
    # OPTUNA + CROSS VALIDATION
    # =========================
    print_section("Random Forest (Optuna + CV)")

    def objective(trial):

        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

        # 🔥 régularisation ajoutée
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )

        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=3,
            scoring='f1'
        ).mean()

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print("Best parameters:", study.best_params)

    # =========================
    # FINAL MODEL
    # =========================
    best_model = RandomForestClassifier(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)

    # =========================
    # FINAL EVALUATION
    # =========================
    y_pred = best_model.predict(X_test)

    print("\nFinal Test Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # =========================
    # VALIDATION CROISÉE FINALE
    # =========================
    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring='f1'
    )

    print("\nCV F1 Score:", cv_scores.mean(), "±", cv_scores.std())

    # =========================
    # SAVE MODEL
    # =========================
    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(X_train.columns.tolist(), "models/features.pkl")

if __name__ == "__main__":
    main()
