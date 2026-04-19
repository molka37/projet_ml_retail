from flask import Flask, request, jsonify, render_template
import os
import sys
import joblib
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(CURRENT_DIR)
SRC_DIR     = os.path.join(BASE_DIR, "src")
DATA_PATH   = os.path.join(BASE_DIR, "data", "raw", "data.csv")
IMPUTE_PATH = os.path.join(BASE_DIR, "models", "impute_values.pkl")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predict import predict

TEMPLATE_DIR = os.path.join(CURRENT_DIR, "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)

impute_values = joblib.load(IMPUTE_PATH)


# CORRECTION : verification des colonnes au demarrage (PDF section 4)
def check_columns():
    """Verifie que les colonnes attendues sont presentes dans data.csv"""
    try:
        df = pd.read_csv(DATA_PATH, nrows=1)
        expected = [
            "CustomerID", "Frequency", "MonetaryTotal", "MonetaryAvg",
            "TotalQuantity", "AvgDaysBetweenPurchases", "UniqueProducts",
            "TotalTransactions", "SupportTicketsCount", "SatisfactionScore",
            "Age", "Gender", "FavoriteSeason", "AccountStatus", "Churn"
        ]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            print(f"[app] ATTENTION colonnes manquantes dans data.csv : {missing}")
        else:
            print("[app] Colonnes CSV verifiees OK")
    except Exception as e:
        print(f"[app] Impossible de lire data.csv : {e}")


def normalize_gender(value):
    if pd.isna(value):
        return "F"
    v = str(value).strip().lower()
    if v in ["f", "female", "femme"]:
        return "F"
    if v in ["m", "male", "homme"]:
        return "M"
    return "F"


def safe_value(sample, col, default=None, cast=None):
    if col not in sample.index or pd.isna(sample[col]):
        return default
    value = sample[col]
    if cast is not None:
        try:
            return cast(value)
        except Exception:
            return default
    return value


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Flask API operationnelle"
    })


@app.route("/random_client", methods=["GET"])
def random_client():
    try:
        df     = pd.read_csv(DATA_PATH)
        sample = df.sample(1).iloc[0]

        client = {
            "CustomerID"             : safe_value(sample, "CustomerID", 0, int),
            "Frequency"              : safe_value(sample, "Frequency", 0, int),
            "MonetaryTotal"          : safe_value(sample, "MonetaryTotal", 0.0, float),
            "MonetaryAvg"            : safe_value(sample, "MonetaryAvg", 0.0, float),
            "TotalQuantity"          : safe_value(sample, "TotalQuantity", 0, int),
            # CORRECTION : nom de colonne verifie coherent avec le CSV
            "AvgDaysBetweenPurchases": safe_value(
                sample, "AvgDaysBetweenPurchases",
                impute_values.get("AvgDaysBetweenPurchases", 14.0), float
            ),
            "UniqueProducts"         : safe_value(sample, "UniqueProducts", 0, int),
            "TotalTransactions"      : safe_value(sample, "TotalTransactions", 0, int),
            "SupportTicketsCount"    : safe_value(
                sample, "SupportTicketsCount",
                impute_values.get("SupportTicketsCount", 2.0), int
            ),
            "SatisfactionScore"      : safe_value(
                sample, "SatisfactionScore",
                impute_values.get("SatisfactionScore", 3.0), float
            ),
            "Age"                    : safe_value(
                sample, "Age",
                impute_values.get("Age", 46.0), float
            ),
            "Gender"                 : normalize_gender(safe_value(sample, "Gender", "F")),
            "FavoriteSeason"         : safe_value(sample, "FavoriteSeason", "Printemps", str),
            "AccountStatus"          : safe_value(sample, "AccountStatus", "Active", str),
        }

        return jsonify({"success": True, "client": client})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict", methods=["GET", "POST"])
def predict_api():
    if request.method == "GET":
        return jsonify({"message": "Envoie une requete POST avec un JSON."})

    try:
        data = request.get_json()

        if data is None:
            return jsonify({"success": False, "error": "Aucune donnee JSON recue"}), 400

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"success": False, "error": "Format JSON invalide"}), 400

        if df.empty:
            return jsonify({"success": False, "error": "DataFrame vide"}), 400

        results = predict(df)

        return jsonify({
            "success"    : True,
            "n_clients"  : len(results),
            "predictions": results.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/verify/<int:customer_id>", methods=["GET"])
def verify(customer_id):
    try:
        df  = pd.read_csv(DATA_PATH)
        row = df[df["CustomerID"] == customer_id]

        if row.empty:
            return jsonify({
                "success": False,
                "error"  : f"CustomerID {customer_id} introuvable dans la base"
            }), 404

        real_churn = int(row["Churn"].values[0]) if "Churn" in row.columns else None

        results = predict(row.reset_index(drop=True))
        pred    = results.to_dict(orient="records")[0]

        return jsonify({
            "success"      : True,
            "customer_id"  : customer_id,
            "real_churn"   : real_churn,
            "churn_pred"   : int(pred["churn_pred"]),
            "churn_proba"  : float(pred["churn_proba"]),
            "risk_level"   : pred["risk_level"],
            "cluster_label": pred["cluster_label"],
            "monetary_pred": float(pred["monetary_pred"]),
            "match"        : real_churn == int(pred["churn_pred"])
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    check_columns()   # CORRECTION : verification colonnes au demarrage
    app.run(debug=True)