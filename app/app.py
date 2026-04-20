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
            return f"""
            <!DOCTYPE html><html><head><meta charset='UTF-8'>
            <title>Client introuvable</title>
            <style>body{{font-family:sans-serif;display:flex;align-items:center;
            justify-content:center;height:100vh;margin:0;background:#f5f5f5;}}
            .box{{background:white;padding:2rem;border-radius:12px;text-align:center;
            border:1px solid #eee;}}
            h2{{color:#e53e3e;}}p{{color:#666;}}</style></head>
            <body><div class='box'><h2>Client introuvable</h2>
            <p>CustomerID <strong>{customer_id}</strong> n'existe pas dans la base.</p>
            <a href='/' style='color:#3182ce;'>Retour accueil</a></div></body></html>
            """, 404

        real_churn = int(row["Churn"].values[0]) if "Churn" in row.columns else None
        results    = predict(row.reset_index(drop=True))
        pred       = results.to_dict(orient="records")[0]

        churn_pred   = int(pred["churn_pred"])
        churn_proba  = float(pred["churn_proba"])
        risk_level   = pred["risk_level"]
        cluster_label= pred["cluster_label"]
        monetary_pred= float(pred["monetary_pred"])
        match        = real_churn == churn_pred
        proba_pct    = round(churn_proba * 100, 1)

        match_color  = "#276749" if match else "#c53030"
        match_bg     = "#f0fff4" if match else "#fff5f5"
        match_border = "#9ae6b4" if match else "#feb2b2"
        match_text   = "Prediction correcte" if match else "Prediction incorrecte"
        match_icon   = "✓" if match else "✗"

        churn_color  = "#276749" if churn_pred == 0 else "#c53030"
        churn_bg     = "#f0fff4" if churn_pred == 0 else "#fff5f5"
        churn_label  = "Client fidele" if churn_pred == 0 else "Churner detecte"

        real_color   = "#276749" if real_churn == 0 else "#c53030"
        real_bg      = "#f0fff4" if real_churn == 0 else "#fff5f5"
        real_label   = "Fidele confirme" if real_churn == 0 else "Churner reel"

        risk_colors  = {
            "Faible"  : ("#2b6cb0", "#ebf8ff", "#bee3f8"),
            "Moyen"   : ("#744210", "#fffff0", "#fefcbf"),
            "Eleve"   : ("#c05621", "#fffaf0", "#fbd38d"),
            "Critique": ("#c53030", "#fff5f5", "#feb2b2"),
        }
        rc, rbg, rbd = risk_colors.get(risk_level, ("#4a5568", "#f7fafc", "#e2e8f0"))

        bar_color    = "#3182ce" if proba_pct < 40 else "#dd6b20" if proba_pct < 70 else "#e53e3e"

        cluster_descriptions = {
            "Cluster 0": "Meilleurs clients — Reactivation prioritaire",
            "Cluster 1": "Actifs standard — Fidelisation ciblee",
            "Cluster 2": "Faible valeur — Retention urgente",
        }
        cluster_desc = cluster_descriptions.get(cluster_label, cluster_label)

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Verification Client #{customer_id}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f7fafc;
      min-height: 100vh;
      padding: 2rem 1rem;
      color: #2d3748;
    }}
    .container {{ max-width: 680px; margin: 0 auto; }}
    .header {{
      background: #1a365d;
      color: white;
      padding: 1.25rem 1.5rem;
      border-radius: 12px 12px 0 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .header h1 {{ font-size: 1.1rem; font-weight: 500; }}
    .header span {{ font-size: 0.85rem; opacity: 0.7; }}
    .card {{
      background: white;
      padding: 1.5rem;
      border: 1px solid #e2e8f0;
    }}
    .card-last {{ border-radius: 0 0 12px 12px; }}
    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 1rem;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin-bottom: 1rem;
    }}
    .metric {{
      padding: 1rem;
      border-radius: 8px;
      text-align: center;
    }}
    .metric-label {{
      font-size: 0.75rem;
      color: #718096;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .metric-value {{
      font-size: 1.5rem;
      font-weight: 600;
    }}
    .metric-sub {{
      font-size: 0.75rem;
      margin-top: 4px;
    }}
    .section-title {{
      font-size: 0.8rem;
      font-weight: 500;
      color: #718096;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.75rem;
    }}
    .bar-wrap {{
      background: #edf2f7;
      border-radius: 4px;
      height: 8px;
      overflow: hidden;
      margin-top: 8px;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 4px;
      background: {bar_color};
      width: {proba_pct}%;
    }}
    .bar-label {{
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      color: #718096;
      margin-bottom: 4px;
    }}
    .alert {{
      padding: 0.875rem 1rem;
      border-radius: 8px;
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.9rem;
      font-weight: 500;
      margin-top: 0;
    }}
    .dot {{
      width: 10px; height: 10px;
      border-radius: 50%;
      flex-shrink: 0;
    }}
    .back-link {{
      display: inline-block;
      margin-top: 1rem;
      color: #3182ce;
      text-decoration: none;
      font-size: 0.875rem;
    }}
    .back-link:hover {{ text-decoration: underline; }}
    .divider {{ border: none; border-top: 1px solid #edf2f7; margin: 1rem 0; }}
  </style>
</head>
<body>
  <div class="container">

    <div class="header">
      <h1>Verification Client #{customer_id}</h1>
      <span>ML Retail — Analyse Churn</span>
    </div>

    <div class="card">
      <p class="section-title">Resultat de la prediction</p>
      <div class="grid-3">
        <div class="metric" style="background:{churn_bg};">
          <div class="metric-label">Churn predit</div>
          <div class="metric-value" style="color:{churn_color};">{churn_pred}</div>
          <div class="metric-sub" style="color:{churn_color};">{churn_label}</div>
        </div>
        <div class="metric" style="background:{real_bg};">
          <div class="metric-label">Churn reel</div>
          <div class="metric-value" style="color:{real_color};">{real_churn}</div>
          <div class="metric-sub" style="color:{real_color};">{real_label}</div>
        </div>
        <div class="metric" style="background:{rbg};">
          <div class="metric-label">Niveau risque</div>
          <div class="metric-value" style="color:{rc};">{risk_level}</div>
          <div class="metric-sub" style="color:{rc};">{proba_pct}% churn</div>
        </div>
      </div>

      <div class="bar-label">
        <span>Probabilite de churn</span>
        <span style="font-weight:500; color:{bar_color};">{proba_pct}%</span>
      </div>
      <div class="bar-wrap"><div class="bar-fill"></div></div>

      <hr class="divider">

      <p class="section-title">Segmentation et valeur</p>
      <div class="grid-2">
        <div class="metric" style="background:#f7fafc; text-align:left; padding:1rem;">
          <div class="metric-label">Segment client</div>
          <div style="font-size:1.1rem; font-weight:600; color:#2d3748; margin:4px 0;">{cluster_label}</div>
          <div style="font-size:0.75rem; color:#718096;">{cluster_desc}</div>
        </div>
        <div class="metric" style="background:#f7fafc; text-align:left; padding:1rem;">
          <div class="metric-label">Depense estimee</div>
          <div style="font-size:1.1rem; font-weight:600; color:#2d3748; margin:4px 0;">£{monetary_pred:,.2f}</div>
          <div style="font-size:0.75rem; color:#718096;">MonetaryTotal estime</div>
        </div>
      </div>

      <div class="alert" style="background:{match_bg}; border:1px solid {match_border}; color:{match_color};">
        <div class="dot" style="background:{match_color};"></div>
        {match_icon} {match_text} — le modele a {'bien' if match else 'mal'} identifie ce client
      </div>

      <a href="/" class="back-link">← Retour a l'accueil</a>
    </div>

  </div>
</body>
</html>"""

        return html

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    check_columns()   
    app.run(debug=True)