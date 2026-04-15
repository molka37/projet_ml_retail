# Projet Machine Learning – Prédiction du Churn


Analyser le comportement des clients d’un site e-commerce et prédire le churn (départ client) à l’aide du Machine Learning.

Le projet suit une chaîne complète de traitement :

> **Exploration → Prétraitement → Modélisation → Évaluation → Déploiement (Flask)**


## 📁 Structure du projet

projet_ml_retail/
│
├── data/
│   ├── raw/           # Données brutes
│   ├── processed/     # Données nettoyées
│   └── train_test/    # Jeux train/test
│
├── notebooks/         # Exploration (Jupyter)
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── models/            # Modèles et artefacts (.pkl)
├── app/               # Application Flask
├── reports/           # Graphiques et résultats
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone <repo_url>
cd projet_ml_retail
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

### 3. Activer l’environnement

* Windows :

```bash
venv\Scripts\activate
```

* Linux / Mac :

```bash
source venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

##  Étapes du projet

### 1. Exploration des données

Utiliser le notebook :

```bash
notebooks/exploration.ipynb
```

---

### 2. Prétraitement des données

* Nettoyage
* Imputation des valeurs manquantes
* Feature engineering
* Encodage
* Split train/test

```bash
python src/preprocessing.py
```

---

### 3. Entraînement du modèle

Ce script réalise :

* Normalisation (StandardScaler)
* ACP (PCA)
* Classification (RandomForest + Optuna)
* Clustering (KMeans)
* Régression (RandomForestRegressor)

```bash
python src/train_model.py
```

---

### 4. Prédiction

Tester le modèle sur de nouveaux clients :

```bash
python src/predict.py
```

---

### 5. Déploiement avec Flask

Lancer l’application web :

```bash
python app/app.py
```

Puis ouvrir dans le navigateur :

```
http://127.0.0.1:5000
```

---

##  Modèles utilisés

* **Classification** : RandomForestClassifier (prédiction du churn)
* **Clustering** : KMeans (segmentation des clients)
* **Régression** : RandomForestRegressor (prédiction des dépenses)

---

## Évaluation

Le modèle de classification est évalué avec :

* Accuracy
* AUC-ROC
* F1-score
* Matrice de confusion

La régression est évaluée avec :

* MAE
* RMSE
* R²

---

## Artefacts sauvegardés

Dans models/ :
- model.pkl        → modèle de classification
- scaler.pkl       → normalisation classification
- features.pkl     → liste des features
- kmeans.pkl       → modèle de clustering
- pca_cluster.pkl  → PCA avant KMeans
- reg_model.pkl    → modèle de régression
- scaler_reg.pkl   → normalisation régression
- reg_features.pkl → features de régression
- `country_means.pkl`→ encodage pays
- `impute_values.pkl`→ valeurs d'imputation

##  Interface Web

L’application Flask permet :

* charger un client aléatoire
* saisir des données
* prédire le churn
* afficher :

  * probabilité
  * niveau de risque
  * prédiction

---

##  Conclusion

Ce projet implémente une chaîne complète de Machine Learning :

✔ Exploration
✔ Prétraitement
✔ Classification
✔ Clustering
✔ Régression
✔ Évaluation
✔ Déploiement Flask

---

##  Auteur

Projet réalisé dans le cadre du module Machine Learning – GI2
