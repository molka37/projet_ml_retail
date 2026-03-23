
Projet Machine Learning réalisé dans le cadre du module ML GI2.
Objectif : analyser le comportement des clients d’un site e-commerce et prédire leurs comportements d’achat.


git clone <lien_github>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

##  Structure du Projet

- data/ : données brutes et nettoyées
- notebooks/ : exploration et analyse
- src/ : scripts de production
- models/ : modèles sauvegardés
- app/ : application Flask
- reports/ : visualisations

---

##  Utilisation

### Prétraitement
python src/preprocessing.py

### Entraînement du modèle
python src/train_model.py

### Prédiction
python src/predict.py