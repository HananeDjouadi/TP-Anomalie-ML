# Atelier Pratique : Détection d'anomalies (M2 DS)

Projet de Machine Learning — Détection d'anomalies et de fraudes avec Python.  
Enseignants : **Haytham Elghazel & Heidi Souibki** (Université Claude Bernard Lyon 1).

---

## Organisation du projet

- `data/` : Jeux de données
  - `creditcard.csv` : Détection de fraudes bancaires (284 807 transactions).
  - `mouse.txt` : Données de synthèse 2D pour la partie 1.
  - `ai4i2020.csv` : Maintenance prédictive industrielle pour la partie 4.
- `notebook/` :
  - [`partie2_fraudes.ipynb`](file:///notebook/partie2_fraudes.ipynb) : Notebook dédié à la **Partie 2 : Détection des fraudes** (Credit Card).
  - [`principal.ipynb`](file:///notebook/principal.ipynb) : Notebook principal pour l'ensemble du projet.
  - [`utils.py`](file:///notebook/utils.py) : Fonctions modulaires et réutilisables (chargement, nettoyage, découpage stratifié, prétraitements sans fuite d'information, harmonisation des scores et recherche de seuil sous contrainte opérationnelle).
- `requirements.txt` : Liste des dépendances (`numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `shap`, `matplotlib`, `seaborn`).

---

## Protocole commun & Règles imposées

1. **Découpage stratifié reproductible :** 60% Apprentissage, 20% Validation, 20% Test (`random_state=42`).
2. **Gestion de `Time` :** Exclue des variables prédictives, conservée pour analyse critique des limites du découpage aléatoire (interpolation vs extrapolation, dérive de concept).
3. **Prévention du Data Leakage :** Suppression des doublons stricts avant découpage pour éviter qu'une même transaction ne figure dans plusieurs partitions. Ajustement (`fit`) des scalers uniquement sur le jeu d'apprentissage.
4. **Contrainte opérationnelle Fraude :** Maximiser le rappel sous la contrainte `Précision >= 80%` sur le jeu de validation avec au moins une alerte.
