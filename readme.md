# Atelier Pratique : Détection d'anomalies (Master 2 Data Science)

Projet académique de Machine Learning — Détection d'anomalies, détection de fraudes bancaires, robustesse et explicabilité des modèles.  
**Enseignants :** Haytham Elghazel & Heidi Souibki  
**Université Claude Bernard Lyon 1** — Année Universitaire 2026-2027

---

## Structure du Projet

```text
TP-Anomalie-ML/
│
├── data/                                      # Jeux de données
│   ├── creditcard.csv                         # Fraudes bancaires (284 807 transactions initiales, 31 colonnes)
│   ├── mouse.txt                              # Données synthétiques 2D (Parties 1 et 5.1)
│   └── ai4i2020.csv                           # Maintenance prédictive industrielle (Partie 4)
│
├── notebook/                                  # Notebooks Jupyter et scripts modulaires
│   ├── mouse.ipynb                            # Notebook : Données synthétiques 2D et Robustesse (Parties 1 & 5.1)
│   ├── partie2et3_fraudes.ipynb               # Notebook complet : Fraudes, EasyEnsemble & SHAP (Parties 2, 3 & 5.2)
│   ├── utils_partie3.py                       # Fonctions modulaires pour la Partie 3 (EasyEnsemble & comparaison)
│   ├── utils_partie5.py                       # Fonctions modulaires pour la Partie 5 (Robustesse & SHAP)
│   ├── utils_credit.py                        # Fonctions modulaires étanches (prétraitements, seuils, courbes)
│   ├── utils_mouse.py                         # Utilitaires pour les données Mouse
│   └── utils.py                               # Utilitaires généraux
│
├── les syntheses/                             # Rapports académiques de synthèse
│   ├── Rapport_Partie2_et_3_Detection_Fraudes.md      # Rapport exhaustif des Parties 2 & 3
│   └── Rapport_Partie5_Robustesse_et_Explicabilite.md # Rapport dédié de la Partie 5 (Robustesse & SHAP)
│
├── requirements.txt                           # Dépendances Python du projet
└── README.md                                  # Présentation générale du projet et résultats
```

---

## Travaux Réalisés et Synthèse Méthodologique

### Partie 1 : Données Synthétiques 2D (`mouse.txt`)
* Exploration des distributions en dimension 2 et comportement des estimateurs de densité de base.

---

### Partie 2 : Détection des Fraudes Bancaires (`creditcard.csv`)
1. **Préparation et audit d'intégrité sans fuite d'information :**
   * Identification et suppression de **1 081 doublons stricts** (évite tout *data leakage* entre partitions).
   * Exclusion de la variable `Time` des prédicteurs et discussion des limites du découpage aléatoire (fuite temporelle, dérive de concept).
   * Découpage stratifié reproductible 60% Train / 20% Validation / 20% Test (`random_state=42`).
   * Mise à l'échelle de `Amount` par **`RobustScaler`** (médiane et IQR), ajusté **exclusivement sur `X_train`**.
2. **Comparaison supervisée et impact du déséquilibre :**
   * Justification de **SMOTE** comme méthode de rééchantillonnage face à *RandomUnderSampler* et *RandomOverSampler*.
   * Évaluation de **12 configurations supervisées** (Régression Logistique et Arbre de décision $\times$ 3 variantes : Sans correction, Pondération `class_weight='balanced'`, SMOTE).
   * Constat : sur la Validation, le rééchantillonnage n'a pas le même effet selon le modèle. Pour la régression logistique, SMOTE et la pondération atteignent la contrainte de précision $\ge 80\%$ (précision de $0.81$ à $0.83$, rappel de $0.766$) ; SMOTE avec $C=0.01$ est retenu. Pour l'arbre de décision, aucune variante avec rééchantillonnage ou pondération n'atteint la contrainte : au seuil maximisant le F1, la précision est au plus de $56\%$ (pondération, profondeur 5) et tombe à $42$ à $45\%$ avec SMOTE. Une explication possible, non testée, est que les fraudes synthétiques dégradent la pureté des feuilles.
3. **Détecteurs non supervisés :**
   * Entraînement strictly non supervisé d'**Isolation Forest** et **Local Outlier Factor (LOF)** avec `novelty=True`.
   * Démonstration de l'échec opérationnel du non-supervisé : malgré un AUC-ROC flatteur ($0.942$), l'Average Precision s'effondre à $0.086$ (confusion entre atypisme statistique et criminalité).
4. **Optimisation sous contrainte opérationnelle métier ($\text{Précision} \ge 80\%$) :**
   * Détermination du seuil sur la Validation : passage du seuil par défaut $0.50$ (précision $5.69\%$, rappel $85.11\%$, non utilisable après SMOTE) au seuil calibré $\tau^* = 0.999996$ (précision $0.8276$, rappel $0.766$).
   * **Modèle retenu :** régression logistique ($C=0.01$) entraînée avec SMOTE sur le Train.
   * **Évaluation finale gelée sur l'ensemble de Test (56 746 transactions) :**
     * **Précision :** $87.95\%$ (73 fraudes sur 83 alertes ; contrainte des $80\%$ respectée)
     * **Rappel :** $76.84\%$ (73 fraudes interceptées sur 95)
     * **F1-score :** $0.8202$ ; **Balanced accuracy :** $0.8841$
     * **Fausses alertes :** 10 alertes indues
     * **Temps d'entraînement :** $1.96\text{ s}$

---

### Partie 3 : Recherche d'une Nouvelle Méthode — EasyEnsemble
* **Méthode retenue :** `EasyEnsembleClassifier` (imbalanced-learn) : $T$ sous-ensembles équilibrés sont formés en sous-échantillonnant la classe majoritaire, un AdaBoost est entraîné sur chaque sous-ensemble, et les scores sont moyennés. Méthode **supervisée**.
* **Publication académique :** Liu, Wu & Zhou (2009), IEEE TSMC-B.
* **Configurations évaluées :** 5, 10 et 20 sous-ensembles, avec les mêmes partitions et la même règle de seuil que la Partie 2 (seuil maximisant le rappel sous contrainte Précision $\ge 80\%$ et au moins une alerte, calculé sur la Validation puis gelé avant le test).
* **Sélection sur la Validation :** 5 sous-ensembles (rappel $0.6915$), devant 20 ($0.6809$) et 10 ($0.6702$).
* **Résultats sur le Test (seuil gelé $\tau = 0.6640$) :** Précision $79.31\%$ (contrainte non respectée sur le test, rapportée telle quelle), Rappel $72.63\%$, F1 $0.7582$, AP $0.7288$, AUC-ROC $0.9741$. Sur 95 fraudes : 69 détectées, 18 fausses alertes, 26 manquées.
* **Temps d'entraînement :** $3.7$ à $4.5\text{ s}$ selon l'exécution (5 sous-ensembles).
* **Point à confirmer :** méthode supervisée dans une partie intitulée « détection d'anomalies » ; validation par l'enseignant en attente.

---

### Partie 5 : Robustesse et Explicabilité (Synthèse Dédiée)

2. **Partie 5.2 — Explicabilité des alertes avec SHAP (Credit Card) :**
   * Explication par `shap.TreeExplainer` de l'Isolation Forest, sur **6 observations de test** (2 TP, 2 FP, 2 FN), avec un fond de 50 observations du Train (`random_state=42`).
   * Grandeur expliquée : la longueur moyenne de chemin $h(x)$, de référence $\mathbb{E}[h(X)] = 13.288$ coupes. Les contributions sont inversées pour que la sortie suive l'anomalie : **une contribution positive pousse vers l'alerte**.
   * Résultats : les fraudes détectées sont portées par V10, V14 et V3 ; les fausses alertes par V1, V20, V21, V27 et V28 ; les fraudes manquées ont un signal sur V14, V10 et V17, sans les variables V8 et V7 qui sont fortes chez les TP.
   * Mises en garde : corrélation $\neq$ causalité ; variables ACP anonymisées dont le sens métier n'est pas restaurable ; six cas seulement.

---

## Tableau Récapitulatif Final sur le Jeu de Test (56 746 transactions)

| Modèle | Statut | Prétraitement | Variante | Seuil $\tau^*$ | Précision | Rappel | F1-score | AP | AUC-ROC | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Régression Logistique ($C=0.01$, SMOTE)** | **★ Déployable** | **RobustScaler** | **SMOTE (Train)** | **0.999996** | **87.95 %** | **76.84 %** | **0.8202** | **0.7098** | **0.9677** | **73** | **10** | **22** |
| Arbre de Décision (`depth=5`) | Rejeté | Invariable | Sans correction | 0.0500 | 85.23 % | 78.95 % | 0.8197 | 0.7403 | 0.8789 | 75 | 13 | 20 |
| Isolation Forest | Rejeté | RobustScaler | Non supervisé | 0.6337 | 23.23 % | 37.89 % | 0.2880 | 0.2668 | 0.9590 | 36 | 119 | 59 |
| EasyEnsemble (5 sous-ensembles) | Rejeté (précision test $< 80\%$) | RobustScaler | Supervisé (ensemble) | 0.6640 | 79.31 % | 72.63 % | 0.7582 | 0.7288 | 0.9741 | 69 | 18 | 26 |

---

## Installation et Reproduction

### 1. Cloner le dépôt et installer l'environnement
```bash
git clone <URL_DU_DEPOT>
cd TP-Anomalie-ML
pip install -r requirements.txt
```

### 2. Lancer les notebooks Jupyter
```bash
cd notebook
# Pour la Partie 1 et la Partie 5.1 (Mouse) :
jupyter notebook mouse.ipynb

# Pour la Partie 2, la Partie 3 et la Partie 5.2 (Credit Card & SHAP) :
jupyter notebook partie2et3_fraudes.ipynb
```

---

## Références Bibliographiques
* **Liu, X.-Y., Wu, J., & Zhou, Z.-H. (2009).** *Exploratory undersampling for class-imbalance learning.* IEEE TSMC-B, 39(2), 539–550.
* **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).** *Isolation Forest.* IEEE ICDM.
* **Lundberg, S. M., & Lee, S. I. (2017).** *A unified approach to interpreting model predictions.* NeurIPS (SHAP).
* **Breunig, M. M. et al. (2000).** *LOF: Identifying Density-Based Local Outliers.* ACM SIGMOD.
* **Chawla, N. V. et al. (2002).** *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
