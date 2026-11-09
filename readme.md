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
│   ├── creditcard.csv                         # Fraudes bancaires (Parties 2, 3, 5.2)
│   ├── mouse.txt                              # Données synthétiques 2D (Parties 1 et 5.1)
│   └── ai4i2020.csv                           # Maintenance prédictive (Partie 4)
│
├── notebook/                                  # Notebooks Jupyter
│   ├── mouse.ipynb                            # Parties 1 et 5.1 (Mouse)
│   ├── partie2et3_fraudes.ipynb               # Parties 2, 3 et 5.2 (Credit Card + SHAP)
│   └── partie4.ipynb                          # Partie 4 (AI4I)
│
├── utils_files/                               # Fonctions factorisées
│   ├── utils_mouse.py                         # Partie 1 (+ helpers Mouse)
│   ├── utils_credit.py                        # Parties 2 / 3 (seuils réutilisés en 4)
│   ├── utils_partie3.py                       # Partie 3 (EasyEnsemble)
│   ├── utils_ai4i.py                          # Partie 4
│   ├── utils_partie5.py                       # Parties 5.1 et 5.2
│   └── __init__.py
│
├── les syntheses/                             # Rapports de synthèse
│   └── rapport_general.md                     # Rapport général (toutes les parties)
│
├── requirements.txt
└── README.md
```

---

## Travaux Réalisés et Synthèse Méthodologique

### Partie 1 : Données Synthétiques 2D (`mouse.txt`)
* **Notebook :** `mouse.ipynb` — **Utils :** `utils_mouse.py`
* Labels : 10 derniers points = anomalies (évaluation uniquement).
* **Outliers :** Isolation Forest (`contamination=0.02`) et LOF ; scores ↑ = plus suspect.
* **Trois règles de seuil sans labels :** défaut, IQR ($Q3 + 1.5\times\mathrm{IQR}$), clustering KMeans ($k=2$) sur les scores.
* **Résultats (F1) :** IF défaut **0.90** (10 alertes) ; IF IQR 0.65 ; IF clustering 0.16 ; LOF défaut **1.00** ; LOF IQR 0.43 ; LOF clustering 0.89.
* **Classement :** IF AP $\approx 0.96$, AUC $\approx 0.999$ ; LOF AP/AUC = 1. Modifier le seuil change F1, **pas** AP/AUC.
* **Nouveautés :** train 343 normaux ; LOF `novelty=True` ; test 157 (10 anomalies) ; F1 $\approx 0.86$, AP $\approx 0.98$ (9 TP, 2 FP, 1 FN).
* **Lecture :** meilleur F1 outliers = LOF + défaut ; avec `contamination=0.02`, IF défaut est déjà bon. Fragilité : seulement 10 anomalies.

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

### Partie 4 : Transfert à la maintenance (`ai4i2020.csv`)
* **Notebook :** `partie4.ipynb` — **Utils :** `utils_ai4i.py` (seuils via `utils_credit.py`)
* **Protocole de transfert :** reprendre IF et LR retenus sur Credit Card (**hyperparams figés**), les réajuster sur AI4I, recalibrer le seuil (précision $\ge 90\%$) sans réutiliser le seuil numérique fraudes.
  * IF : `n_estimators=100`, `max_samples=1024`
  * LR : `C=0.01` + SMOTE sur le Train
* Cible `Machine failure` ; retrait des ids et colonnes de types de panne (fuite) ; split 60/20/20 ; OneHot(`Type`) + StandardScaler (fit train only).
* **Validation :** contrainte 90 % atteinte ; rappel faible (IF $\approx 0.03$, LR $\approx 0.13$). LR > IF (AP/AUC).
* **Test (seuils figés) :** précision 1.00 des deux côtés, mais rappel très bas (IF : 1 TP / 67 FN ; LR : 2 TP / 66 FN).
* **Conclusion :** le succès fraudes ne se transfère pas automatiquement (autre problème, ~3.4 % de pannes vs ~0.17 % de fraudes, contrainte plus dure).

---

### Partie 5 : Robustesse et Explicabilité (Synthèse Dédiée)

1. **Partie 5.1 — Contaminer la référence normale (Mouse) :**
   * **Notebook :** `mouse.ipynb` — **Utils :** `utils_partie5.py`
   * Trains de même taille à 0 / 5 / 10 % d'anomalies synthétiques ($\mu + 3\sigma$) par **remplacement** ; test réel fixe ; IF/LOF hyperparams fixes ; 3 graines → moyenne ± écart-type de l'AP.
   * **Résultat :** LOF très stable (AP $\approx 0.97$–$0.99$) ; IF plus bas et plus variable ($\approx 0.75$ à 0 %, puis $\approx 0.85$–$0.86$). Conclusions dépendantes du générateur.

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

# Pour la Partie 4 (AI4I / maintenance) :
jupyter notebook partie4.ipynb
```

---

## Références Bibliographiques
* **Liu, X.-Y., Wu, J., & Zhou, Z.-H. (2009).** *Exploratory undersampling for class-imbalance learning.* IEEE TSMC-B, 39(2), 539–550.
* **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).** *Isolation Forest.* IEEE ICDM.
* **Lundberg, S. M., & Lee, S. I. (2017).** *A unified approach to interpreting model predictions.* NeurIPS (SHAP).
* **Breunig, M. M. et al. (2000).** *LOF: Identifying Density-Based Local Outliers.* ACM SIGMOD.
* **Chawla, N. V. et al. (2002).** *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
