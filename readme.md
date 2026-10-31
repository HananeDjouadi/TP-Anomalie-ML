# Atelier Pratique : Détection d'anomalies (Master 2 Data Science)

Projet académique de Machine Learning — Détection d'anomalies, détection de fraudes bancaires et explicabilité des modèles.  
**Enseignants :** Haytham Elghazel & Heidi Souibki  
**Université Claude Bernard Lyon 1** — Année Universitaire 2026-2027

---

## Structure du Projet

```text
TP-Anomalie-ML/
│
├── data/                                      # Jeux de données
│   ├── creditcard.csv                         # Fraudes bancaires (284 807 transactions initiales, 31 colonnes)
│   ├── mouse.txt                              # Données synthétiques 2D (Partie 1)
│   └── ai4i2020.csv                           # Maintenance prédictive industrielle (Partie 4)
│
├── notebook/                                  # Notebooks Jupyter et scripts modulaires
│   ├── partie2et3_fraudes.ipynb               # Notebook complet : Fraudes, Auto-encodeur & SHAP
│   ├── utils_credit.py                        # Fonctions modulaires étanches (prétraitements, seuils, courbes)
│   ├── utils.py                               # Utilitaires généraux
│   └── principal.ipynb                        # Notebook d'assemblage global
│
├── les syntheses/                             # Rapports académiques de synthèse
│   └── Rapport_Partie2_et_3_Detection_Fraudes.md  # Rapport exhaustif des Parties 2, 3 et Explicabilité SHAP
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
   * Constat : Le rééchantillonnage synthétique dégrade la pureté des feuilles de l'arbre et fait chuter la précision sous $56\%$.
3. **Détecteurs non supervisés :**
   * Entraînement strictly non supervisé d'**Isolation Forest** et **Local Outlier Factor (LOF)** avec `novelty=True`.
   * Démonstration de l'échec opérationnel du non-supervisé : malgré un AUC-ROC flatteur ($0.942$), l'Average Precision s'effondre à $0.086$ (confusion entre atypisme statistique et criminalité).
4. **Optimisation sous contrainte opérationnelle métier ($\text{Précision} \ge 80\%$) :**
   * Détermination du seuil optimal sur la Validation : passage du seuil par défaut $0.50$ (49 fraudes détectées, 45 manquées) au seuil calibré $\tau^* = 0.0756$ (72 fraudes détectées, seulement 22 manquées).
   * **Évaluation finale gelée sur l'ensemble de Test (56 746 transactions) :**
     * **Précision :** $80.22\%$ (contrainte des $80\%$ strictement respectée)
     * **Rappel :** $76.84\%$ (73 fraudes interceptées sur 95)
     * **Fausses alertes :** seulement 18 alertes indues ($0.03\%$ du flux)
     * **Temps de calcul :** $< 1\text{ seconde}$

---

### Partie 3 : Recherche d'une Nouvelle Méthode — L'Auto-encodeur Neuronal (MLP)
* **Méthode retenue :** Auto-encodeur non linéaire par Réseau de Neurones Artificiels (*Multilayer Perceptron Autoencoder* via `MLPRegressor`).
* **Publication académique officielle :** *Sakurada & Yairi (2014)*, IEEE/ACM MLSDA.
* **Score d'anomalie :** Erreur Quadratique Moyenne de reconstruction $\text{MSE}(x, \hat{x}) = \frac{1}{D} \|x - \hat{x}\|_2^2$, naturellement orienté positivement.
* **Hypothèse de la variété (*Manifold Hypothesis*) :** Les transactions normales résident sur une sous-variété non linéaire de dimension réduite $d \ll D$. Les fraudes s'en écartent et subissent une forte distorsion de reconstruction.
* **Résultats empiriques :** La configuration à goulot étroit `(12, 4, 12)` ($\text{AP} = 0.0380$) divise par deux les fausses alertes (181 contre 398) par rapport à la structure standard, mais plafonne à $9.50\%$ de précision maximale.
* **Analyse comparative :**
  * *Points forts :* Variété continue évitant les découpages orthogonaux d'Isolation Forest et immunité au fléau de la dimension par rapport à LOF.
  * *Complexité :* Scalabilité linéaire $O(N)$ en $15.1\text{ s}$ et inférence ultra-rapide ($< 0.1\text{ ms}$).

---

### Partie Spéciale : Explicabilité des Alertes avec SHAP sur Isolation Forest
* **Protocole :** Explication par `shap.TreeExplainer` de **6 observations de test** (2 Vrais Positifs, 2 Faux Positifs, 2 Faux Négatifs) avec un *background dataset* de 50 observations issues strictement du Train.
* **Identification de la grandeur interne expliquée :** L'outil explique la **longueur de chemin moyenne $f(x)$** (la profondeur dans l'arbre) :
  * Valeur attendue normale : $\mathbb{E}[f(X)] = 13.288$ coupes.
  * Fraudes détectées ($TP$) : $f(x)$ s'effondre à **$5.80$** et **$4.05$** coupes (les valeurs extrêmes de $V_8, V_3, V_1, V_5$ raccourcissent le chemin $\implies$ contributions SHAP négatives $\phi_j < 0$).
  * Fausses alertes ($FP$) : $f(x)$ chute à **$6.20$** et **$5.76$** coupes sous l'effet de composantes marginales ($V_1, V_2, V_8$), prouvant l'erreur d'aiguillage du modèle non supervisé sur des clients honnêtes atypiques.
  * Fraudes manquées ($FN$) : $f(x) > 11.8$ coupes (fraudes mimétiques au profil médian).
* **Mise en garde épistémologique :**
  1. Une contribution SHAP forte est une **corrélation exploitée par l'algorithme, pas une preuve de causalité**.
  2. Les variables $V_1 \dots V_{28}$ étant issues d'une ACP anonymisée, **SHAP n'en restaure pas la signification métier**.

---

## Tableau Récapitulatif Final sur le Jeu de Test (56 746 transactions)

| Modèle | Statut | Prétraitement | Variante | Seuil $\tau^*$ | Précision | Rappel | F1-score | AP | AUC-ROC | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Régression Logistique ($C=1.0$)** | **★ Déployable** | **RobustScaler** | **Sans correction** | **0.0756** | **80.22 %** | **76.84 %** | **0.7849** | **0.7417** | **0.9706** | **73** | **18** | **22** |
| Arbre de Décision (`depth=5`) | Rejeté | Invariable | Sans correction | 0.0500 | 85.23 % | 78.95 % | 0.8197 | 0.7403 | 0.8789 | 75 | 13 | 20 |
| Isolation Forest | Rejeté | RobustScaler | Non supervisé | 0.6569 | 21.43 % | 25.26 % | 0.2319 | 0.1482 | 0.9546 | 24 | 88 | 71 |
| Auto-encodeur (`12-4-12`) | Rejeté | RobustScaler | Non supervisé | 4.9973 | 9.50 % | 20.21 % | 0.1293 | 0.0380 | 0.9092 | 19 | 181 | 75 |

---

## Installation et Reproduction

### 1. Cloner le dépôt et installer l'environnement
```bash
git clone <URL_DU_DEPOT>
cd TP-Anomalie-ML
pip install -r requirements.txt
```

### 2. Lancer le notebook Jupyter
```bash
cd notebook
jupyter notebook partie2et3_fraudes.ipynb
```

---

## Références Bibliographiques
* **Sakurada, M., & Yairi, T. (2014).** *Anomaly detection using autoencoders with nonlinear dimensionality reduction.* IEEE/ACM MLSDA.
* **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).** *Isolation Forest.* IEEE ICDM.
* **Lundberg, S. M., & Lee, S. I. (2017).** *A unified approach to interpreting model predictions.* NeurIPS (SHAP).
* **Chawla, N. V. et al. (2002).** *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
