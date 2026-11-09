# Rapport général — Atelier Pratique : Détection d'anomalies

*Master 2 Data Science — Université Claude Bernard Lyon 1*

## 1. Protocole

- **Jeux de données :** Credit Card (283 726 transactions uniques après suppression de 1 081 doublons, 473 fraudes), AI4I 2020 (Partie 4, données synthétiques), Mouse (Parties 1 et 5.1, données 2D pédagogiques, 500 points dont 10 anomalies).
- **Découpage Credit Card / AI4I :** stratifié 60 % Train / 20 % Validation / 20 % Test, `random_state=42`. Sur Credit Card, la variable `Time` est écartée des prédicteurs ; le risque de fuite temporelle d'un découpage aléatoire est discuté. Sur Mouse, ce découpage 60/20/20 n'est **pas** appliqué (protocole spécifique outliers / nouveautés du sujet).
- **Prétraitements ajustés sur le Train uniquement :** `RobustScaler` sur `Amount` (Credit Card) ; `OneHotEncoder` sur `Type` et `StandardScaler` sur les numériques (AI4I).
- **Règle de décision (Credit / AI4I) :** sur la Validation, seuil maximisant le rappel sous contrainte de précision (Credit : $\ge 80\%$ ; AI4I : $\ge 90\%$) et au moins une alerte. Si la contrainte n'est pas atteinte, seuil à F1 maximal, avec le statut « non atteint ». Le seuil est ensuite figé et appliqué au Test sans modification.
- **Garde-fous contre la fuite de données :** suppression des doublons avant partition, scalers ajustés sur le Train, SMOTE appliqué au Train seul, colonnes de types de panne retirées sur AI4I, fond SHAP tiré du Train.

## 2. Partie 1 — Mouse (outliers, seuils, nouveautés)

- **Protocole :** labels construits avant mélange (10 derniers points = anomalies) pour l'évaluation seulement. IF (`contamination=0.02`) et LOF en outliers ; scores ↑ = plus suspect. Trois règles de seuil **sans tuner avec `y`** : défaut, IQR ($Q3+1.5\times\mathrm{IQR}$), clustering KMeans ($k=2$) sur les scores. Classement via AP / AUC-ROC. Nouveautés : train 100 % normal, LOF `novelty=True`, évaluation sur test réservé.
- **Seuils (outliers) :**

| Méthode | Règle de seuil | Nb alertes | F1 anomalie |
|:---|:---|:---:|:---:|
| Isolation Forest | Défaut (`contamination=0.02`) | 10 | **0.90** |
| Isolation Forest | IQR | 21 | 0.65 |
| Isolation Forest | Clustering (KMeans $k=2$) | 115 | 0.16 |
| LOF | Défaut | 10 | **1.00** |
| LOF | IQR | 37 | 0.43 |
| LOF | Clustering (KMeans $k=2$) | 8 | 0.89 |

- **Lecture :** avec `contamination=0.02`, le défaut IF est déjà très bon (9 TP, 1 FP, 1 FN). IQR/clustering ne remplacent pas ce réglage : clustering trop sensible pour IF ; IQR trop large pour LOF. Meilleur F1 outliers : **LOF + défaut**.
- **Classement :** IF AP $\approx 0.96$, AUC $\approx 0.999$ ; LOF AP = 1, AUC = 1. Modifier seulement le seuil change F1 / confusion, **pas** AP/AUC.
- **Nouveautés (LOF) :** train 343 normaux ; test 157 dont 10 anomalies ; F1 $\approx 0.86$, AP $\approx 0.98$ (9 TP, 2 FP, 1 FN).
- **Limite Mouse :** 10 anomalies seulement → conclusions fragiles à 1–2 erreurs ; résultats descriptifs de l'échantillon, pas une estimation future.

## 3. Résultats principaux (Credit Card, Test : 56 746 transactions, 95 fraudes)

| Modèle | Seuil (Validation) | Précision | Rappel | AP | TP | FP | FN | Contrainte $\ge 80\%$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Régression logistique ($C=0.01$, SMOTE sur le Train) — retenue** | **0.999996** | **87.95 %** | **76.84 %** | 0.7098 | 73 | 10 | 22 | **Respectée** |
| Arbre de décision (`max_depth=5`, sans correction) | 0.0500 | 85.23 % | 78.95 % | 0.7403 | 75 | 13 | 20 | Respectée |
| EasyEnsemble (5 sous-ensembles) | 0.6640 | 79.31 % | 72.63 % | 0.7288 | 69 | 18 | 26 | Non respectée |
| Isolation Forest (non supervisé) | 0.6337 | 23.23 % | 37.89 % | 0.2668 | 36 | 119 | 59 | Non respectée |
| LOF (non supervisé) | 2.6744 | 54.70 % | 67.37 % | 0.5031 | 64 | 53 | 31 | Non respectée |

- **Modèle retenu :** la régression logistique avec SMOTE (C=0.01). Sur la Validation, elle atteint la contrainte de précision (0.8276) avec le meilleur rappel (0.766). Le modèle et son seuil sont fixés sur la Validation, puis appliqués au Test.
- **Isolation Forest retenue (hyperparams) :** `n_estimators=100`, `max_samples=1024`, `contamination="auto"` (meilleure AP de validation parmi la grille non supervisée).
- **Seuil :** $\tau = 0.999996$. Ce seuil très proche de 1 s'explique par SMOTE, qui modifie les probabilités prédites. Sur la Validation, au seuil par défaut 0.5, la précision tombe à 5.69 % (rappel 85.11 %) : le seuil par défaut n'est pas utilisable après SMOTE.
- **Lecture sur le Test :** 83 alertes, dont 73 fraudes (précision 87.95 %) et 10 fausses alertes ; 22 fraudes échappent à la détection. La contrainte est respectée sans modification du seuil.
- **Arbre de décision :** sur le Test, il fait un rappel légèrement supérieur (78.95 % contre 76.84 %, soit 75 fraudes contre 73) pour 13 fausses alertes au lieu de 10. L'écart est de deux fraudes sur 95 : il ne remet pas en cause le choix, fait sur la Validation.
- **Partie 2 (variantes) :** SMOTE et la pondération n'améliorent pas l'arbre, dont la précision chute sous 56 % avec ces deux variantes.
- **Non supervisés :** Isolation Forest et LOF ont une AUC-ROC correcte mais une précision insuffisante ; ils ne peuvent pas servir de décision binaire directe.

## 4. Méthode recherchée et sources

- **Méthode testée en Partie 3 :** EasyEnsemble (Liu, Wu & Zhou, 2009), proposée sur les conseils de l'enseignant. Elle entraîne plusieurs AdaBoost, chacun sur un sous-ensemble équilibré formé de toutes les fraudes et d'un échantillon de transactions normales, puis moyenne leurs scores. Elle exploite ainsi la classe majoritaire sans que le déséquilibre domine l'apprentissage.
- **Résultat :** AP de 0.7288 sur le Test, proche de la régression logistique (0.7098), mais précision de 79.31 %, sous la contrainte. Le seuil gelé est rapporté tel quel.
- **Autres références :** Isolation Forest (Liu, Ting & Zhou, 2008), LOF (Breunig et al., 2000), SMOTE (Chawla et al., 2002), SHAP (Lundberg & Lee, 2017).

## 5. Partie 4 — Transfert à la maintenance prédictive (AI4I 2020)

- **Protocole de transfert :** reprendre les modèles retenus sur Credit Card (**hyperparams figés**), les **réajuster** sur AI4I, et **recalibrer** le seuil (précision $\ge 90\%$) sans réutiliser la valeur numérique du seuil fraudes.
  - IF : `n_estimators=100`, `max_samples=1024`, `contamination="auto"` ;
  - LR : `C=0.01` + SMOTE sur le Train ;
  - cible `Machine failure` ; colonnes d'identifiants et de types de panne retirées (fuite) ;
  - évaluation via les fonctions Credit (`trouver_meilleur_seuil`, `evaluer_seuil`) sans les modifier.
- **Validation (choix du seuil) :**

| Méthode | Seuil | Précision | Rappel | TP | FP | FN | AP | AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Isolation Forest | $\approx 0.68$ | 1.00 | 0.029 | 2 | 0 | 66 | 0.14 | 0.77 |
| Régression logistique ($C=0.01$, SMOTE) | $\approx 0.96$ | 0.90 | 0.132 | 9 | 1 | 59 | 0.39 | 0.88 |

- **Test (seuils figés) :**

| Méthode | Précision | Rappel | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|:---:|
| Isolation Forest | 1.00 | 0.015 | 1 | 0 | 67 |
| Régression logistique | 1.00 | 0.029 | 2 | 0 | 66 |

- **Lecture :** LR > IF en ranking (AP/AUC). Sous précision $\ge 90\%$, les seuils deviennent très exigeants → peu d'alertes, rappel très bas, beaucoup de FN (pannes manquées). Sur le Test, la précision reste haute mais le système détecte presque aucune panne. FP = interventions inutiles ; FN = pannes manquées.
- **Conclusion 4.e :** le succès Credit Card ne se transfère pas automatiquement (autre cible, autres variables, ~3.4 % de pannes contre ~0.17 % de fraudes, contrainte plus dure). On ne compare pas directement les AP des deux jeux.

## 6. Partie 5 — Robustesse et explicabilité

- **5.1 (Mouse) :** trains de même taille contaminés à 0 %, 5 % et 10 % par **remplacement** de normaux (`$x=\mu+3\sigma\odot\varepsilon$`) ; test réel fixe ; IF/LOF hyperparams fixes ; 3 graines → moyenne ± écart-type de l'AP.
  - LOF : AP $\approx 0.97$–$0.99$, très stable ;
  - IF : AP plus basse ($\approx 0.75$ à 0 %) puis $\approx 0.85$–$0.86$ à 5–10 %, variance plus forte ;
  - la contamination n'effondre pas les performances avec ce générateur ; les conclusions **dépendent** du générateur synthétique.
- **5.2 (Credit Card, SHAP) :** l'explicateur porte sur la longueur moyenne de chemin $h(x)$ de l'Isolation Forest, et non sur le score final. Après inversion, une contribution positive pousse vers l'alerte. Les fraudes détectées sont portées par V10, V14 et V3 ; les fausses alertes par V1, V20, V21, V27 et V28 ; les fraudes manquées ont un signal sur V14, V10 et V17, sans les variables V8 et V7 qui sont fortes chez les fraudes détectées.

## 7. Contraintes métier atteintes ou non

| Système | Contrainte | Résultat sur le Test | Statut |
|:---|:---|:---|:---|
| Régression logistique, SMOTE (Credit Card) | Précision $\ge 80\%$ | 87.95 % | Atteinte |
| EasyEnsemble (Credit Card) | Précision $\ge 80\%$ | 79.31 % | Non atteinte |
| Isolation Forest, LOF (Credit Card) | Précision $\ge 80\%$ | 23.23 % et 54.70 % | Non atteintes |
| Régression logistique (AI4I) | Précision $\ge 90\%$ | 100 % (rappel 2.9 %) | Atteinte en précision, mais rappel très faible |
| Isolation Forest (AI4I) | Précision $\ge 90\%$ | 100 % (rappel 1.5 %) | Atteinte en précision, mais rappel très faible |

## 8. Limites de généralisation

- **Taille des classes positives :** 95 fraudes au Test et 94 à la Validation ; 10 anomalies seulement sur Mouse ; 68 pannes au Test AI4I. Un écart d'une ou deux observations modifie sensiblement précision et rappel.
- **Un seul découpage :** les résultats Credit / AI4I sont obtenus avec `random_state=42`. Seule la Partie 5.1 mesure la variabilité sur plusieurs graines.
- **Données :** Credit Card est anonymisé (V1 à V28 issues d'une ACP), couvre 48 heures de 2013 et ne permet pas d'affirmer un comportement actuel. AI4I et Mouse sont synthétiques.
- **Découpage aléatoire :** la fuite temporelle est discutée mais non évitée ; un découpage temporel donnerait une estimation plus réaliste.
- **Seuil proche de 1 (Credit) :** le seuil retenu est sensible à SMOTE ; un autre rééchantillonnage ou une autre calibration donnerait un seuil différent.
- **Transfert AI4I :** respecter une précision $\ge 90\%$ peut rendre le système opérationnellement peu utile (presque aucune panne détectée).
- **SHAP :** six cas illustrent les mécanismes du modèle, sans permettre de conclure sur l'ensemble des alertes. Une contribution n'est pas une cause, et les variables anonymisées n'ont pas de sens métier restituable.

## 9. Reproductibilité

- Dépendances : `requirements.txt`.
- Notebooks exécutés : `partie2et3_fraudes.ipynb` (Parties 2, 3 et 5.2), `partie4.ipynb` (Partie 4), `mouse.ipynb` (Parties 1 et 5.1).
- Fonctions réutilisées : `utils_credit.py`, `utils_partie3.py`, `utils_partie5.py`, `utils_mouse.py`, `utils_ai4i.py`.
