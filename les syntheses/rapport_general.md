# Rapport général — Atelier Pratique : Détection d'anomalies

*Master 2 Data Science — Université Claude Bernard Lyon 1*

## 1. Protocole

- **Jeux de données :** Credit Card (283 726 transactions uniques après suppression de 1 081 doublons, 473 fraudes), AI4I 2020 (Partie 4, données synthétiques), Mouse (Parties 1 et 5.1, données 2D pédagogiques).
- **Découpage :** stratifié 60 % Train / 20 % Validation / 20 % Test, `random_state=42`. La variable `Time` est écartée des prédicteurs ; le risque de fuite temporelle d'un découpage aléatoire est discuté.
- **Prétraitements ajustés sur le Train uniquement :** `RobustScaler` sur `Amount` (Credit Card) ; `OneHotEncoder` sur `Type` et `StandardScaler` sur les numériques (AI4I).
- **Règle de décision :** sur la Validation, seuil maximisant le rappel sous contrainte Précision $\ge 80\%$ et au moins une alerte. Si la contrainte n'est pas atteinte, seuil à F1 maximal, avec le statut « non atteint ». Le seuil est ensuite figé et appliqué au Test sans modification.
- **Garde-fous contre la fuite de données :** suppression des doublons avant partition, scalers ajustés sur le Train, SMOTE appliqué au Train seul, fond SHAP tiré du Train.

## 2. Résultats principaux (Credit Card, Test : 56 746 transactions, 95 fraudes)

| Modèle | Seuil (Validation) | Précision | Rappel | AP | TP | FP | FN | Contrainte $\ge 80\%$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Régression logistique ($C=0.01$, SMOTE sur le Train) — retenue** | **0.999996** | **87.95 %** | **76.84 %** | 0.7098 | 73 | 10 | 22 | **Respectée** |
| Arbre de décision (`max_depth=5`, sans correction) | 0.0500 | 85.23 % | 78.95 % | 0.7403 | 75 | 13 | 20 | Respectée |
| EasyEnsemble (5 sous-ensembles) | 0.6640 | 79.31 % | 72.63 % | 0.7288 | 69 | 18 | 26 | Non respectée |
| Isolation Forest (non supervisé) | 0.6337 | 23.23 % | 37.89 % | 0.2668 | 36 | 119 | 59 | Non respectée |
| LOF (non supervisé) | 2.6744 | 54.70 % | 67.37 % | 0.5031 | 64 | 53 | 31 | Non respectée |

- **Modèle retenu :** la régression logistique avec SMOTE (C=0.01). Sur la Validation, elle atteint la contrainte de précision (0.8276) avec le meilleur rappel (0.766). Le modèle et son seuil sont fixés sur la Validation, puis appliqués au Test.
- **Seuil :** $\tau = 0.999996$. Ce seuil très proche de 1 s'explique par SMOTE, qui modifie les probabilités prédites. Sur la Validation, au seuil par défaut 0.5, la précision tombe à 5.69 % (rappel 85.11 %) : le seuil par défaut n'est pas utilisable après SMOTE.
- **Lecture sur le Test :** 83 alertes, dont 73 fraudes (précision 87.95 %) et 10 fausses alertes ; 22 fraudes échappent à la détection. La contrainte est respectée sans modification du seuil.
- **Arbre de décision :** sur le Test, il fait un rappel légèrement supérieur (78.95 % contre 76.84 %, soit 75 fraudes contre 73) pour 13 fausses alertes au lieu de 10. L'écart est de deux fraudes sur 95 : il ne remet pas en cause le choix, fait sur la Validation.
- **Partie 2 (variantes) :** SMOTE et la pondération n'améliorent pas l'arbre, dont la précision chute sous 56 % avec ces deux variantes.
- **Non supervisés :** Isolation Forest et LOF ont une AUC-ROC correcte mais une précision insuffisante ; ils ne peuvent pas servir de décision binaire directe.

## 3. Méthode recherchée et sources

- **Méthode testée en Partie 3 :** EasyEnsemble (Liu, Wu & Zhou, 2009), proposée sur les conseils de l'enseignant. Elle entraîne plusieurs AdaBoost, chacun sur un sous-ensemble équilibré formé de toutes les fraudes et d'un échantillon de transactions normales, puis moyenne leurs scores. Elle exploite ainsi la classe majoritaire sans que le déséquilibre domine l'apprentissage.
- **Résultat :** AP de 0.7288 sur le Test, proche de la régression logistique (0.7098), mais précision de 79.31 %, sous la contrainte. Le seuil gelé est rapporté tel quel.
- **Autres références :** Isolation Forest (Liu, Ting & Zhou, 2008), LOF (Breunig et al., 2000), SMOTE (Chawla et al., 2002), SHAP (Lundberg & Lee, 2017).

## 4. Partie 4 — Transfert à la maintenance prédictive (AI4I 2020)

- **Protocole :** cible `Machine failure`, colonnes d'identifiants et de fuite retirées, IF et régression logistique réajustés sur AI4I, seuil choisi sur la Validation pour une précision $\ge 90\%$, puis figé.
- **Validation :** contrainte atteinte pour les deux modèles, mais avec un rappel faible (régression logistique : 0.15, soit 10 pannes détectées).
- **Test :** la régression logistique a une précision de 0.75 (contrainte non respectée) et un rappel de 0.044 (3 pannes détectées sur 68). IF ne détecte aucune panne sur le Test.
- **Conclusion :** le succès obtenu sur Credit Card ne se transfère pas automatiquement : cible, proportion de positifs (environ 3.4 % contre 0.17 %) et contrainte différentes.

## 5. Partie 5 — Robustesse et explicabilité

- **5.1 (Mouse) :** avec 0 %, 5 % et 10 % de contamination du Train, LOF reste stable (AP $\approx 0.98$), tandis qu'IF gagne en AP mais voit sa variance augmenter (0.748 à 0.863, écart-type jusqu'à 0.095). Les conclusions dépendent du générateur d'anomalies synthétiques.
- **5.2 (Credit Card, SHAP) :** l'explicateur porte sur la longueur moyenne de chemin $h(x)$ de l'Isolation Forest, et non sur le score final. Après inversion, une contribution positive pousse vers l'alerte. Les fraudes détectées sont portées par V10, V14 et V3 ; les fausses alertes par V1, V20, V21, V27 et V28 ; les fraudes manquées ont un signal sur V14, V10 et V17, sans les variables V8 et V7 qui sont fortes chez les fraudes détectées.

## 6. Contraintes métier atteintes ou non

| Système | Contrainte | Résultat sur le Test | Statut |
|:---|:---|:---|:---|
| Régression logistique, SMOTE (Credit Card) | Précision $\ge 80\%$ | 87.95 % | Atteinte |
| EasyEnsemble (Credit Card) | Précision $\ge 80\%$ | 79.31 % | Non atteinte |
| Régression logistique (AI4I) | Précision $\ge 90\%$ | 0.75 | Non atteinte |
| Isolation Forest, LOF (Credit Card) | Précision $\ge 80\%$ | 23.23 % et 54.70 % | Non atteintes |

## 7. Limites de généralisation

- **Taille des classes positives :** 95 fraudes au Test et 94 à la Validation. Un écart d'une ou deux fraudes modifie sensiblement la précision et le rappel : les intervalles de confiance sont larges.
- **Un seul découpage :** les résultats sont obtenus avec `random_state=42`. Seule la Partie 5.1 mesure la variabilité sur plusieurs graines.
- **Données :** Credit Card est anonymisé (V1 à V28 issues d'une ACP), couvre 48 heures de 2013 et ne permet pas d'affirmer un comportement actuel. AI4I et Mouse sont synthétiques.
- **Découpage aléatoire :** la fuite temporelle est discutée mais non évitée ; un découpage temporel donnerait une estimation plus réaliste.
- **Seuil proche de 1 :** le seuil retenu est sensible à SMOTE ; un autre rééchantillonnage ou une autre calibration donnerait un seuil différent.
- **SHAP :** six cas illustrent les mécanismes du modèle, sans permettre de conclure sur l'ensemble des alertes. Une contribution n'est pas une cause, et les variables anonymisées n'ont pas de sens métier restituable.

## 8. Reproductibilité

- Dépendances : `requirements.txt`.
- Notebooks exécutés : `partie2et3_fraudes.ipynb` (Parties 2, 3 et 5.2), `partie4.ipynb` (Partie 4), `mouse.ipynb` (Parties 1 et 5.1).
- Fonctions réutilisées : `utils.py`, `utils_credit.py`, `utils_partie3.py`, `utils_partie5.py`, `utils_mouse.py`.
