# Synthèse — Atelier Détection d’anomalies

**Module :** Machine Learning (M2 DS)  
**Périmètre actuel :** Partie 1 — jeu de données Mouse  
*(Les parties Credit Card, méthode hors cours, AI4I et SHAP seront complétées au fur et à mesure.)*

---

## 1. Protocole (Partie Mouse)

- Chargement de `mouse.txt` (500 points, 2 variables).
- Construction des labels **avant tout mélange** : les 10 derniers points = anomalies (classe 1). Les labels servent **uniquement à l’évaluation** pour les méthodes non supervisées.
- Méthodes : **Isolation Forest** et **LOF** en détection d’outliers sur l’ensemble Mouse (`novelty=False` pour LOF).
- Scores harmonisés : plus le score est élevé, plus le point est suspect (inversion des scores sklearn si besoin).
- Deux règles de seuil comparées **sans tuner avec les labels** :
  - seuil **par défaut** de chaque modèle ;
  - seuil **IQR** : `Q3 + 1.5 × IQR` (heuristique de Tukey, `k=1.5` fixé a priori).
- Évaluation binaire : matrice de confusion et **F1** (mesure principale après seuil).
- Évaluation de classement : **AP** et **AUC-ROC** sur les scores continus.
- Détection de **nouveautés** : train = normaux uniquement ; test = normaux réservés + toutes les anomalies ; LOF avec `novelty=True`, évaluation sur le test seulement.
- Reproductibilité : graines aléatoires fixées ; code factorisé dans `utils_mouse.py`.

Le protocole commun 60/20/20 (Credit Card / AI4I) n’a **pas** été appliqué sur Mouse, conformément au sujet.

---

## 2. Résultats principaux (Mouse)

### 2.1 Seuils (outliers)

| Méthode | Règle de seuil | Nb alertes | F1 |
|---------|----------------|------------|-----|
| Isolation Forest | Défaut | 125 | 0.15 |
| Isolation Forest | IQR | 21 | 0.65 |
| LOF | Défaut | 10 | 1.00 |
| LOF | IQR | 37 | 0.43 |

- IF avec seuil défaut est trop sensible (beaucoup de bords de clusters).
- L’IQR améliore nettement IF, mais dégrade LOF ici.
- Meilleur F1 outliers : **LOF + seuil défaut** (sans utilisation des labels à l’apprentissage).

### 2.2 Classement (AP / AUC-ROC)

| Méthode | AP | AUC-ROC |
|---------|----|---------|
| Isolation Forest | ≈ 0.96 | ≈ 0.999 |
| LOF | 1.00 | 1.00 |

Modifier seulement le seuil ne change pas AP/AUC : ces mesures dépendent des **scores** et de leur ordre, pas de la coupure 0/1.

### 2.3 Nouveautés (LOF, `novelty=True`)

| Mesure | Valeur |
|--------|--------|
| Train | 343 normaux (0 anomalie) |
| Test | 157 points dont 10 anomalies |
| F1 | 0.86 |
| AP | 0.98 |
| AUC-ROC | 0.998 |

Bon transfert sur le test réservé (9 TP, 2 FP, 1 FN).

---

## 3. Interprétation et désaccords IF / LOF

- **IF** isole les points faciles à séparer → marque souvent les **périphéries** des groupes.
- **LOF** compare les **densités locales** → comportement plus sélectif sur Mouse.
- Les désaccords sont attendus : deux définitions différentes d’“anormal”.
- Un bon ranking (AP/AUC élevés) n’implique pas un bon F1 si le seuil est mal calibré (cas d’IF défaut).

---

## 4. Contraintes métier / méthode recherchée

- **Contraintes opérationnelles** (précision ≥ 80 % fraude, ≥ 90 % maintenance) : **non concernées** sur Mouse ; à traiter en parties Credit Card et AI4I.
- **Méthode hors cours** : **pas encore réalisée** (Partie 3).

---

## 5. Limites de généralisation

- Seulement **10 anomalies** : F1 et conclusions très sensibles à 1–2 erreurs.
- Résultats outliers = détection **dans l’échantillon observé**, pas une estimation sur données futures.
- L’IQR est une heuristique : elle n’est pas toujours meilleure (contre-exemple LOF).
- Utiliser les labels pour construire un train 100 % normal (nouveautés) change l’information disponible : ce n’est plus le même problème que la détection d’outliers “à l’aveugle”.
- Un score élevé obtenu avec une **fuite de labels** dans l’apprentissage ne serait pas valide ; ici, IF/LOF outliers n’utilisent pas `y` pour s’ajuster.

---

## 6. Livrables associés (état actuel)

- Notebook Mouse exécuté + figures détaillées
- `utils_mouse.py` (chargement, IF/LOF, seuils, métriques, nouveautés)
- Cette synthèse (Partie 1) — à étendre aux parties 2 à 5
