# Synthèse — Partie 4 : Transfert à la maintenance (AI4I 2020)

**Module :** Machine Learning (M2 DS)  
**Périmètre :** Partie 4 — jeu AI4I Predictive Maintenance

---

## 1. Protocole

- Cible : **`Machine failure`** (1 = panne, 0 = normal).
- Colonnes retirées (ids / fuite) : `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
- Split stratifié reproductible : **60 % train / 20 % val / 20 % test**.
- Prétraitement (fit **train only**) :
  - `Type` → OneHotEncoder ;
  - variables numériques → StandardScaler.
- Modèles repris de Credit Card (**hyperparams figés**, modèles **réajustés** sur AI4I) :
  - Isolation Forest (`n_estimators=100`, `contamination="auto"`) ;
  - Régression logistique (`C=1.0`, sans correction du déséquilibre).
- Seuil choisi sur la **validation** : maximiser le rappel sous **précision ≥ 90 %**, puis figer avant le test.
- Réutilisation des fonctions de `utils_credit` (seuil, métriques, courbes) **sans les modifier** ; préparation AI4I dans `utils_ai4i.py`.

Les observations AI4I sont **synthétiques**.

---

## 2. Résultats principaux

### 2.1 Validation (choix du seuil, précision ≥ 90 %)

| Méthode | Seuil | Précision | Rappel | TP | FP | FN | AP | AUC-ROC |
|---------|-------|-----------|--------|----|----|----|----|---------|
| Isolation Forest | ≈ 0.67 | 1.00 | 0.015 | 1 | 0 | 67 | 0.11 | 0.75 |
| Logistic Regression | ≈ 0.68 | 0.91 | 0.15 | 10 | 1 | 58 | 0.43 | 0.87 |

- Contrainte 90 % **atteinte** pour les deux sur la validation.
- Pour la respecter, les seuils sont très exigeants → **rappel faible**, beaucoup de pannes manquées.
- **LR > IF** (meilleur AP/AUC et meilleur rappel).

### 2.2 Test (seuils figés, non retouchés)

| Méthode | Précision | Rappel | TP | FP | FN |
|---------|-----------|--------|----|----|----|
| Isolation Forest | 0.00 | 0.00 | 0 | 1 | 68 |
| Logistic Regression | 0.75 | 0.044 | 3 | 1 | 65 |

- **FP** = interventions inutiles ; **FN** = pannes manquées.
- Sur le test, LR ne conserve plus la précision ≥ 90 % (0.75) : constat signalé **sans modifier le seuil après coup**.

---

## 3. Interprétation (4.e)

Le modèle gagnant sur les fraudes (LR) reste meilleur qu’IF sur AI4I, mais le succès métier **ne se transfère pas automatiquement** :
- cible et variables différentes ;
- proportion de positifs différente (~3.4 % vs ~0.17 %) ;
- contrainte plus dure (90 % vs 80 %) ;
- chaque alerte maintenance est coûteuse → peu d’alertes, peu de rappel.

On ne compare **pas** directement les AP Credit Card vs AI4I comme si la difficulté était la même.

---

## 4. Limites

- Données synthétiques → conclusions pédagogiques.
- Sous précision ≥ 90 %, le système détecte très peu de pannes.
- Un bon protocole de transfert ≠ bonnes performances opérationnelles garanties.

---

## 5. Livrables associés

- `notebook/partie4.ipynb`
- `notebook/utils_ai4i.py`
- fonctions d’évaluation réutilisées depuis `notebook/utils_credit.py`
