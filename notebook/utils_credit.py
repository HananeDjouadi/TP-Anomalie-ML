import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_clean_data(filepath="../data/creditcard.csv"):
    """Charge le fichier CSV et supprime les doublons."""
    df = pd.read_csv(filepath)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_data(df, random_state=42):
    """Découpage stratifié 60% Train, 20% Validation, 20% Test.

    Écarte 'Time' et 'Class' des variables prédictives X.
    """
    X = df.drop(columns=["Time", "Class"])
    y = df["Class"]

    # 1. 60% Train et 40% Reste
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=random_state
    )

    # 2. Séparation du reste en 20% Val et 20% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_amount(X_train, X_val, X_test):
    """Met à l'échelle 'Amount' avec RobustScaler (médiane/IQR).

    Ajusté uniquement sur le Train pour éviter toute fuite de données.
    """
    X_train_s = X_train.copy()
    X_val_s = X_val.copy()
    X_test_s = X_test.copy()

    scaler = RobustScaler()
    X_train_s["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_val_s["Amount"] = scaler.transform(X_val[["Amount"]])
    X_test_s["Amount"] = scaler.transform(X_test[["Amount"]])

    return X_train_s, X_val_s, X_test_s


from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)


def evaluer_seuil(y_true, scores, seuil=0.5):
  """Calcule la matrice de confusion, Précision, Rappel, F1 et Balanced Accuracy pour un seuil donné."""
  y_pred = (scores >= seuil).astype(int)
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

  return {
      "Seuil": seuil,
      "TP (Fraudes détectées)": tp,
      "FP (Fausses alertes)": fp,
      "FN (Fraudes manquées)": fn,
      "TN (Normales correctes)": tn,
      "Précision": round(precision_score(y_true, y_pred, zero_division=0), 4),
      "Rappel": round(recall_score(y_true, y_pred, zero_division=0), 4),
      "F1-score": round(f1_score(y_true, y_pred, zero_division=0), 4),
      "Balanced Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
  }

import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

def tracer_courbes_roc_pr(y_true, modeles_scores, min_precision=0.80):
  """Trace les courbes ROC et Précision-Rappel pour un ou plusieurs modèles.

  modeles_scores : dictionnaire {"Nom Modèle": vecteur_de_scores}
  """
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  for nom, scores in modeles_scores.items():
    # 1. Courbe ROC
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    ax1.plot(fpr, tpr, label=f"{nom} (AUC = {auc:.3f})")

    # 2. Courbe Précision-Rappel
    prec, rec, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    ax2.plot(rec, prec, label=f"{nom} (AP = {ap:.3f})")

  # Style ROC
  ax1.plot([0, 1], [0, 1], "k--", alpha=0.6)
  ax1.set_title("Courbe ROC")
  ax1.set_xlabel("Taux de faux positifs (FPR)")
  ax1.set_ylabel("Rappel (TPR)")
  ax1.legend()

  # Style Précision-Rappel
  if min_precision:
    ax2.axhline(
        y=min_precision,
        color="r",
        linestyle="--",
        label=f"Contrainte Précision >= {int(min_precision*100)}%",
    )
  ax2.set_title("Courbe Précision-Rappel")
  ax2.set_xlabel("Rappel")
  ax2.set_ylabel("Précision")
  ax2.legend()

  plt.tight_layout()
  plt.show()




def trouver_meilleur_seuil(y_true, scores, min_precision=0.80):
  """Trouve le seuil qui maximise le rappel avec Précision >= 80% (au moins 1 alerte).

  Calcule toutes les métriques imposées par le sujet.
  """
  precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
  ap = average_precision_score(y_true, scores)
  auc = roc_auc_score(y_true, scores)

  best_thresh = None
  best_rec = -1.0
  best_prec = -1.0

  # Recherche du seuil optimal sous contrainte
  for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
    if np.sum(scores >= t) >= 1 and p >= min_precision:
      if r > best_rec:
        best_rec = r
        best_prec = p
        best_thresh = t
      elif np.isclose(r, best_rec) and p > best_prec:
        best_prec = p
        best_thresh = t

  # Si la contrainte n'est pas atteinte : meilleur F1 possible
  if best_thresh is None:
    f1s = (
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    )
    best_thresh = thresholds[np.argmax(f1s)]
    statut = "Non atteint"
  else:
    statut = "Atteint"

  # Calcul des métriques au seuil trouvé
  metriques = evaluer_seuil(y_true, scores, seuil=best_thresh)
  metriques["AP"] = round(ap, 4)
  metriques["AUC-ROC"] = round(auc, 4)
  metriques["Contrainte (>=80%)"] = statut

  return metriques
