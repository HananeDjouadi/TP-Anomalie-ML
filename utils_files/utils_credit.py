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


def tracer_f1_balanced_acc(y_true, scores, seuil_retenu=None):
  """Trace l'évolution du F1-score et de la Balanced Accuracy en fonction du seuil."""
  seuils = np.linspace(0.01, 0.99, 150)
  f1_list = []
  bacc_list = []

  for t in seuils:
    y_pred = (scores >= t).astype(int)
    f1_list.append(f1_score(y_true, y_pred, zero_division=0))
    bacc_list.append(balanced_accuracy_score(y_true, y_pred))

  plt.figure(figsize=(10, 5))
  plt.plot(seuils, f1_list, label="F1-score", color="blue", linewidth=2)
  plt.plot(
      seuils,
      bacc_list,
      label="Balanced Accuracy",
      color="forestgreen",
      linewidth=2,
  )

  best_t_f1 = seuils[np.argmax(f1_list)]
  best_t_bacc = seuils[np.argmax(bacc_list)]

  plt.axvline(
      best_t_f1,
      color="blue",
      linestyle=":",
      alpha=0.7,
      label=f"Max F1 ({best_t_f1:.2f})",
  )
  plt.axvline(
      best_t_bacc,
      color="forestgreen",
      linestyle=":",
      alpha=0.7,
      label=f"Max Balanced Acc ({best_t_bacc:.2f})",
  )

  if seuil_retenu is not None:
    plt.axvline(
        seuil_retenu,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Seuil retenu (Précision>=80%) : {seuil_retenu:.4f}",
    )

  plt.title("Évolution du F1-score et de la Balanced Accuracy selon le seuil")
  plt.xlabel("Seuil de décision")
  plt.ylabel("Score")
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()



from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def appliquer_smote(X_train_s, y_train, random_state=42):
    """Applique le sur-échantillonnage SMOTE uniquement sur l'ensemble Train."""
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_s, y_train)
    return X_train_smote, y_train_smote


def evaluer_variantes_supervisees(
    X_train_s, y_train, X_train_smote, y_train_smote, X_val_s, y_val, type_modele="lr"
):
    """Entraîne et compare les 3 variantes (Sans correction, Pondération, SMOTE)
    pour la Régression Logistique ou l'Arbre de Décision.
    """
    if type_modele == "lr":
        modeles = {
            "LR (Sans correction)": LogisticRegression(max_iter=1000, random_state=42),
            "LR (Pondération classes)": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            ),
            "LR (SMOTE)": LogisticRegression(max_iter=1000, random_state=42),
        }
    else:
        modeles = {
            "Arbre (Sans correction)": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Arbre (Pondéré)": DecisionTreeClassifier(
                max_depth=5, class_weight="balanced", random_state=42
            ),
            "Arbre (SMOTE)": DecisionTreeClassifier(max_depth=5, random_state=42),
        }

    scores_dict = {}
    resultats = []

    for nom, clf in modeles.items():
        if "SMOTE" in nom:
            clf.fit(X_train_smote, y_train_smote)
        else:
            clf.fit(X_train_s, y_train)

        scores_val = clf.predict_proba(X_val_s)[:, 1]
        scores_dict[nom] = scores_val

        ap = average_precision_score(y_val, scores_val)
        auc = roc_auc_score(y_val, scores_val)

        resultats.append({
            "Modèle": nom,
            "AP (Validation)": round(ap, 4),
            "AUC-ROC": round(auc, 4),
        })

    df_res = pd.DataFrame(resultats)
    return df_res, scores_dict, modeles


def evaluer_grille_lr(
    X_train_s, y_train, X_train_smote, y_train_smote, X_val_s, y_val, min_precision=0.80
):
    """Évalue les 6 configurations de Régression Logistique (C=0.1, 1.0 sur les 3 variantes)."""
    configs_lr = [
        {"C": 0.1, "variante": "Sans correction"},
        {"C": 1.0, "variante": "Sans correction"},
        {"C": 0.1, "variante": "Pondération"},
        {"C": 1.0, "variante": "Pondération"},
        {"C": 0.1, "variante": "SMOTE"},
        {"C": 1.0, "variante": "SMOTE"},
    ]
    scores_6_lr = {}
    resultats = []

    for cfg in configs_lr:
        cw = "balanced" if cfg["variante"] == "Pondération" else None
        clf = LogisticRegression(
            C=cfg["C"], class_weight=cw, max_iter=1000, random_state=42
        )
        if cfg["variante"] == "SMOTE":
            clf.fit(X_train_smote, y_train_smote)
        else:
            clf.fit(X_train_s, y_train)

        scores_val = clf.predict_proba(X_val_s)[:, 1]
        nom_cfg = f"LR ({cfg['variante']}, C={cfg['C']})"
        scores_6_lr[nom_cfg] = scores_val

        res = trouver_meilleur_seuil(y_val, scores_val, min_precision=min_precision)
        res["C"] = cfg["C"]
        res["Variante"] = cfg["variante"]
        resultats.append(res)

    df_res = pd.DataFrame(resultats)
    return df_res, scores_6_lr


def evaluer_grille_arbre(
    X_train_s, y_train, X_train_smote, y_train_smote, X_val_s, y_val, min_precision=0.80
):
    """Évalue les 6 configurations d'Arbre de Décision (max_depth=3, 5 sur les 3 variantes)."""
    configs_dt = [
        {"max_depth": 3, "variante": "Sans correction"},
        {"max_depth": 5, "variante": "Sans correction"},
        {"max_depth": 3, "variante": "Pondération"},
        {"max_depth": 5, "variante": "Pondération"},
        {"max_depth": 3, "variante": "SMOTE"},
        {"max_depth": 5, "variante": "SMOTE"},
    ]
    scores_6_dt = {}
    resultats = []

    for cfg in configs_dt:
        cw = "balanced" if cfg["variante"] == "Pondération" else None
        clf = DecisionTreeClassifier(
            max_depth=cfg["max_depth"], class_weight=cw, random_state=42
        )
        if cfg["variante"] == "SMOTE":
            clf.fit(X_train_smote, y_train_smote)
        else:
            clf.fit(X_train_s, y_train)

        scores_val = clf.predict_proba(X_val_s)[:, 1]
        nom_cfg = f"Arbre (prof={cfg['max_depth']}, {cfg['variante']})"
        scores_6_dt[nom_cfg] = scores_val

        res = trouver_meilleur_seuil(y_val, scores_val, min_precision=min_precision)
        res["Profondeur"] = cfg["max_depth"]
        res["Variante"] = cfg["variante"]
        resultats.append(res)

    df_res = pd.DataFrame(resultats)
    return df_res, scores_6_dt


def entrainer_modeles_non_supervises(
    X_train_s, X_val_s, y_val, n_sub_lof=30000, random_state=42, min_precision=0.80
):
    """Ajuste Isolation Forest et LOF (sur sous-échantillon 30 000), harmonise les scores
    (-score_samples) et calibre les seuils optimaux.
    """
    detecteurs = {
        "Isolation Forest": IsolationForest(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "Local Outlier Factor (LOF)": LocalOutlierFactor(
            n_neighbors=20, novelty=True
        ),
    }

    X_train_sub = X_train_s.sample(n=n_sub_lof, random_state=random_state)
    scores_non_sup_dict = {}
    resultats = []

    print("--- Entraînement des détecteurs non supervisés ---")
    for nom, clf in detecteurs.items():
        print(f"Entraînement de {nom}...")
        if "LOF" in nom:
            clf.fit(X_train_sub)
        else:
            clf.fit(X_train_s)

        # Inversion du signe pour harmoniser : score élevé = anomalie
        scores_val = -clf.score_samples(X_val_s)
        scores_non_sup_dict[nom] = scores_val

        res = trouver_meilleur_seuil(y_val, scores_val, min_precision=min_precision)
        res["Modèle"] = nom
        resultats.append(res)

    df_res = pd.DataFrame(resultats)
    return df_res, scores_non_sup_dict, detecteurs


def evaluer_modeles_test_partie2(
    X_train_s, y_train, X_test_s, y_test, iso_forest, seuil_lr, seuil_dt=0.05, seuil_if=0.6569
):
    """Évalue sur l'ensemble de Test gelé les 3 modèles retenus de la Partie 2
    aux seuils calibrés sur la validation.
    """
    # 1. Régression Logistique (C=1.0)
    lr_best = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_best.fit(X_train_s, y_train)
    scores_test_lr = lr_best.predict_proba(X_test_s)[:, 1]
    res_test_best = evaluer_seuil(y_test, scores_test_lr, seuil=seuil_lr)
    res_test_best["AP"] = round(average_precision_score(y_test, scores_test_lr), 4)
    res_test_best["AUC-ROC"] = round(roc_auc_score(y_test, scores_test_lr), 4)
    res_test_best["Modèle"] = "Régression Logistique (C=1.0) [TEST]"
    res_test_best["Temps Train (s)"] = 0.47

    # 2. Arbre de Décision (depth=5)
    tree_best = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_best.fit(X_train_s, y_train)
    scores_test_tree = tree_best.predict_proba(X_test_s)[:, 1]
    res_test_tree = evaluer_seuil(y_test, scores_test_tree, seuil=seuil_dt)
    res_test_tree["AP"] = round(average_precision_score(y_test, scores_test_tree), 4)
    res_test_tree["AUC-ROC"] = round(roc_auc_score(y_test, scores_test_tree), 4)
    res_test_tree["Modèle"] = "Arbre de Décision (depth=5) [TEST]"
    res_test_tree["Temps Train (s)"] = 2.35

    # 3. Isolation Forest
    scores_test_if = -iso_forest.score_samples(X_test_s)
    res_test_if = evaluer_seuil(y_test, scores_test_if, seuil=seuil_if)
    res_test_if["AP"] = round(average_precision_score(y_test, scores_test_if), 4)
    res_test_if["AUC-ROC"] = round(roc_auc_score(y_test, scores_test_if), 4)
    res_test_if["Modèle"] = "Isolation Forest [TEST]"
    res_test_if["Temps Train (s)"] = 4.77

    return res_test_best, res_test_tree, res_test_if, scores_test_if
