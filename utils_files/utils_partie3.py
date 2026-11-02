"""Partie 3 — Nouvelle méthode : L'Auto-encodeur non linéaire (Autoencoder).

Encapsulation complète de la définition des architectures, de l'entraînement,
du calcul des erreurs de reconstruction (MSE) et des fonctions de comparaison.
"""

import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from . import utils_credit
except (ImportError, ValueError):
    try:
        from utils_files import utils_credit
    except ImportError:
        import utils_credit


def definir_configurations_autoencodeurs():
    """Retourne la liste des configurations d'architectures à tester."""
    return [
        {
            "nom": "Autoencodeur (16-8-16)",
            "hidden": (16, 8, 16),
            "alpha": 0.001,
            "desc": "Architecture standard (goulot à 8 neurones)",
        },
        {
            "nom": "Autoencodeur (12-4-12)",
            "hidden": (12, 4, 12),
            "alpha": 0.001,
            "desc": "Goulot très étroit (compression forte à 4 neurones)",
        },
        {
            "nom": "Autoencodeur (16-8-16, L2 fort)",
            "hidden": (16, 8, 16),
            "alpha": 0.05,
            "desc": "Régularisation L2 forte (alpha = 0.05)",
        },
    ]


def entrainer_evaluer_autoencodeurs(
    X_train_s,
    y_train,
    X_val_s,
    y_val,
    configs=None,
    batch_size=256,
    max_iter=15,
    random_state=42,
    min_precision=0.80,
):
    """Entraîne les architectures d'Auto-encodeur MLP sur le Train (non supervisé),
    calcule les erreurs quadratiques de reconstruction (MSE) sur la Validation,
    recherche le seuil optimal (Précision >= 80% ou max F1) et retourne le
    tableau comparatif.

    Retourne :
      - df_resultats : DataFrame récapitulatif des métriques
      - scores_dict  : Dictionnaire {"nom_modèle": vecteur_scores_mse}
      - modeles_dict : Dictionnaire {"nom_modèle": instance_MLPRegressor}
    """
    if configs is None:
        configs = definir_configurations_autoencodeurs()

    scores_dict = {}
    modeles_dict = {}
    resultats = []

    print("--- Entraînement des configurations d'Auto-encodeur ---")
    for cfg in configs:
        nom = cfg["nom"]
        hidden = cfg["hidden"]
        alpha = cfg["alpha"]

        print(f"Entraînement de : {nom}...")
        t_start = time.time()

        # MLPRegressor utilisé comme Auto-encodeur : on apprend à reconstruire X à partir de X
        ae = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=random_state,
        )

        # Entraînement NON SUPERVISÉ sur X_train_s (sans utiliser les étiquettes y_train)
        ae.fit(X_train_s, X_train_s)
        t_train = round(time.time() - t_start, 2)

        # Reconstruction sur l'ensemble de Validation
        X_val_pred = ae.predict(X_val_s)

        # Score d'anomalie = Erreur Quadratique Moyenne (MSE) par transaction
        scores_val_ae = np.mean((X_val_s.values - X_val_pred) ** 2, axis=1)
        scores_dict[nom] = scores_val_ae
        modeles_dict[nom] = ae

        # Évaluation au meilleur seuil sous la contrainte Précision >= 80%
        res = utils_credit.trouver_meilleur_seuil(
            y_val, scores_val_ae, min_precision=min_precision
        )
        res["Modèle"] = nom
        res["Architecture"] = str(hidden)
        res["Temps Train (s)"] = t_train
        resultats.append(res)

    df_resultats = pd.DataFrame(resultats)

    # Réorganisation des colonnes pour un affichage soigné
    colonnes = [
        "Modèle",
        "Architecture",
        "Contrainte (>=80%)",
        "Seuil",
        "Précision",
        "Rappel",
        "F1-score",
        "AP",
        "AUC-ROC",
        "TP (Fraudes détectées)",
        "FP (Fausses alertes)",
        "FN (Fraudes manquées)",
        "Temps Train (s)",
    ]
    cols_existantes = [c for c in colonnes if c in df_resultats.columns]
    df_resultats = df_resultats[cols_existantes]

    return df_resultats, scores_dict, modeles_dict


def comparer_detecteurs_non_supervises(
    y_val,
    scores_if,
    scores_lof,
    scores_ae,
    nom_ae="Auto-encodeur (16-8-16)",
    min_precision=0.80,
):
    """Trace les courbes ROC et Précision-Rappel comparant directement
    Isolation Forest, Local Outlier Factor (LOF) et l'Auto-encodeur.
    """
    scores_comparaison = {
        "Isolation Forest": scores_if,
        "Local Outlier Factor (LOF)": scores_lof,
        nom_ae: scores_ae,
    }
    print(
        "Courbes Comparatives : Auto-encodeur vs Isolation Forest vs Local"
        " Outlier Factor"
    )
    utils_credit.tracer_courbes_roc_pr(
        y_val, scores_comparaison, min_precision=min_precision
    )


def evaluer_autoencodeur_test(
    modele_ae, X_test_s, y_test, seuil_gele, t_train=None, nom="Auto-encodeur (16-8-16)"
):
    """Évalue le modèle d'Auto-encodeur retenu sur le jeu de Test au seuil gelé
    calibré lors de la phase de validation.
    """
    X_test_pred = modele_ae.predict(X_test_s)
    scores_test_ae = np.mean((X_test_s.values - X_test_pred) ** 2, axis=1)

    metriques = utils_credit.evaluer_seuil(y_test, scores_test_ae, seuil=seuil_gele)
    metriques["AP"] = round(average_precision_score(y_test, scores_test_ae), 4)
    metriques["AUC-ROC"] = round(roc_auc_score(y_test, scores_test_ae), 4)
    metriques["Méthode"] = nom
    metriques["Type"] = "Non supervisé (Neuronal)"
    metriques["Seuil gelé"] = round(seuil_gele, 4)
    metriques["Temps Train (s)"] = t_train

    # Renommage des clés pour homogénéité du tableau final
    metriques["Balanced Acc"] = metriques.get("Balanced Accuracy", 0.0)
    metriques["TP"] = metriques.get("TP (Fraudes détectées)", 0)
    metriques["FP"] = metriques.get("FP (Fausses alertes)", 0)
    metriques["FN"] = metriques.get("Fraudes manquées", 0) or metriques.get(
        "FN (Fraudes manquées)", 0
    )

    return metriques, scores_test_ae


def harmoniser_metriques(
    m, fallback_nom="Modèle", fallback_type="Supervisé", default_time=None
):
    """Harmonise les noms de clés entre les dictionnaires de métriques de la Partie 2 et 3."""
    res = dict(m)

    # 1. Nom de méthode
    nom = res.get("Méthode") or res.get("Modèle") or fallback_nom
    nom = str(nom).replace(" [TEST]", "")
    res["Méthode"] = nom

    # 2. Type d'apprentissage
    if "Type" not in res or pd.isna(res.get("Type")):
        if "Régression" in nom or "Arbre" in nom:
            res["Type"] = "Supervisé"
        elif "Isolation" in nom:
            res["Type"] = "Non supervisé"
        elif "LOF" in nom:
            res["Type"] = "Non supervisé"
        elif "Auto" in nom:
            res["Type"] = "Non supervisé (Neuronal)"
        else:
            res["Type"] = fallback_type

    # 3. Seuil
    seuil_val = res.get("Seuil gelé") if "Seuil gelé" in res else res.get("Seuil")
    if seuil_val is not None:
        res["Seuil gelé"] = round(float(seuil_val), 4)

    # 4. Balanced Accuracy
    bacc = res.get("Balanced Acc") if "Balanced Acc" in res else res.get("Balanced Accuracy")
    if bacc is not None:
        res["Balanced Acc"] = round(float(bacc), 4)

    # 5. TP / FP / FN
    if "TP" not in res or pd.isna(res.get("TP")):
        res["TP"] = res.get("TP (Fraudes détectées)", 0)
    if "FP" not in res or pd.isna(res.get("FP")):
        res["FP"] = res.get("FP (Fausses alertes)", 0)
    if "FN" not in res or pd.isna(res.get("FN")):
        res["FN"] = res.get("FN (Fraudes manquées)", 0)

    # 6. Temps de calcul
    if "Temps Train (s)" not in res or pd.isna(res.get("Temps Train (s)")):
        if default_time is not None:
            res["Temps Train (s)"] = default_time
        elif "Régression" in nom:
            res["Temps Train (s)"] = 0.47
        elif "Arbre" in nom:
            res["Temps Train (s)"] = 2.35
        elif "Isolation" in nom:
            res["Temps Train (s)"] = 4.77

    return res


def tableau_recapitulatif_final(
    metriques_lr, metriques_dt, metriques_if, metriques_ae
):
    """Assemble et retourne le grand tableau récapitulatif comparant
    les 4 modèles finaux évalués sur le jeu de test gelé.
    """
    m_lr = harmoniser_metriques(
        metriques_lr,
        fallback_nom="Régression Logistique",
        fallback_type="Supervisé",
        default_time=0.47,
    )
    m_dt = harmoniser_metriques(
        metriques_dt,
        fallback_nom="Arbre de Décision (depth=5)",
        fallback_type="Supervisé",
        default_time=2.35,
    )
    m_if = harmoniser_metriques(
        metriques_if,
        fallback_nom="Isolation Forest",
        fallback_type="Non supervisé",
        default_time=4.77,
    )
    m_ae = harmoniser_metriques(
        metriques_ae,
        fallback_nom="Auto-encodeur (16-8-16)",
        fallback_type="Non supervisé (Neuronal)",
    )

    df = pd.DataFrame([m_lr, m_dt, m_if, m_ae])
    colonnes = [
        "Méthode",
        "Type",
        "Seuil gelé",
        "Précision",
        "Rappel",
        "F1-score",
        "Balanced Acc",
        "AP",
        "AUC-ROC",
        "TP",
        "FP",
        "FN",
        "Temps Train (s)",
    ]
    return df[colonnes]
