import time
import numpy as np
import pandas as pd
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.base import clone

try:
    from . import utils_credit
except (ImportError, ValueError):
    try:
        from utils_files import utils_credit
    except ImportError:
        import utils_credit

def definir_configurations_easyensemble():
    return [
        {"nom": "EasyEnsemble (5 sous-ensembles)", "n_estimators": 5},
        {"nom": "EasyEnsemble (10 sous-ensembles)", "n_estimators": 10},
        {"nom": "EasyEnsemble (20 sous-ensembles)", "n_estimators": 20},
    ]



def entrainer_evaluer_easyensemble(
    X_train_s, y_train, X_val_s, y_val, configs=None, random_state=42, min_precision=0.80
):
    if configs is None:
        configs = definir_configurations_easyensemble()

    scores_dict, modeles_dict, resultats = {}, {}, []
    for cfg in configs:
        nom = cfg["nom"]
        debut = time.perf_counter()
        modele = EasyEnsembleClassifier(n_estimators=cfg["n_estimators"], random_state=random_state)
        modele.fit(X_train_s, y_train)
        t_train = round(time.perf_counter() - debut, 2)

        scores_val = modele.predict_proba(X_val_s)[:, 1]
        scores_dict[nom] = scores_val
        modeles_dict[nom] = modele

        res = utils_credit.trouver_meilleur_seuil(y_val, scores_val, min_precision=min_precision)
        res["Modèle"] = nom
        res["Sous-ensembles"] = cfg["n_estimators"]
        res["Temps Train (s)"] = t_train
        resultats.append(res)

    df_resultats = pd.DataFrame(resultats)
    colonnes = [
        "Modèle", "Sous-ensembles", "Contrainte (>=80%)", "Seuil", "Précision", "Rappel",
        "F1-score", "AP", "AUC-ROC", "TP (Fraudes détectées)", "FP (Fausses alertes)",
        "FN (Fraudes manquées)", "Temps Train (s)",
    ]
    return df_resultats[[c for c in colonnes if c in df_resultats.columns]], scores_dict, modeles_dict



def evaluer_easyensemble_test(modele, X_test_s, y_test, seuil_gele, t_train, nom):
    scores_test = modele.predict_proba(X_test_s)[:, 1]
    m = utils_credit.evaluer_seuil(y_test, scores_test, seuil=seuil_gele)
    ligne = {
        "Méthode": nom,
        "Type": "Supervisé (ensemble)",
        "Seuil gelé": round(seuil_gele, 4),
        "Précision": m["Précision"],
        "Rappel": m["Rappel"],
        "F1-score": m["F1-score"],
        "Balanced Acc": m["Balanced Accuracy"],
        "AP": round(average_precision_score(y_test, scores_test), 4),
        "AUC-ROC": round(roc_auc_score(y_test, scores_test), 4),
        "TP": m["TP (Fraudes détectées)"],
        "FP": m["FP (Fausses alertes)"],
        "FN": m["FN (Fraudes manquées)"],
        "TN": m["TN (Normales correctes)"],
        "Temps Train (s)": t_train,
    }
    return pd.DataFrame([ligne]), scores_test





def comparer_easyensemble_aux_non_supervises(
    modele_ee, nom_ee, seuil_ee, t_ee, detecteurs, seuils, X_train_s, X_train_lof, X_test_s, y_test
):
    X_tr = np.asarray(X_train_s)
    X_lof = np.asarray(X_train_lof)
    X_te = np.asarray(X_test_s)

    def seuil_de(nom):
        return seuils.loc[seuils["Modèle"] == nom, "Seuil optimal"].iloc[0]

    def ajuster(nom, X):
        debut = time.perf_counter()
        clf = clone(detecteurs[nom]).fit(X)
        return clf, round(time.perf_counter() - debut, 2)

    def scorer(fonction):
        debut = time.perf_counter()
        scores = fonction()
        return scores, round(time.perf_counter() - debut, 4)

    def ligne(nom, type_, scores, seuil, t_fit, t_score):
        m = utils_credit.evaluer_seuil(y_test, scores, seuil=seuil)
        return {
            "Modèle": nom,
            "Type": type_,
            "Seuil": round(seuil, 4),
            "Précision": m["Précision"],
            "Rappel": m["Rappel"],
            "F1-score": m["F1-score"],
            "Balanced Acc": m["Balanced Accuracy"],
            "AP": round(average_precision_score(y_test, scores), 4),
            "AUC-ROC": round(roc_auc_score(y_test, scores), 4),
            "TP": m["TP (Fraudes détectées)"],
            "FP": m["FP (Fausses alertes)"],
            "FN": m["FN (Fraudes manquées)"],
            "TN": m["TN (Normales correctes)"],
            "Temps entraînement (s)": t_fit,
            "Temps scoring test (s)": t_score,
        }

    scores_ee, t_score_ee = scorer(lambda: modele_ee.predict_proba(X_te)[:, 1])

    clf_if, t_fit_if = ajuster("Isolation Forest", X_tr)
    scores_if, t_score_if = scorer(lambda: -clf_if.score_samples(X_te))

    clf_lof, t_fit_lof = ajuster("Local Outlier Factor (LOF)", X_lof)
    scores_lof, t_score_lof = scorer(lambda: -clf_lof.score_samples(X_te))

    seuil_if = seuil_de("Isolation Forest")
    seuil_lof = seuil_de("Local Outlier Factor (LOF)")

    tableau = pd.DataFrame([
        ligne(nom_ee, "Supervisé (ensemble)", scores_ee, seuil_ee, t_ee, t_score_ee),
        ligne("Isolation Forest", "Non supervisé", scores_if, seuil_if, t_fit_if, t_score_if),
        ligne("Local Outlier Factor (LOF)", "Non supervisé", scores_lof, seuil_lof, t_fit_lof, t_score_lof),
    ])
    scores = {nom_ee: scores_ee, "Isolation Forest": scores_if, "Local Outlier Factor (LOF)": scores_lof}
    seuils_utilises = {nom_ee: seuil_ee, "Isolation Forest": seuil_if, "Local Outlier Factor (LOF)": seuil_lof}
    return tableau, scores, seuils_utilises


def cas_discordants(y_test, scores_a, seuil_a, scores_b, seuil_b, nom_a, nom_b):
    y = np.asarray(y_test) == 1
    detecte_a = y & (np.asarray(scores_a) >= seuil_a)
    detecte_b = y & (np.asarray(scores_b) >= seuil_b)
    return {
        f"Fraudes détectées par {nom_a} seulement": np.flatnonzero(detecte_a & ~detecte_b),
        f"Fraudes détectées par {nom_b} seulement": np.flatnonzero(detecte_b & ~detecte_a),
    }