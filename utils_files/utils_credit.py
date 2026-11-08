import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def examiner_donnees_brutes(filepath="../data/creditcard.csv"):
    """Charge le fichier CSV brut et affiche un résumé complet :
    dimensions, valeurs manquantes, nombre de doublons et types des colonnes.
    """
    df_raw = pd.read_csv(filepath)
    n_lignes, n_cols = df_raw.shape
    n_na = int(df_raw.isna().sum().sum())
    n_doublons = int(df_raw.duplicated().sum())
    types_dict = {str(k): int(v) for k, v in df_raw.dtypes.value_counts().items()}

    print("=== Examen initial du jeu de données brut ===")
    print(f"Dimensions               : {n_lignes} lignes, {n_cols} colonnes")
    print(f"Valeurs manquantes       : {n_na}")
    print(f"Nombre de doublons       : {n_doublons} ({n_doublons / n_lignes * 100:.2f}%)")
    print(f"Types de colonnes        : {types_dict}")
    return df_raw


def load_clean_data(filepath="../data/creditcard.csv"):
    """Charge le fichier CSV et supprime les doublons stricts."""
    df = pd.read_csv(filepath)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def afficher_desequilibre_classes(df):
    """Affiche le décompte et le pourcentage des classes (fraude vs normale)."""
    comptes = df["Class"].value_counts()
    n_norm = int(comptes.get(0, 0))
    n_fraud = int(comptes.get(1, 0))
    total = len(df)
    pct_fraud = (n_fraud / total) * 100

    print("=== Distribution des classes ===")
    print(f"Transactions normales (0) : {n_norm} ({100 - pct_fraud:.3f}%)")
    print(f"Fraudes (1)               : {n_fraud} ({pct_fraud:.3f}%)")
    print(f"Ratio de déséquilibre     : 1 fraude pour {int(round(n_norm / max(1, n_fraud)))} transactions normales")


def tracer_distribution_classes(df):
    """Affiche l'histogramme du déséquilibre de classe en échelle logarithmique."""
    plt.figure(figsize=(6, 3.5))
    counts = df["Class"].value_counts()
    bars = plt.bar(
        ["Normales (0)", "Fraudes (1)"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=["#2b5c8f", "#d95f02"],
        width=0.55,
    )
    plt.yscale("log")
    plt.ylim(50, 3e6)  # Espace suffisant au-dessus pour que le texte ne touche pas le haut
    plt.title("Distribution des classes (échelle logarithmique)", pad=15)
    plt.ylabel("Nombre de transactions (log)")
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.4,
            f"{int(height):,}".replace(",", " "),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


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


def verifier_decoupage(y_train, y_val, y_test):
    """Vérifie la bonne stratification des découpages Train / Val / Test."""
    partitions = [("Train (60%)", y_train), ("Val (20%)", y_val), ("Test (20%)", y_test)]
    res = []
    for nom, y_part in partitions:
        n_tot = len(y_part)
        n_f = int((y_part == 1).sum())
        pct_f = (n_f / n_tot) * 100
        res.append({
            "Partition": nom,
            "Total transactions": f"{n_tot:,}".replace(",", " "),
            "Nb fraudes": n_f,
            "Taux de fraude": f"{pct_f:.3f}%",
        })
    df_verif = pd.DataFrame(res)
    return df_verif


def tracer_distribution_temporelle(df):
    """Trace la distribution temporelle des transactions (en heures sur 48h) pour normales vs fraudes."""
    time_heures = df["Time"] / 3600
    plt.figure(figsize=(10, 4))
    plt.hist(
        time_heures[df["Class"] == 0],
        bins=48,
        alpha=0.6,
        label="Transactions Normales",
        density=True,
        color="#2b5c8f",
    )
    plt.hist(
        time_heures[df["Class"] == 1],
        bins=48,
        alpha=0.6,
        label="Fraudes",
        density=True,
        color="#d95f02",
    )
    plt.title("Distribution temporelle sur 48 heures (densité)")
    plt.xlabel("Heures écoulées (0 à 48h)")
    plt.ylabel("Densité de probabilité")
    plt.legend()
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


def analyser_echelles_variables(X_train):
    """Affiche les statistiques descriptives comparées de 'Amount' et des composantes V1..V28."""
    v_cols = [c for c in X_train.columns if c.startswith("V")]

    amount_stats = X_train["Amount"].describe()
    print("=== Échelle de la variable 'Amount' (Train) ===")
    print(
        f"Médiane : {amount_stats['50%']:.2f} € | Moyenne : {amount_stats['mean']:.2f} € | Écart-type : {amount_stats['std']:.2f} €"
    )
    print(
        f"Min     : {amount_stats['min']:.2f} € | Max     : {amount_stats['max']:.2f} € | IQR (Q3 - Q1) : {amount_stats['75%'] - amount_stats['25%']:.2f} €"
    )

    print("\n=== Échelle des variables issues de l'ACP (V1 à V28) (Train) ===")
    print(f"Moyenne globale des V_i     : {X_train[v_cols].mean().mean():.4f} (centrées en 0)")
    print(f"Écart-type moyen des V_i    : {X_train[v_cols].std().mean():.4f} (unitaires)")
    print(f"Étendue globale (min / max) : [{X_train[v_cols].min().min():.2f} ; {X_train[v_cols].max().max():.2f}]")


def tracer_distribution_montant(X_train):
    """Affiche deux histogrammes : le montant brut (asymétrique) et log(1 + Amount)."""
    plt.figure(figsize=(10, 3.5))

    plt.subplot(1, 2, 1)
    plt.hist(X_train["Amount"], bins=50, color="#2b5c8f", edgecolor="black", linewidth=0.3)
    plt.title("Distribution brute de 'Amount' (forte asymétrie)")
    plt.xlabel("Montant (€)")
    plt.ylabel("Fréquence")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(X_train["Amount"]), bins=50, color="#2ca02c", edgecolor="black", linewidth=0.3)
    plt.title("Transformation log(1 + Amount)")
    plt.xlabel("log(1 + Montant)")
    plt.ylabel("Fréquence")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


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

def tracer_boxplots_comparatifs(X_train, X_train_s):
    """Affiche deux boxplots côte à côte :
    1. L'écrasement provoqué par Amount brut face aux composantes V1..V4
    2. L'alignement parfait des échelles après RobustScaler
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 1. Avant normalisation (avec échelle log pour voir les boîtes malgré l'écart énorme)
    cols_preview = ["V1", "V2", "V3", "Amount"]
    X_train[cols_preview].boxplot(ax=ax1, showfliers=False)
    ax1.set_title("Échelles brutes (sans outliers)")
    ax1.set_ylabel("Valeur")
    
    # 2. Après RobustScaler
    df_scaled = X_train_s[["V1", "V2", "V3", "Amount"]].copy()
    df_scaled.boxplot(ax=ax2, showfliers=False)
    ax2.set_title("Échelles après RobustScaler (alignées)")
    ax2.set_ylabel("Valeur standardisée")
    
    plt.tight_layout()
    plt.show()


def verifier_mise_a_echelle(X_train_s):
    """Vérifie les propriétés de RobustScaler sur Train : médiane à 0 et IQR normalisé."""
    stats = X_train_s["Amount"].describe()
    print("=== Vérification de 'Amount' après RobustScaler (Train) ===")
    print(f"Médiane (doit être 0.0)  : {stats['50%']:.4f}")
    print(f"Q1 (-0.227 typique)      : {stats['25%']:.4f}")
    print(f"Q3 (+0.773 typique)      : {stats['75%']:.4f}")
    print(f"IQR normalisé (Q3 - Q1)  : {stats['75%'] - stats['25%']:.4f} (égal à 1.0)")
    print(f"Min / Max                : [{stats['min']:.3f} ; {stats['max']:.3f}]")


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


def appliquer_smote(X_train_s, y_train, random_state=42):
    """Applique le sur-échantillonnage SMOTE uniquement sur l'ensemble Train
    et affiche les effectifs avant et après rééchantillonnage.
    """
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_s, y_train)
    print(
        f"Effectifs Train initial : {len(y_train)} dont {(y_train == 1).sum()} fraudes"
    )
    print(
        f"Effectifs Train SMOTE   : {len(y_train_smote)} dont {(y_train_smote == 1).sum()} fraudes"
    )
    return X_train_smote, y_train_smote

def comparer_modeles(df_res_lr, df_res_dt, metrique="AP (Validation)", afficher=True):
    """Fusionne les résultats LR et Arbre, classe les 6 modèles selon la métrique choisie
    et affiche le tableau comparatif ainsi qu'un diagramme en barres AP / AUC-ROC.
    Retourne le tableau trié et le nom du meilleur modèle.
    """
    df_all = pd.concat([df_res_lr, df_res_dt], ignore_index=True)
    df_all["Famille"] = df_all["Modèle"].str.split(" ").str[0]
    df_all["Variante"] = df_all["Modèle"].str.extract(r"\((.*)\)")

    df_all = df_all.sort_values(metrique, ascending=False).reset_index(drop=True)
    df_all["Rang"] = df_all.index + 1
    meilleur = df_all.loc[0, "Modèle"]

    if afficher:
        print(f"Classement des modèles selon : {metrique}")
        try:
            from IPython.display import display
            display(df_all[["Rang", "Modèle", "AP (Validation)", "AUC-ROC"]])
        except Exception:
            print(df_all[["Rang", "Modèle", "AP (Validation)", "AUC-ROC"]])

        # Diagramme en barres : une barre par métrique, un groupe par modèle
        ax = df_all.set_index("Modèle")[["AP (Validation)", "AUC-ROC"]].plot(
            kind="barh", figsize=(9, 5)
        )
        ax.invert_yaxis()  # le meilleur en haut
        ax.set_xlabel("Score")
        ax.set_title("Comparaison des 6 modèles (Validation)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Variante la plus efficace pour chaque famille de modèles
        print("\nMeilleure variante par famille :")
        for fam, g in df_all.groupby("Famille"):
            best = g.loc[g[metrique].idxmax()]
            print(f"  {fam} -> {best['Variante']} ({metrique} = {best[metrique]})")

        print(f"\nMeilleur modèle global : {meilleur}")

    return df_all, meilleur



def evaluer_variantes_supervisees(
    X_train_s, y_train, X_train_smote, y_train_smote, X_val_s, y_val, type_modele="lr", afficher=True
):
    """Entraîne et compare les 3 variantes (Sans correction, Pondération, SMOTE)
    pour la Régression Logistique ou l'Arbre de Décision.
    Affiche le tableau comparatif et les courbes ROC et PR.
    """
    if type_modele == "lr":
        modeles = {
            "LR (Sans correction)": LogisticRegression(max_iter=1000, random_state=42),
            "LR (Pondération classes)": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            ),
            "LR (SMOTE)": LogisticRegression(max_iter=1000, random_state=42),
        }
        titre = "Régression Logistique : Comparaison des 3 variantes"
    else:
        modeles = {
            "Arbre (Sans correction)": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Arbre (Pondération classes)": DecisionTreeClassifier(
                max_depth=5, class_weight="balanced", random_state=42
            ),
            "Arbre (SMOTE)": DecisionTreeClassifier(max_depth=5, random_state=42),
        }
        titre = "Arbre de Décision : Comparaison des 3 variantes"

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

    if afficher:
        print(titre)
        try:
            from IPython.display import display
            display(df_res)
        except Exception:
            print(df_res)
        tracer_courbes_roc_pr(y_val, scores_dict)

    return df_res, scores_dict, modeles


def evaluer_grille_lr(
    X_train_s, y_train, X_train_smote, y_train_smote, X_val_s, y_val, min_precision=0.80, afficher=True
):
    """Évalue les 6 configurations de Régression Logistique (C=0.1, 1.0 sur les 3 variantes).
    Calcule le seuil optimal pour chaque variante et ordonne selon l'AP décroissant.
    """
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

        res = {"C": cfg["C"], "Variante": cfg["variante"]}
        metrics = trouver_meilleur_seuil(y_val, scores_val, min_precision=min_precision)
        res.update(metrics)
        resultats.append(res)

    df_res = pd.DataFrame(resultats)
    cols_order = [
        "C",
        "Variante",
        "AP",
        "AUC-ROC",
        "Contrainte (>=80%)",
        "Seuil",
        "Précision",
        "Rappel",
        "F1-score",
        "Balanced Accuracy",
        "FP (Fausses alertes)",
        "FN (Fraudes manquées)",
        "TP (Fraudes détectées)",
        "TN (Normales correctes)",
    ]
    existing_cols = [c for c in cols_order if c in df_res.columns]
    df_res = df_res[existing_cols]
    df_res = df_res.sort_values(by="AP", ascending=False)

    if afficher:
        print("Régression Logistique : Évaluation des 6 configurations")
        try:
            from IPython.display import display
            display(df_res)
        except Exception:
            print(df_res)

    return df_res, scores_6_lr


def evaluer_grille_arbre(
    X_train_s, y_train, X_train_smote, y_train_smote, X_val_s, y_val, min_precision=0.80, afficher=True
):
    """Évalue les 6 configurations d'Arbre de Décision (max_depth=3, 5 sur les 3 variantes).
    Calcule le seuil optimal pour chaque variante et ordonne selon l'AP décroissant.
    """
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

        res = {"Profondeur": cfg["max_depth"], "Variante": cfg["variante"]}
        metrics = trouver_meilleur_seuil(y_val, scores_val, min_precision=min_precision)
        res.update(metrics)
        resultats.append(res)

    df_res = pd.DataFrame(resultats)
    cols_order = [
        "Profondeur",
        "Variante",
        "AP",
        "AUC-ROC",
        "Contrainte (>=80%)",
        "Seuil",
        "Précision",
        "Rappel",
        "F1-score",
        "Balanced Accuracy",
        "FP (Fausses alertes)",
        "FN (Fraudes manquées)",
        "TP (Fraudes détectées)",
        "TN (Normales correctes)",
    ]
    existing_cols = [c for c in cols_order if c in df_res.columns]
    df_res = df_res[existing_cols]
    df_res = df_res.sort_values(by="AP", ascending=False)

    if afficher:
        print("Arbre de Décision : Évaluation des 6 configurations")
        try:
            from IPython.display import display
            display(df_res)
        except Exception:
            print(df_res)

    return df_res, scores_6_dt


def entrainer_modeles_non_supervises(
    X_train_s, X_val_s, y_val, n_sub_lof=30000, random_state=42, min_precision=0.80, afficher=True
):
    """Ajuste Isolation Forest et LOF (sur sous-échantillon fixe de 30 000 transactions),
    harmonise les scores (-score_samples) et calibre les seuils optimaux selon la contrainte opérationnelle.
    """
    detecteurs = {
        "Isolation Forest": IsolationForest(
            n_estimators=100, contamination="auto", random_state=random_state, n_jobs=-1
        ),
        "Local Outlier Factor (LOF)": LocalOutlierFactor(
            n_neighbors=20, novelty=True, n_jobs=-1
        ),
    }

    X_train_sub = X_train_s.sample(n=n_sub_lof, random_state=random_state)
    scores_non_sup_dict = {}
    resultats = []

    print(f"Effectif utilisé pour l'entraînement de LOF : {len(X_train_sub)} transactions.")
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
    cols_order = [
        "Modèle",
        "AP",
        "AUC-ROC",
        "Contrainte (>=80%)",
        "Seuil",
        "Précision",
        "Rappel",
        "F1-score",
        "Balanced Accuracy",
        "FP (Fausses alertes)",
        "FN (Fraudes manquées)",
        "TP (Fraudes détectées)",
        "TN (Normales correctes)",
    ]
    existing_cols = [c for c in cols_order if c in df_res.columns]
    df_res = df_res[existing_cols]

    if afficher:
        print("\nDétecteurs non supervisés : Résultats sur la Validation")
        try:
            from IPython.display import display
            display(df_res)
        except Exception:
            print(df_res)

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
