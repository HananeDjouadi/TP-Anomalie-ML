import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay

def load_mouse(path="../data/mouse.txt"):
    """Charge Mouse et construit les labels (10 derniers = anomalies)."""
    X = pd.read_csv(path, sep=r"\s+", header=None, names=["x1", "x2"])
    y = np.zeros(len(X), dtype=int)
    y[-10:] = 1  # labels AVANT tout mélange
    return X, y


def check_dataset(X, y, name="dataset"):
    """Vérifie format, NaN et proportion d'anomalies."""
    print(f"=== {name} ===")
    print("Shape X :", X.shape)
    print("Valeurs manquantes :\n", X.isna().sum())
    print("Nb anomalies :", int(y.sum()), f"({100 * y.mean():.2f} %)")
    print(X.head())


def plot_mouse(X, y, title="Mouse — nuage de points"):
    """Scatter plot des points Mouse colorés par label."""
    plt.figure(figsize=(6, 6))
    plt.scatter(X.loc[y == 0, "x1"], X.loc[y == 0, "x2"],
                s=20, alpha=0.7, label="Normal (0)")
    plt.scatter(X.loc[y == 1, "x1"], X.loc[y == 1, "x2"],
                s=40, c="red", marker="x", label="Anomalie (1)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def fit_isolation_forest(X, random_state=42, contamination=0.02, **kwargs):
    """
    Ajuste Isolation Forest (sans utiliser y pour le fit).

    contamination=0.02 ≈ 10/500 sur Mouse : on indique la *proportion*
    d'anomalies attendue pour le seuil de predict (pas quels points).
    Surbrider via contamination=... si besoin (ex. \"auto\").
    """
    model = IsolationForest(
        random_state=random_state, contamination=contamination, **kwargs
    )
    model.fit(X)
    # score_samples : plus bas = plus anormal dans sklearn
    # on inverse pour : plus haut = plus suspect (protocole du TP)
    scores = -model.score_samples(X)
    preds = model.predict(X)  # 1 = normal, -1 = anomalie
    y_pred = (preds == -1).astype(int)
    return model, scores, y_pred



def fit_lof_outliers(X, n_neighbors=20, **kwargs):
    """LOF en mode outliers (fit sur tout l'échantillon)."""
    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False, **kwargs)
    preds = model.fit_predict(X)  # 1 = normal, -1 = anomalie
    # negative_outlier_factor_ : plus bas = plus anormal
    scores = -model.negative_outlier_factor_
    y_pred = (preds == -1).astype(int)
    return model, scores, y_pred


def plot_detection(X, y_pred, scores, title, y_true=None):
    """
    Points détectés + couleur = score + histogramme.

    Si y_true est fourni (évaluation seulement), on distingue :
    - TP : alerte correcte (cercle vert)
    - FP : fausse alerte (croix rouge)
    - FN : anomalie manquée (cercle orange vide)
    """
    y_pred = np.asarray(y_pred).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # gauche : détection
    if y_true is None:
        axes[0].scatter(
            X.loc[y_pred == 0, "x1"],
            X.loc[y_pred == 0, "x2"],
            s=20,
            alpha=0.6,
            label="Prédit normal",
        )
        axes[0].scatter(
            X.loc[y_pred == 1, "x1"],
            X.loc[y_pred == 1, "x2"],
            s=40,
            c="red",
            marker="x",
            label="Prédit anomalie",
        )
    else:
        y_true = np.asarray(y_true).ravel()
        tn = (y_pred == 0) & (y_true == 0)
        tp = (y_pred == 1) & (y_true == 1)
        fp = (y_pred == 1) & (y_true == 0)
        fn = (y_pred == 0) & (y_true == 1)

        axes[0].scatter(
            X.iloc[tn]["x1"], X.iloc[tn]["x2"], s=18, alpha=0.45, c="0.7", label="Normal OK"
        )
        axes[0].scatter(
            X.iloc[tp]["x1"],
            X.iloc[tp]["x2"],
            s=90,
            facecolors="none",
            edgecolors="green",
            linewidths=2,
            label="TP (vraie anomalie détectée)",
        )
        axes[0].scatter(
            X.iloc[fp]["x1"],
            X.iloc[fp]["x2"],
            s=70,
            c="red",
            marker="x",
            linewidths=2,
            label="FP (fausse alerte)",
        )
        axes[0].scatter(
            X.iloc[fn]["x1"],
            X.iloc[fn]["x2"],
            s=90,
            facecolors="none",
            edgecolors="orange",
            linewidths=2,
            label="FN (anomalie manquée)",
        )

    axes[0].set_title(f"{title} — points détectés")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].axis("equal")

    # droite : couleur = score
    sc = axes[1].scatter(X["x1"], X["x2"], c=scores, cmap="viridis", s=25)
    plt.colorbar(sc, ax=axes[1], label="Score (↑ = plus suspect)")
    axes[1].set_title(f"{title} — scores")
    axes[1].axis("equal")

    plt.tight_layout()
    plt.show()

    # histogramme des scores
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Score d'anomalie")
    plt.ylabel("Fréquence")
    plt.title(f"{title} — histogramme des scores")
    plt.tight_layout()
    plt.show()


from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)
from sklearn.cluster import KMeans


def threshold_iqr(scores, k=1.5):
    """
    Seuil IQR sur les scores (sans utiliser y).
    Anomalie si score > Q3 + k * IQR.
    """
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    threshold = q3 + k * iqr
    y_pred = (scores > threshold).astype(int)
    return threshold, y_pred


def threshold_clustering(scores, random_state=42):
    """
    Seuil par clustering (KMeans, k=2) sur les scores — sans utiliser y.
    Le cluster de score moyen le plus élevé = anomalies.
    Le seuil reporté est la frontière entre les 2 centres.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    X_1d = scores.reshape(-1, 1)
    km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_1d)
    centers = km.cluster_centers_.ravel()
    anom_cluster = int(np.argmax(centers))
    y_pred = (labels == anom_cluster).astype(int)
    # frontière approximative entre les deux centres
    threshold = float(np.mean(centers))
    return threshold, y_pred


def evaluate_binary(y_true, y_pred, name=""):
    """
    Évaluation après seuil (protocole §2) :
    confusion, précision, rappel, F1 (classe 1), balanced accuracy.
    Les labels y servent uniquement à évaluer.
    """
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    n_alertes = int(np.asarray(y_pred).sum())

    print(f"\n=== {name} ===")
    print("Seuil → nb alertes :", n_alertes)
    print("Matrice de confusion [[TN FP],[FN TP]] :\n", cm)
    print("Précision (classe anomalie) :", round(precision, 4))
    print("Rappel    (classe anomalie) :", round(recall, 4))
    print("F1        (classe anomalie) :", round(f1, 4))
    print("Balanced accuracy           :", round(bal_acc, 4))

    return {
        "confusion": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "n_alertes": n_alertes,
    }


def compare_thresholds(X, y_true, scores, y_pred_default, method_name="Methode"):
    """Compare seuil par défaut vs IQR vs clustering (KMeans)."""
    # 1) défaut
    print(f"\n######## {method_name} — seuil PAR DÉFAUT ########")
    evaluate_binary(y_true, y_pred_default, name=f"{method_name} | défaut")
    plot_detection(X, y_pred_default, scores, f"{method_name} — défaut", y_true=y_true)

    # 2) IQR (k=1.5 fixé a priori, sans tuner avec y)
    thr_iqr, y_pred_iqr = threshold_iqr(scores, k=1.5)
    print(f"\n######## {method_name} — seuil IQR ########")
    print("Seuil IQR =", round(thr_iqr, 6))
    evaluate_binary(y_true, y_pred_iqr, name=f"{method_name} | IQR")
    plot_detection(X, y_pred_iqr, scores, f"{method_name} — IQR", y_true=y_true)

    # 3) Clustering KMeans (k=2) sur les scores — sans utiliser y
    thr_cl, y_pred_cl = threshold_clustering(scores)
    print(f"\n######## {method_name} — seuil CLUSTERING (KMeans k=2) ########")
    print("Frontière (moyenne des centres) =", round(thr_cl, 6))
    evaluate_binary(y_true, y_pred_cl, name=f"{method_name} | clustering")
    plot_detection(X, y_pred_cl, scores, f"{method_name} — clustering", y_true=y_true)

    return {
        "iqr": (thr_iqr, y_pred_iqr),
        "clustering": (thr_cl, y_pred_cl),
    }



def ranking_metrics(y_true, scores, name=""):
    """
    Évaluation du classement (scores continus).
    Ne dépend PAS du seuil.
    """
    ap = average_precision_score(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    print(f"\n=== {name} — classement ===")
    print("Average Precision (AP) :", round(ap, 4))
    print("AUC-ROC               :", round(auc, 4))
    return {"AP": ap, "AUC_ROC": auc}


def plot_ranking_curves(y_true, scores, name="Methode"):
    """Courbes ROC et Précision-Rappel."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    RocCurveDisplay.from_predictions(y_true, scores, ax=axes[0], name=name)
    axes[0].set_title(f"{name} — ROC")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)

    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=axes[1], name=name)
    axes[1].set_title(f"{name} — Précision-Rappel")

    plt.tight_layout()
    plt.show()


def demonstrate_threshold_independence(y_true, scores, y_pred_default, name="Methode"):
    """
    Preuve pédagogique : changer le seuil change y_pred, pas les scores
    → AP / AUC restent identiques (ils n'utilisent pas y_pred).
    """
    _, y_pred_iqr = threshold_iqr(scores)
    ap = average_precision_score(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    print(f"\n--- Preuve sur {name} : le seuil change les décisions, pas AP/AUC ---")
    print("Nb alertes seuil défaut :", int(np.asarray(y_pred_default).sum()))
    print("Nb alertes seuil IQR    :", int(y_pred_iqr.sum()))
    print("AP  (mêmes scores) :", round(ap, 4))
    print("AUC (mêmes scores) :", round(auc, 4))
    print("(AP/AUC ne prennent pas y_pred en entrée, seulement scores + y_true)")
    return {
        "AP": ap,
        "AUC_ROC": auc,
        "n_default": int(np.asarray(y_pred_default).sum()),
        "n_iqr": int(y_pred_iqr.sum()),
    }


from sklearn.model_selection import train_test_split


def split_novelty_mouse(X, y, test_size_normal=0.3, random_state=42):
    """
    Réserve une partie des points normaux + TOUTES les anomalies pour le test.
    L'apprentissage ne contient que des points normaux.
    """
    X = X.reset_index(drop=True)
    y = np.asarray(y)

    X_norm = X[y == 0]
    X_anom = X[y == 1]
    y_anom = y[y == 1]

    X_norm_train, X_norm_test, _, _ = train_test_split(
        X_norm, y[y == 0],
        test_size=test_size_normal,
        random_state=random_state
    )

    X_train = X_norm_train.reset_index(drop=True)
    y_train = np.zeros(len(X_train), dtype=int)  # que des normaux

    X_test = pd.concat([X_norm_test, X_anom], axis=0).reset_index(drop=True)
    y_test = np.concatenate([
        np.zeros(len(X_norm_test), dtype=int),
        y_anom
    ])

    return X_train, y_train, X_test, y_test


def fit_lof_novelty(X_train, X_test, n_neighbors=20, **kwargs):
    """
    LOF en détection de nouveautés :
    - fit sur X_train (normaux)
    - scores sur X_test (nouveaux points)
    """
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,
        **kwargs
    )
    model.fit(X_train)

    # score_samples : plus bas = plus anormal → on inverse
    scores_test = -model.score_samples(X_test)
    preds = model.predict(X_test)  # 1 normal, -1 nouveauté/anomalie
    y_pred = (preds == -1).astype(int)

    return model, scores_test, y_pred


def plot_novelty(X_train, X_test, y_test, y_pred, title="LOF nouveautés"):
    """Apprentissage (normaux) vs test, avec TP/FP/FN sur le panneau droit."""
    y_test = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # gauche : vérité (train + test)
    axes[0].scatter(X_train["x1"], X_train["x2"], s=20, alpha=0.7, label="Train (normaux)")
    axes[0].scatter(
        X_test.iloc[y_test == 0]["x1"],
        X_test.iloc[y_test == 0]["x2"],
        s=25,
        marker="o",
        facecolors="none",
        edgecolors="green",
        label="Test normal",
    )
    axes[0].scatter(
        X_test.iloc[y_test == 1]["x1"],
        X_test.iloc[y_test == 1]["x2"],
        s=40,
        c="red",
        marker="x",
        label="Test anomalie (vrai)",
    )
    axes[0].set_title(f"{title} — train vs test (vraies étiquettes)")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].axis("equal")

    # droite : erreurs mises en évidence (TP / FP / FN)
    tn = (y_pred == 0) & (y_test == 0)
    tp = (y_pred == 1) & (y_test == 1)
    fp = (y_pred == 1) & (y_test == 0)
    fn = (y_pred == 0) & (y_test == 1)

    axes[1].scatter(
        X_test.iloc[tn]["x1"],
        X_test.iloc[tn]["x2"],
        s=18,
        alpha=0.45,
        c="0.7",
        label="Normal OK",
    )
    axes[1].scatter(
        X_test.iloc[tp]["x1"],
        X_test.iloc[tp]["x2"],
        s=90,
        facecolors="none",
        edgecolors="green",
        linewidths=2,
        label="TP (vraie anomalie détectée)",
    )
    axes[1].scatter(
        X_test.iloc[fp]["x1"],
        X_test.iloc[fp]["x2"],
        s=70,
        c="red",
        marker="x",
        linewidths=2,
        label="FP (fausse alerte)",
    )
    axes[1].scatter(
        X_test.iloc[fn]["x1"],
        X_test.iloc[fn]["x2"],
        s=90,
        facecolors="none",
        edgecolors="orange",
        linewidths=2,
        label="FN (anomalie manquée)",
    )
    axes[1].set_title(f"{title} — prédictions test (erreurs)")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].axis("equal")

    plt.tight_layout()
    plt.show()