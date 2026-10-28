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


def fit_isolation_forest(X, random_state=42, **kwargs):
    """Ajuste Isolation Forest (sans utiliser y)."""
    model = IsolationForest(random_state=random_state, **kwargs)
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


def plot_detection(X, y_pred, scores, title):
    """Points détectés + couleur = score + histogramme."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # gauche : détection (prédictions)
    axes[0].scatter(X.loc[y_pred == 0, "x1"], X.loc[y_pred == 0, "x2"],
                    s=20, alpha=0.6, label="Prédit normal")
    axes[0].scatter(X.loc[y_pred == 1, "x1"], X.loc[y_pred == 1, "x2"],
                    s=40, c="red", marker="x", label="Prédit anomalie")
    axes[0].set_title(f"{title} — points détectés")
    axes[0].legend()
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


from sklearn.metrics import confusion_matrix, f1_score


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


def evaluate_binary(y_true, y_pred, name=""):
    """Évaluation après seuil : confusion + F1 (y seulement pour évaluer)."""
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    n_alertes = int(y_pred.sum())
    print(f"\n=== {name} ===")
    print("Seuil → nb alertes :", n_alertes)
    print("Matrice de confusion [[TN FP],[FN TP]] :\n", cm)
    print("F1 (classe anomalie) :", round(f1, 4))
    return {"confusion": cm, "f1": f1, "n_alertes": n_alertes}


def compare_thresholds(X, y_true, scores, y_pred_default, method_name="Methode"):
    """Compare seuil par défaut vs seuil IQR."""
    # 1) défaut
    print(f"\n######## {method_name} — seuil PAR DÉFAUT ########")
    evaluate_binary(y_true, y_pred_default, name=f"{method_name} | défaut")
    plot_detection(X, y_pred_default, scores, f"{method_name} — défaut")

    # 2) IQR (k=1.5 fixé a priori, sans tuner avec y)
    thr_iqr, y_pred_iqr = threshold_iqr(scores, k=1.5)
    print(f"\n######## {method_name} — seuil IQR ########")
    print("Seuil IQR =", round(thr_iqr, 6))
    evaluate_binary(y_true, y_pred_iqr, name=f"{method_name} | IQR")
    plot_detection(X, y_pred_iqr, scores, f"{method_name} — IQR")

    return thr_iqr, y_pred_iqr



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
    """Apprentissage (normaux) vs test (prédictions)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X_train["x1"], X_train["x2"], s=20, alpha=0.7, label="Train (normaux)")
    axes[0].scatter(X_test.loc[y_test == 0, "x1"], X_test.loc[y_test == 0, "x2"],
                    s=25, marker="o", facecolors="none", edgecolors="green", label="Test normal")
    axes[0].scatter(X_test.loc[y_test == 1, "x1"], X_test.loc[y_test == 1, "x2"],
                    s=40, c="red", marker="x", label="Test anomalie (vrai)")
    axes[0].set_title(f"{title} — train vs test (vraies étiquettes)")
    axes[0].legend()
    axes[0].axis("equal")

    axes[1].scatter(X_test.loc[y_pred == 0, "x1"], X_test.loc[y_pred == 0, "x2"],
                    s=25, alpha=0.7, label="Prédit normal")
    axes[1].scatter(X_test.loc[y_pred == 1, "x1"], X_test.loc[y_pred == 1, "x2"],
                    s=40, c="red", marker="x", label="Prédit nouveauté")
    axes[1].set_title(f"{title} — prédictions sur le test")
    axes[1].legend()
    axes[1].axis("equal")

    plt.tight_layout()
    plt.show()