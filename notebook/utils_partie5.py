"""Partie 5.1 — Robustesse : contamination de la référence normale (Mouse)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score


def generate_synthetic_anomalies(X_normal, n, random_state, scale=3.0):
    """
    Génère n anomalies synthétiques 2D.

    Méthode : autour de la moyenne des normaux, avec un bruit gaussien
    amplifié (scale * écart-type). Ces points sont typiquement éloignés
    du nuage "Mickey" → anomalies artificielles.

    Les conclusions dépendent de ce générateur (à discuter dans l'analyse).
    """
    rng = np.random.RandomState(random_state)
    X_arr = np.asarray(X_normal, dtype=float)
    mu = X_arr.mean(axis=0)
    std = X_arr.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    synth = mu + scale * std * rng.randn(n, X_arr.shape[1])
    cols = list(X_normal.columns) if hasattr(X_normal, "columns") else ["x1", "x2"]
    return pd.DataFrame(synth, columns=cols)


def contaminate_train(X_train_normal, rate, random_state, scale=3.0):
    """
    Remplace une fraction `rate` des points normaux par des anomalies synthétiques.
    Conserve exactement la même taille d'apprentissage.
    rate=0 → train inchangé (0 % contamination).
    """
    X_train = X_train_normal.reset_index(drop=True).copy()
    n = len(X_train)
    n_contam = int(round(n * rate))

    if n_contam == 0:
        return X_train

    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=n_contam, replace=False)
    synth = generate_synthetic_anomalies(
        X_train, n_contam, random_state=random_state, scale=scale
    )
    X_out = X_train.copy()
    X_out.iloc[idx] = synth.values
    return X_out.reset_index(drop=True)


def score_if(X_train, X_test, random_state=42):
    """IF hyperparams fixes ; score test (↑ = plus suspect)."""
    model = IsolationForest(
        n_estimators=100, contamination="auto", random_state=random_state, n_jobs=-1
    )
    model.fit(X_train)
    return -model.score_samples(X_test)


def score_lof(X_train, X_test, random_state=42):
    """LOF novelty=True, hyperparams fixes ; score test (↑ = plus suspect)."""
    # random_state non utilisé par LOF sklearn, gardé pour signature homogène
    model = LocalOutlierFactor(n_neighbors=20, novelty=True)
    model.fit(X_train)
    return -model.score_samples(X_test)


def run_contamination_experiment(
    X_train_clean,
    X_test,
    y_test,
    rates=(0.0, 0.05, 0.10),
    seeds=(42, 43, 44),
    scale=3.0,
):
    """
    Pour chaque taux de contamination et chaque graine :
    contamine le train, réajuste IF/LOF, calcule AP sur le MÊME test réel.
    Ne contamine jamais le test.
    """
    rows = []
    for rate in rates:
        for seed in seeds:
            X_tr = contaminate_train(
                X_train_clean, rate=rate, random_state=seed, scale=scale
            )
            for name, scorer in [("Isolation Forest", score_if), ("LOF", score_lof)]:
                scores = scorer(X_tr, X_test, random_state=seed)
                ap = average_precision_score(y_test, scores)
                rows.append(
                    {
                        "contamination": rate,
                        "seed": seed,
                        "methode": name,
                        "AP": ap,
                    }
                )
    return pd.DataFrame(rows)


def summarize_ap(df_results):
    """Moyenne et écart-type d'AP par méthode et taux de contamination."""
    return (
        df_results.groupby(["methode", "contamination"])["AP"]
        .agg(mean="mean", std="std")
        .reset_index()
    )


def plot_ap_vs_contamination(df_summary, title="Robustesse — AP vs contamination"):
    """Courbe moyenne ± écart-type de l'AP selon le % de contamination."""
    plt.figure(figsize=(8, 5))
    for methode, g in df_summary.groupby("methode"):
        g = g.sort_values("contamination")
        x = 100 * g["contamination"].to_numpy()
        y = g["mean"].to_numpy()
        err = g["std"].fillna(0).to_numpy()
        plt.errorbar(x, y, yerr=err, marker="o", capsize=4, label=methode)

    plt.xlabel("Contamination du train (%)")
    plt.ylabel("Average Precision (AP) sur le test réel")
    plt.title(title)
    plt.xticks([0, 5, 10])
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
