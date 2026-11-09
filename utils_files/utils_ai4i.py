import numpy as np
import pandas as pd


DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]


def load_ai4i(path="../data/ai4i2020.csv"):
    """
    Charge AI4I, cible = Machine failure.
    Retire identifiants et colonnes de types de panne (fuite vers la cible).
    """
    df = pd.read_csv(path)

    missing = [c for c in DROP_COLS + ["Machine failure"] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    y = df["Machine failure"].astype(int).to_numpy()
    X = df.drop(columns=DROP_COLS + ["Machine failure"])
    return X, y, df


def check_ai4i(X, y, df=None):
    """Vérifie format, NaN, fréquence des pannes et colonnes restantes."""
    print("=== AI4I 2020 ===")
    print("Shape X :", X.shape)
    print("Colonnes X :", list(X.columns))
    print("Valeurs manquantes :\n", X.isna().sum())
    print("Nb pannes (1) :", int(y.sum()), f"({100 * y.mean():.2f} %)")
    print("Nb normaux (0) :", int((y == 0).sum()))
    if df is not None:
        print("\nColonnes retirées (fuite / id) :", DROP_COLS)
        print("Aperçu :")
        print(X.head())


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Réutiliser TELS QUELS depuis utils_credit (sans les modifier) :
# evaluer_seuil, trouver_meilleur_seuil(min_precision=0.90),
# tracer_courbes_roc_pr(...), tracer_f1_balanced_acc(...)
# NB : split_data / scale_amount sont spécifiques Credit Card → non réutilisables ici.


def split_ai4i(X, y, random_state=42):
    """Découpage stratifié 60% train / 20% val / 20% test (protocole commun)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_preprocessor(X_train):
    """OneHotEncoder pour Type + StandardScaler pour les numériques (fit train only)."""
    cat_cols = ["Type"] if "Type" in X_train.columns else []
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    return preprocessor, cat_cols, num_cols


def preprocess_ai4i(X_train, X_val, X_test):
    """Prétraitement AI4I : fit sur train uniquement, puis transform val/test."""
    preprocessor, cat_cols, num_cols = make_preprocessor(X_train)
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)
    return X_train_p, X_val_p, X_test_p, preprocessor, cat_cols, num_cols


import time
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE


def fit_isolation_forest_ai4i(
    X_train,
    X_val,
    X_test,
    n_estimators=100,
    max_samples=1024,
    random_state=42,
    n_jobs=-1,
):
    """
    Isolation Forest (hyperparams retenus Credit Card) réentraîné sur AI4I :
    n_estimators=100, max_samples=1024, contamination="auto".
    Scores : plus haut = plus suspect (-score_samples).
    Retourne model, scores_val, scores_test, temps.
    """
    t0 = time.time()
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination="auto",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train)
    elapsed = time.time() - t0
    scores_val = -model.score_samples(X_val)
    scores_test = -model.score_samples(X_test)
    return model, scores_val, scores_test, elapsed


def fit_logistic_regression_ai4i(
    X_train, y_train, X_val, X_test, C=0.01, max_iter=1000, random_state=42
):
    """
    Régression logistique (meilleur modèle Credit Card) réentraînée sur AI4I :
    C=0.01 + SMOTE sur le train uniquement (comme retenu en Partie 2/3).
    Retourne model, scores_val, scores_test, temps (proba classe 1).
    """
    t0 = time.time()
    smote = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_train_sm, y_train_sm)
    elapsed = time.time() - t0
    scores_val = model.predict_proba(X_val)[:, 1]
    scores_test = model.predict_proba(X_test)[:, 1]
    return model, scores_val, scores_test, elapsed
