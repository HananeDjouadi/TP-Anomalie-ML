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
