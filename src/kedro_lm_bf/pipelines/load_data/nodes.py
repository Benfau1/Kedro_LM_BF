import pandas as pd













def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données en supprimant les valeurs aberrantes et interpolant les valeurs manquantes.

    Args:
        data (pd.DataFrame): Jeu de données brut.

    Returns:
        pd.DataFrame: Jeu de données nettoyé.
    """

    # Suppression des valeurs aberrantes
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data_cleaned = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Interpolation des valeurs manquantes
    data_cleaned = data_cleaned.interpolate(method="linear")

    return data_cleaned
