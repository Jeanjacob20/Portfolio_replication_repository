"""
Module d'évaluation des modèles de réplication.

Ce module contient les fonctions pour calculer les métriques de performance
et de stabilité des modèles.
"""

import numpy as np


def evaluate_model(X, y, omega):
    """
    Calcule le tracking error (RMSE) et les rendements prédits.
    
    Parameters:
    -----------
    X : np.ndarray
        Matrice des rendements sectoriels (n_samples, n_features)
    y : np.ndarray
        Vecteur des rendements thématiques réels (n_samples,)
    omega : np.ndarray
        Vecteur de poids optimaux (n_features,)
    
    Returns:
    --------
    tuple
        (tracking_error, predicted_returns)
    """
    predicted_returns = X @ omega
    tracking_error = np.sqrt(np.mean((y - predicted_returns) ** 2))
    return tracking_error, predicted_returns


def calculate_weight_stability(all_weights):
    """
    Calcule la stabilité des poids optimaux.
    
    Métriques calculées:
    - Mean Absolute Deviation (MAD) : moyenne des écarts absolus moyens
    - Turnover : moyenne du turnover entre rébalancements
    - Volatility : volatilité moyenne des poids
    
    Parameters:
    -----------
    all_weights : list of np.ndarray
        Liste des vecteurs de poids à chaque période de rebalancement
    
    Returns:
    --------
    dict
        Dictionnaire avec les métriques de stabilité
    """
    if len(all_weights) < 2:
        return {
            'mean_abs_deviation': np.nan,
            'mean_turnover': np.nan,
            'mean_weight_volatility': np.nan
        }
    
    weights_array = np.array(all_weights)  # (n_periods, n_sectors)
    
    # 1. Mean Absolute Deviation (MAD) - moyenne des écarts absolus par rapport à la moyenne
    mean_weights = np.mean(weights_array, axis=0)
    mad = np.mean(np.abs(weights_array - mean_weights), axis=1).mean()
    
    # 2. Turnover - somme des changements absolus de poids entre périodes
    turnovers = []
    for i in range(1, len(all_weights)):
        turnover = np.sum(np.abs(weights_array[i] - weights_array[i-1]))
        turnovers.append(turnover)
    mean_turnover = np.mean(turnovers) if turnovers else np.nan
    
    # 3. Volatilité moyenne des poids (écart-type moyen)
    weight_std = np.std(weights_array, axis=0)
    mean_weight_volatility = np.mean(weight_std)
    
    return {
        'mean_abs_deviation': mad,
        'mean_turnover': mean_turnover,
        'mean_weight_volatility': mean_weight_volatility
    }


def calculate_rmse(y_true, y_pred):
    """
    Calcule le Root Mean Squared Error (RMSE).
    
    Parameters:
    -----------
    y_true : np.ndarray
        Valeurs réelles
    y_pred : np.ndarray
        Valeurs prédites
    
    Returns:
    --------
    float
        RMSE
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


