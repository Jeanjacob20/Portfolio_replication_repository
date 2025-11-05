"""
Module de traitement des données pour la réplication de portefeuilles.

Ce module contient les fonctions pour charger et préparer les données
de rendements MSCI (secteurs, thématiques, benchmark).
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_returns_data(file_path, date_format='%d/%m/%Y'):
    """
    Charge les fichiers CSV contenant les retours MSCI.
    
    Parameters:
    -----------
    file_path : str
        Chemin vers le fichier CSV
    date_format : str
        Format de la date (par défaut: '%d/%m/%Y')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame avec 'date' comme colonne et les colonnes d'actifs
    """
    df = pd.read_csv(file_path, sep=';')
    
    # Convertir la colonne DATE en datetime
    df['date'] = pd.to_datetime(df['DATE'], format=date_format)
    
    # Supprimer la colonne DATE originale
    df = df.drop('DATE', axis=1)
    
    # Convertir toutes les colonnes numériques (sauf date) en float
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Supprimer les lignes dont toutes les colonnes d'actifs sont NaN (ie weekends)
    asset_columns = [col for col in df.columns if col != 'date']
    df = df.dropna(subset=asset_columns, how='all').copy()
    
    # Trier par date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def compose_monthly_returns(df):
    """
    Compose les rendements journaliers pour obtenir des rendements mensuels.
    
    Pour chaque mois, calcule : (1+r₁) × (1+r₂) × ... × (1+rₙ) - 1
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec colonne 'date' et colonnes de rendements journaliers
    
    Returns:
    --------
    pd.DataFrame
        DataFrame avec rendements mensuels composés
    """
    df = df.set_index('date')
    # Applique la composition (1+r1)*(1+r2)*...-1 sur chaque colonne, par mois calendaire
    return ((1 + df).resample('M').prod() - 1).reset_index()

def rebalanced_equal_weighted_portfolio_clean(df, freq='M'):
    """
    Calcule la performance d'un portefeuille équipondéré rééquilibré à une fréquence donnée.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame contenant une colonne 'date' et des colonnes de rendements par actif.
    freq : str
        Fréquence de rééquilibrage pandas ('M' pour mensuelle, 'W' pour hebdomadaire, etc)

    Returns:
    --------
    pd.DataFrame
        DataFrame avec les dates de rééquilibrage et les rendements du portefeuille équipondéré.
    """
    df = df.set_index('date').sort_index()
    asset_cols = df.columns
    if len(asset_cols) == 0:
        raise ValueError("No asset columns found in input DataFrame.")
    result = []
    prev_weights = None

    for period_end, group in df.groupby(pd.Grouper(freq=freq)):
        if group.empty:
            continue
        # Rééquilibrage à poids égaux (hors NaN, pour chaque période)
        notnan = group[asset_cols].notna().all(axis=0)
        valid_assets = asset_cols[notnan]
        if len(valid_assets) == 0:
            period_return = float('nan')
        else:
            weights = pd.Series(1/len(valid_assets), index=valid_assets)
            period_returns = ((1 + group[valid_assets]).prod() - 1)
            period_return = (period_returns * weights).sum()
        result.append({'date': period_end, 'return': period_return})
    return pd.DataFrame(result)

