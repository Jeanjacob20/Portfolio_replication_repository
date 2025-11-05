"""
Évaluation de tous les modèles pour tous les portefeuilles thématiques.

Ce script reproduit la logique du notebook Test1.ipynb mais pour tous les portefeuilles
et génère des statistiques descriptives sur les RMSE et la stabilité.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Ajouter le répertoire racine au path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.models import (
    constrained_linear_regression,
    walk_forward_rebalancing_sliding_window,
    walk_forward_ridge_rebalancing_sliding_window,
    walk_forward_mean_variance_tracking,
    walk_forward_genetic_tracking,
    find_optimal_ridge_alpha
)
from src.evaluation import calculate_weight_stability
from src.data_processing import load_returns_data, compose_monthly_returns


def evaluate_single_portfolio(thematic_name, thematic_df, sector_df, 
                               split_ratio=0.7, window_size_linear=12, 
                               window_size_advanced=18, verbose=True):
    """
    Évalue tous les modèles pour un portefeuille thématique donné.
    Basé sur la logique du notebook Test1.ipynb.
    
    Parameters:
    -----------
    thematic_name : str
        Nom du portefeuille thématique
    thematic_df : pd.DataFrame
        DataFrame avec rendements mensuels thématiques
    sector_df : pd.DataFrame
        DataFrame avec rendements mensuels sectoriels
    split_ratio : float
        Ratio de séparation train/test (défaut: 0.7)
    window_size_linear : int
        Taille de fenêtre pour modèles linéaires (défaut: 12)
    window_size_advanced : int
        Taille de fenêtre pour modèles avancés (défaut: 18)
    verbose : bool
        Afficher les messages de progression
    
    Returns:
    --------
    dict
        Dictionnaire avec tous les résultats (RMSE, stabilité, etc.)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Portefeuille: {thematic_name}")
        print(f"{'='*80}")
    
    # Préparer les données
    dates = thematic_df['date'].values
    X_full = sector_df.drop(columns='date').values
    y_full = thematic_df[thematic_name].values
    
    # Vérifications
    n_months = len(y_full)
    if n_months < window_size_advanced + 12:
        if verbose:
            print(f"  ⚠️  Pas assez de données (n={n_months})")
        return None
    
    if np.isnan(y_full).sum() > len(y_full) * 0.1:
        if verbose:
            print(f"  ⚠️  Trop de valeurs manquantes")
        return None
    
    # Découpe in-sample / out-of-sample (70%/30%)
    split_idx = int(split_ratio * len(y_full))
    X_in, y_in = X_full[:split_idx], y_full[:split_idx]
    X_out, y_out = X_full[split_idx:], y_full[split_idx:]
    
    # Paramètres pour walk-forward
    initial_train_size = len(y_in)  # calibré pour que IS = in-sample
    rebalance_every = 1
    
    results = {
        'Portefeuille_Thematique': thematic_name,
        'N_Observations': n_months,
        'Split_Idx': split_idx,
        'N_Train': len(y_in),
        'N_Test': len(y_out)
    }
    
    # ========================================================================
    # MODÈLE 1: Constrained Linear Regression (Simple)
    # ========================================================================
    try:
        if verbose:
            print("  → Modèle 1: Constrained Linear (Simple)...")
        
        w1 = constrained_linear_regression(X_in, y_in)
        yhat1 = X_out @ w1
        rmse1 = np.sqrt(np.mean((y_out - yhat1)**2))
        
        results['RMSE_Linear_Simple'] = rmse1
        # Pas de stabilité pour ce modèle (un seul poids)
        results['Stabilite_MAD_Linear_Simple'] = 0.0
        results['Stabilite_Turnover_Linear_Simple'] = 0.0
        results['Stabilite_Vol_Linear_Simple'] = 0.0
        
        if verbose:
            print(f"     ✓ RMSE: {rmse1:.4%}")
            
    except Exception as e:
        if verbose:
            print(f"     ✗ Erreur: {e}")
        results['RMSE_Linear_Simple'] = np.nan
        results['Stabilite_MAD_Linear_Simple'] = np.nan
        results['Stabilite_Turnover_Linear_Simple'] = np.nan
        results['Stabilite_Vol_Linear_Simple'] = np.nan
    
    # ========================================================================
    # MODÈLE 2: Constrained Linear Regression (Walk-Forward)
    # ========================================================================
    try:
        if verbose:
            print("  → Modèle 2: Constrained Linear (Walk-Forward)...")
        
        w2, yhat2, rmse2 = walk_forward_rebalancing_sliding_window(
            X_full, y_full, initial_train_size, window_size_linear, rebalance_every
        )
        
        results['RMSE_Linear_WF'] = rmse2
        stability_wf = calculate_weight_stability(w2)
        results['Stabilite_MAD_Linear_WF'] = stability_wf['mean_abs_deviation']
        results['Stabilite_Turnover_Linear_WF'] = stability_wf['mean_turnover']
        results['Stabilite_Vol_Linear_WF'] = stability_wf['mean_weight_volatility']
        
        if verbose:
            print(f"     ✓ RMSE: {rmse2:.4%}")
            
    except Exception as e:
        if verbose:
            print(f"     ✗ Erreur: {e}")
        results['RMSE_Linear_WF'] = np.nan
        results['Stabilite_MAD_Linear_WF'] = np.nan
        results['Stabilite_Turnover_Linear_WF'] = np.nan
        results['Stabilite_Vol_Linear_WF'] = np.nan
    
    # ========================================================================
    # MODÈLE 3: Constrained Ridge Regression
    # ========================================================================
    try:
        if verbose:
            print("  → Modèle 3: Constrained Ridge...")
        
        # Trouver l'alpha optimal
        alpha_range = np.logspace(-3, 1, 20)  # De 0.001 à 10, 20 valeurs
        optimal_alpha = find_optimal_ridge_alpha(
            X_full, y_full,
            split_idx,
            window_size_linear,
            alpha_range,
            validation_split=0.2
        )
        
        # Appliquer le modèle Ridge avec l'alpha optimal
        w3, yhat3, rmse3 = walk_forward_ridge_rebalancing_sliding_window(
            X_full, y_full, initial_train_size, window_size_linear, 
            optimal_alpha, rebalance_every
        )
        
        results['RMSE_Ridge'] = rmse3
        results['Ridge_Alpha'] = optimal_alpha
        stability_ridge = calculate_weight_stability(w3)
        results['Stabilite_MAD_Ridge'] = stability_ridge['mean_abs_deviation']
        results['Stabilite_Turnover_Ridge'] = stability_ridge['mean_turnover']
        results['Stabilite_Vol_Ridge'] = stability_ridge['mean_weight_volatility']
        
        if verbose:
            print(f"     ✓ RMSE: {rmse3:.4%} (alpha={optimal_alpha:.4f})")
            
    except Exception as e:
        if verbose:
            print(f"     ✗ Erreur: {e}")
        results['RMSE_Ridge'] = np.nan
        results['Ridge_Alpha'] = np.nan
        results['Stabilite_MAD_Ridge'] = np.nan
        results['Stabilite_Turnover_Ridge'] = np.nan
        results['Stabilite_Vol_Ridge'] = np.nan
    
    # ========================================================================
    # MODÈLE 4: Genetic Algorithm
    # ========================================================================
    try:
        if verbose:
            print("  → Modèle 4: Genetic Algorithm...")
        
        # Paramètres de l'algorithme génétique
        ngen = 50
        pop_size = 100
        cxpb = 0.5
        mutpb = 0.2
        
        w4, yhat4, rmse4, fitness_history = walk_forward_genetic_tracking(
            X_full, y_full,
            initial_train_size,
            window_size_advanced,
            ngen=ngen,
            pop_size=pop_size,
            cxpb=cxpb,
            mutpb=mutpb,
            rebalance_every=1,
            verbose=False
        )
        
        results['RMSE_Genetic'] = rmse4
        stability_genetic = calculate_weight_stability(w4)
        results['Stabilite_MAD_Genetic'] = stability_genetic['mean_abs_deviation']
        results['Stabilite_Turnover_Genetic'] = stability_genetic['mean_turnover']
        results['Stabilite_Vol_Genetic'] = stability_genetic['mean_weight_volatility']
        
        if verbose:
            print(f"     ✓ RMSE: {rmse4:.4%}")
            
    except Exception as e:
        if verbose:
            print(f"     ✗ Erreur: {e}")
        results['RMSE_Genetic'] = np.nan
        results['Stabilite_MAD_Genetic'] = np.nan
        results['Stabilite_Turnover_Genetic'] = np.nan
        results['Stabilite_Vol_Genetic'] = np.nan
    
    # ========================================================================
    # MODÈLE 5: Mean-Variance Tracking
    # ========================================================================
    try:
        if verbose:
            print("  → Modèle 5: Mean-Variance Tracking...")
        
        w5, yhat5, mv5 = walk_forward_mean_variance_tracking(
            X_full, y_full, 
            initial_train_size,
            window_size_advanced,
            rebalance_every=1
        )

        rmse5 = np.sqrt(np.mean((y_out - yhat5)**2))
        
        results['RMSE_MeanVar'] = rmse5
        stability_mv = calculate_weight_stability(w5)
        results['Stabilite_MAD_MeanVar'] = stability_mv['mean_abs_deviation']
        results['Stabilite_Turnover_MeanVar'] = stability_mv['mean_turnover']
        results['Stabilite_Vol_MeanVar'] = stability_mv['mean_weight_volatility']
        
        if verbose:
            print(f"     ✓ RMSE: {rmse5:.4%}")
            
    except Exception as e:
        if verbose:
            print(f"     ✗ Erreur: {e}")
        results['RMSE_MeanVar'] = np.nan
        results['Stabilite_MAD_MeanVar'] = np.nan
        results['Stabilite_Turnover_MeanVar'] = np.nan
        results['Stabilite_Vol_MeanVar'] = np.nan
    
    if verbose:
        print(f"  ✓ Portefeuille terminé")
    
    return results


def main():
    """Fonction principale"""
    print("="*80)
    print("ÉVALUATION DE TOUS LES MODÈLES POUR TOUS LES PORTEFEUILLES THÉMATIQUES")
    print("="*80)
    
    # Paramètres
    split_ratio = 0.7
    window_size_linear = 12
    window_size_advanced = 18
    
    print(f"\nParamètres:")
    print(f"  - Split ratio: {split_ratio} (70% train / 30% test)")
    print(f"  - Window size (linéaires): {window_size_linear} mois")
    print(f"  - Window size (avancés): {window_size_advanced} mois")
    
    # 1. Chargement et préparation des données
    print("\n" + "="*80)
    print("ÉTAPE 1: Chargement et préparation des données")
    print("="*80)
    
    thematic_file = "data/raw/msci_acwi_imi_thematics_daily_returns_202007_202508.csv"
    sector_file = "data/raw/msci_acwi_imi_sectors_daily_returns_202007_202508.csv"
    
    thematic_df = load_returns_data(thematic_file)
    sector_df = load_returns_data(sector_file)
    
    # Nettoyage
    if not thematic_df.empty:
        debut_date = thematic_df['date'].min()
        drop_cols = [
            col for col in thematic_df.columns if col != 'date'
            and pd.isna(thematic_df.loc[thematic_df['date'] == debut_date, col]).any()
        ]
        thematic_df = thematic_df.drop(columns=drop_cols)
    
    thematic_df = thematic_df.iloc[:-1, :]
    sector_df = sector_df.iloc[:-1, :]
    
    # Transformation en rendements mensuels
    thematic_monthly = compose_monthly_returns(thematic_df)
    sector_monthly = compose_monthly_returns(sector_df)
    
    thematic_df = thematic_monthly.copy()
    sector_df = sector_monthly.copy()
    
    thematic_cols = [col for col in thematic_df.columns if col != 'date']
    
    print(f"  ✓ {len(thematic_cols)} portefeuilles thématiques trouvés")
    print(f"  ✓ {len(sector_df.columns)-1} secteurs")
    
    # 2. Évaluation pour chaque portefeuille thématique
    print("\n" + "="*80)
    print("ÉTAPE 2: Évaluation des modèles pour chaque portefeuille")
    print("="*80)
    
    all_results = []
    
    for idx, thematic_name in enumerate(thematic_cols):
        print(f"\n[{idx+1}/{len(thematic_cols)}]")
        result = evaluate_single_portfolio(
            thematic_name, thematic_df, sector_df,
            split_ratio, window_size_linear, window_size_advanced,
            verbose=True
        )
        
        if result is not None:
            all_results.append(result)
    
    # 3. Création du DataFrame récapitulatif
    print("\n" + "="*80)
    print("ÉTAPE 3: Création du tableau récapitulatif")
    print("="*80)
    
    df_results = pd.DataFrame(all_results)
    
    # Réorganiser les colonnes
    column_order = [
        'Portefeuille_Thematique', 'N_Observations', 'N_Train', 'N_Test',
        'RMSE_Linear_Simple', 'Stabilite_MAD_Linear_Simple',
        'Stabilite_Turnover_Linear_Simple', 'Stabilite_Vol_Linear_Simple',
        'RMSE_Linear_WF', 'Stabilite_MAD_Linear_WF',
        'Stabilite_Turnover_Linear_WF', 'Stabilite_Vol_Linear_WF',
        'RMSE_Ridge', 'Ridge_Alpha', 'Stabilite_MAD_Ridge',
        'Stabilite_Turnover_Ridge', 'Stabilite_Vol_Ridge',
        'RMSE_Genetic', 'Stabilite_MAD_Genetic',
        'Stabilite_Turnover_Genetic', 'Stabilite_Vol_Genetic',
        'RMSE_MeanVar', 'Stabilite_MAD_MeanVar',
        'Stabilite_Turnover_MeanVar', 'Stabilite_Vol_MeanVar'
    ]
    
    existing_cols = [col for col in column_order if col in df_results.columns]
    df_results = df_results[existing_cols]
    
    # 4. Sauvegarde
    output_file = "results/tables/evaluation_all_models_all_portfolios.csv"
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Résultats sauvegardés dans: {output_file}")
    
    # 5. Statistiques descriptives sur les RMSE
    print("\n" + "="*80)
    print("ÉTAPE 4: Statistiques descriptives - RMSE")
    print("="*80)
    
    models_rmse = {
        'Linear_Simple': 'RMSE_Linear_Simple',
        'Linear_WF': 'RMSE_Linear_WF',
        'Ridge': 'RMSE_Ridge',
        'Genetic': 'RMSE_Genetic',
        'MeanVar': 'RMSE_MeanVar'
    }
    
    stats_rmse = []
    
    for model_name, rmse_col in models_rmse.items():
        if rmse_col in df_results.columns:
            rmse_values = df_results[rmse_col].dropna()
            if len(rmse_values) > 0:
                stats_rmse.append({
                    'Modèle': model_name,
                    'Moyenne': rmse_values.mean(),
                    'Médiane': rmse_values.median(),
                    'Min': rmse_values.min(),
                    'Max': rmse_values.max(),
                    'Écart-type': rmse_values.std(),
                    'Q1 (25%)': rmse_values.quantile(0.25),
                    'Q3 (75%)': rmse_values.quantile(0.75),
                    'N_Valides': len(rmse_values)
                })
    
    df_stats_rmse = pd.DataFrame(stats_rmse)
    print("\nStatistiques descriptives des RMSE:")
    print(df_stats_rmse.to_string(index=False))
    
    # 6. Statistiques descriptives sur la stabilité
    print("\n" + "="*80)
    print("ÉTAPE 5: Statistiques descriptives - Stabilité (MAD)")
    print("="*80)
    
    models_stability = {
        'Linear_WF': 'Stabilite_MAD_Linear_WF',
        'Ridge': 'Stabilite_MAD_Ridge',
        'Genetic': 'Stabilite_MAD_Genetic',
        'MeanVar': 'Stabilite_MAD_MeanVar'
    }
    
    stats_stability = []
    
    for model_name, mad_col in models_stability.items():
        if mad_col in df_results.columns:
            mad_values = df_results[mad_col].dropna()
            if len(mad_values) > 0:
                stats_stability.append({
                    'Modèle': model_name,
                    'MAD Moyenne': mad_values.mean(),
                    'MAD Médiane': mad_values.median(),
                    'MAD Min': mad_values.min(),
                    'MAD Max': mad_values.max(),
                    'MAD Écart-type': mad_values.std(),
                    'N_Valides': len(mad_values)
                })
    
    df_stats_stability = pd.DataFrame(stats_stability)
    print("\nStatistiques descriptives de la stabilité (MAD):")
    print(df_stats_stability.to_string(index=False))
    
    # 7. Statistiques sur le Turnover
    print("\n" + "="*80)
    print("ÉTAPE 6: Statistiques descriptives - Turnover")
    print("="*80)
    
    models_turnover = {
        'Linear_WF': 'Stabilite_Turnover_Linear_WF',
        'Ridge': 'Stabilite_Turnover_Ridge',
        'Genetic': 'Stabilite_Turnover_Genetic',
        'MeanVar': 'Stabilite_Turnover_MeanVar'
    }
    
    stats_turnover = []
    
    for model_name, turnover_col in models_turnover.items():
        if turnover_col in df_results.columns:
            turnover_values = df_results[turnover_col].dropna()
            if len(turnover_values) > 0:
                stats_turnover.append({
                    'Modèle': model_name,
                    'Turnover Moyen': turnover_values.mean(),
                    'Turnover Médian': turnover_values.median(),
                    'Turnover Min': turnover_values.min(),
                    'Turnover Max': turnover_values.max(),
                    'Turnover Écart-type': turnover_values.std(),
                    'N_Valides': len(turnover_values)
                })
    
    df_stats_turnover = pd.DataFrame(stats_turnover)
    print("\nStatistiques descriptives du Turnover:")
    print(df_stats_turnover.to_string(index=False))
    
    # 8. Sauvegarder les statistiques
    stats_file = "results/tables/statistiques_descriptives_rmse_stabilite.csv"
    
    # Combiner toutes les statistiques
    stats_combined = pd.concat([
        df_stats_rmse.set_index('Modèle'),
        df_stats_stability.set_index('Modèle'),
        df_stats_turnover.set_index('Modèle')
    ], axis=1)
    
    stats_combined.to_csv(stats_file, encoding='utf-8-sig')
    print(f"\n✓ Statistiques descriptives sauvegardées dans: {stats_file}")
    
    # 9. Afficher le tableau récapitulatif
    print("\n" + "="*80)
    print("TABLEAU RÉCAPITULATIF COMPLET")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df_results.to_string(index=False))
    
    print(f"\n✓ Évaluation terminée pour {len(all_results)} portefeuilles")
    print(f"✓ Fichiers générés:")
    print(f"  - {output_file}")
    print(f"  - {stats_file}")
    
    return df_results, df_stats_rmse, df_stats_stability, df_stats_turnover


if __name__ == "__main__":
    df_results, stats_rmse, stats_stability, stats_turnover = main()