"""
Module d'optimisation pour la réplication de portefeuilles.

Ce module contient les fonctions d'optimisation convexe (MSE, Ridge, Mean-Variance)
avec rebalancement et fenêtres glissantes.
"""

import numpy as np
import cvxpy as cp
from sklearn.covariance import LedoitWolf


def constrained_linear_regression(X_train, y_train):
    """
    Régression linéaire avec contraintes (somme=1, poids entre 0 et 1).
    Minimise : ||y - X @ omega||²
    Sous contraintes :
    - somme(omega) = 1
    - 0 <= omega <= 1
    
    Parameters:
    -----------
    X_train : np.ndarray
        Matrice des rendements sectoriels (n_samples, n_features)
    y_train : np.ndarray
        Vecteur des rendements thématiques (n_samples,)
    
    Returns:
    --------
    np.ndarray
        Vecteur de poids optimaux
    """
    n_features = X_train.shape[1]
    omega = cp.Variable(n_features)
    objective = cp.Minimize(cp.sum_squares(y_train - X_train @ omega))
    constraints = [cp.sum(omega) == 1, omega >= 0, omega <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return omega.value


def walk_forward_rebalancing_sliding_window(X, y, initial_train_size, window_size, 
                                            rebalance_every=1):
    """
    Walk-forward avec fenêtre glissante pour le rebalancement.
    
    Parameters:
    -----------
    X : np.ndarray
        Matrice des rendements sectoriels (n_samples, n_features)
    y : np.ndarray
        Vecteur des rendements thématiques (n_samples,)
    initial_train_size : int
        Taille initiale de l'ensemble d'entraînement
    window_size : int
        Taille de la fenêtre glissante
    rebalance_every : int
        Fréquence de rebalancement (nombre de périodes)
    
    Returns:
    --------
    tuple
        (all_weights, predicted_returns, tracking_error)
    """
    n_months = len(y)
    
    if initial_train_size < window_size:
        raise ValueError(f"initial_train_size ({initial_train_size}) doit être >= window_size ({window_size})")
    
    if initial_train_size >= n_months:
        raise ValueError("Taille d'entraînement initiale trop grande")
    
    all_weights = []
    predicted_returns = []

    # Boucle sur les mois de test
    for t in range(initial_train_size, n_months):
        # Fenêtre glissante : utilise seulement les window_size derniers mois
        # De t-window_size jusqu'à t-1 (excluant t)
        window_start = max(0, t - window_size)  # max pour éviter les indices négatifs
        X_train = X[window_start:t]
        y_train = y[window_start:t]
        
        # Optimisation des poids
        omega_opt = constrained_linear_regression(X_train, y_train)
        all_weights.append(omega_opt)
        
        # Prédiction pour le mois t (utilisant les poids optimisés)
        X_test_t = X[t]
        # Calcul des rendements prédit pour le mois t à partir de omega_opt
        y_pred_t = np.dot(X_test_t, omega_opt)
        predicted_returns.append(y_pred_t)
    
    # Convertir en np.ndarray pour predicted_returns et actual_returns
    predicted_returns = np.array(predicted_returns)
    actual_returns = np.array(y[initial_train_size:])
    tracking_error = np.sqrt(np.mean((actual_returns - predicted_returns) ** 2))

    return all_weights, predicted_returns, tracking_error

def constrained_ridge_regression(X_train, y_train, alpha=0.1):
    """
    Régression Ridge avec contraintes (somme=1, poids entre 0 et 1).
    
    Minimise : ||y - X @ omega||² + alpha * ||omega||²
    Sous contraintes :
    - somme(omega) = 1
    - 0 <= omega <= 1
    
    Parameters:
    -----------
    X_train : np.ndarray
        Matrice des rendements sectoriels (n_samples, n_features)
    y_train : np.ndarray
        Vecteur des rendements thématiques (n_samples,)
    alpha : float
        Paramètre de régularisation L2
    
    Returns:
    --------
    np.ndarray
        Vecteur de poids optimaux
    """
    omega = cp.Variable(X_train.shape[1])
    
    # Fonction objectif : MSE + régularisation Ridge (L2)
    # L = ||y - X@omega||² + alpha * ||omega||²
    objective = cp.Minimize(
        cp.sum_squares(y_train - X_train @ omega) + alpha * cp.sum_squares(omega)
    )
    
    # Contraintes : somme des poids = 1, poids >= 0
    constraints = [
        cp.sum(omega) == 1,
        omega >= 0
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return omega.value

def walk_forward_ridge_rebalancing_sliding_window(X, y, initial_train_size, window_size, 
                                                   alpha=0.1, rebalance_every=1):
    """
    Walk-forward Ridge avec fenêtre glissante.
    
    Parameters:
    -----------
    X : np.ndarray
        Matrice des rendements sectoriels (n_samples, n_features)
    y : np.ndarray
        Vecteur des rendements thématiques (n_samples,)
    initial_train_size : int
        Taille initiale de l'ensemble d'entraînement
    window_size : int
        Taille de la fenêtre glissante
    alpha : float
        Paramètre de régularisation Ridge
    rebalance_every : int
        Fréquence de rebalancement
    
    Returns:
    --------
    tuple
        (all_weights, predicted_returns, tracking_error)
    """
    n_months = len(y)
    
    if initial_train_size < window_size:
        raise ValueError(f"initial_train_size ({initial_train_size}) doit être >= window_size ({window_size})")
    
    if initial_train_size >= n_months:
        raise ValueError("Taille d'entraînement initiale trop grande")
    
    all_weights = []
    predicted_returns = []

    # Boucle sur les mois de test
    for t in range(initial_train_size, n_months):
        # Fenêtre glissante : utilise seulement les window_size derniers mois
        # De t-window_size jusqu'à t-1 (excluant t)
        window_start = max(0, t - window_size)  # max pour éviter les indices négatifs
        X_train = X[window_start:t]
        y_train = y[window_start:t]
        
        # Optimisation des poids avec régularisation Ridge
        omega_opt = constrained_ridge_regression(X_train, y_train, alpha=alpha)
        all_weights.append(omega_opt)
        
        # Prédiction pour le mois t (utilisant les poids optimisés)
        X_test_t = X[t]
        # Calcul des rendements prédit pour le mois t à partir de omega_opt
        y_pred_t = np.dot(X_test_t, omega_opt)
        predicted_returns.append(y_pred_t)
    
    # Convertir en np.ndarray pour predicted_returns et actual_returns
    predicted_returns = np.array(predicted_returns)
    actual_returns = np.array(y[initial_train_size:])
    tracking_error = np.sqrt(np.mean((actual_returns - predicted_returns) ** 2))

    return all_weights, predicted_returns, tracking_error

def find_optimal_ridge_alpha(X, y, initial_train_size, window_size, alpha_range, 
                              validation_split=0.2):
    """
    Trouve le paramètre alpha optimal pour la régularisation Ridge
    en utilisant une validation sur une portion des données d'entraînement.
    
    Args:
        X: Données features (rendements sectoriels)
        y: Target (rendement thématique)
        initial_train_size: Index où commence la période de test
        window_size: Taille de la fenêtre glissante
        alpha_range: Liste ou array des valeurs de alpha à tester
        validation_split: Proportion des données d'entraînement à utiliser pour la validation
    
    Returns:
        optimal_alpha: Valeur optimale de alpha
        results: Dictionnaire avec tracking_error pour chaque alpha
    """
    # Séparer les données d'entraînement en train et validation
    n_train = initial_train_size
    n_val = int(n_train * validation_split)
    train_end = n_train - n_val
    
    results = {}
    
    for alpha in alpha_range:
        try:
            # Utiliser une fenêtre glissante pour la validation
            val_weights, val_predicted, val_tracking_error = \
                walk_forward_ridge_rebalancing_sliding_window(
                    X[:n_train], y[:n_train],
                    initial_train_size=train_end,
                    window_size=window_size,
                    alpha=alpha,
                    rebalance_every=1
                )
            results[alpha] = val_tracking_error
            print(f"Alpha = {alpha:.4f}: Tracking Error = {val_tracking_error:.4%}")
        except Exception as e:
            print(f"Erreur pour alpha = {alpha}: {e}")
            results[alpha] = np.inf
    
    # Trouver l'alpha optimal (minimise tracking error)
    optimal_alpha = min(results, key=results.get)
    
    return optimal_alpha


def mean_variance_tracking_optimization(X_train, y_train):
    """
    Version alternative avec calcul explicite de covariance centrée.
    """
    # Centrer les données (soustraire la moyenne)
    X_train_centered = X_train - np.mean(X_train, axis=0)
    y_train_centered = y_train - np.mean(y_train)
    
    # Matrice de covariance des secteurs
    # Cov(X) = E[(X - E[X])(X - E[X])^T]

    cov_X = LedoitWolf().fit(X_train).covariance_
    
    # Vecteur de covariance entre secteurs et thématique
    # Cov(X, y) = E[(X - E[X])(y - E[y])]
    cov_Xy = np.mean(X_train_centered * y_train_centered[:, np.newaxis], axis=0)
    
    # Définition de la variable d'optimisation
    omega = cp.Variable(X_train.shape[1])
    
    # Objectif : minimiser Var(y - X @ omega)
    # = omega^T @ Cov(X) @ omega - 2 * omega^T @ Cov(X, y) + Var(y)
    objective = cp.Minimize(
        cp.quad_form(omega, cov_X) - 2 * (cov_Xy @ omega)
    )
    
    # Contraintes
    constraints = [
        cp.sum(omega) == 1,
        omega >= 0
    ]
    
    # Résolution
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return omega.value

def walk_forward_mean_variance_tracking(
    X, y, initial_train_size, window_size, rebalance_every=1
):
    """
    Walk-forward validation avec optimisation Mean-Variance Tracking et fenêtre glissante.
    """
    n_months = len(y)
    
    if initial_train_size < window_size:
        raise ValueError(f"initial_train_size ({initial_train_size}) doit être >= window_size ({window_size})")
    
    if initial_train_size >= n_months:
        raise ValueError("Taille d'entraînement initiale trop grande")
    
    all_weights = []
    predicted_returns = []

    # Boucle sur les mois de test
    for t in range(initial_train_size, n_months):
        # Fenêtre glissante
        window_start = max(0, t - window_size)
        X_train = X[window_start:t]
        y_train = y[window_start:t]
        
        # Optimisation Mean-Variance Tracking
        omega_opt = mean_variance_tracking_optimization(X_train, y_train)
        all_weights.append(omega_opt)
        
        # Prédiction pour le mois t
        X_test_t = X[t]
        y_pred_t = np.dot(X_test_t, omega_opt)
        predicted_returns.append(y_pred_t)
    
    predicted_returns = np.array(predicted_returns)
    actual_returns = np.array(y[initial_train_size:])
    tracking_error = np.sqrt(np.mean((actual_returns - predicted_returns) ** 2))

    return all_weights, predicted_returns, tracking_error

import numpy as np
from deap import base, creator, tools, algorithms
import random

def create_genetic_algorithm_setup(n_assets, ngen=100, pop_size=100, 
                                   cxpb=0.5, mutpb=0.2, tournsize=3):
    """
    Configure l'algorithme génétique pour l'optimisation de portefeuille.
    
    Args:
        n_assets: Nombre d'actifs (secteurs)
        ngen: Nombre de générations
        pop_size: Taille de la population
        cxpb: Probabilité de croisement
        mutpb: Probabilité de mutation
        tournsize: Taille du tournoi pour la sélection
    
    Returns:
        toolbox: Toolbox configuré pour l'algorithme génétique
    """
    # Créer les classes fitness et individu
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimiser
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Fonction pour créer un individu (vecteur de poids)
    # Les poids sont initialisés aléatoirement mais seront normalisés
    def create_individual():
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)  # Normaliser pour somme = 1
        return creator.Individual(weights.tolist())
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Fonction d'évaluation (sera définie plus tard avec les données)
    # toolbox.register("evaluate", evaluate_portfolio)
    
    # Opérateurs génétiques
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Croisement blend
    toolbox.register("mutate", mutate_weights, indpb=0.1, sigma=0.1)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    
    return toolbox

def mutate_weights(individual, indpb, sigma):
    """
    Mutation des poids avec respect des contraintes.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            # Mutation gaussienne
            individual[i] += random.gauss(0, sigma)
            individual[i] = max(0, individual[i])  # Contrainte >= 0
    
    # Renormaliser pour respecter la contrainte somme = 1
    total = sum(individual)
    if total > 0:
        for i in range(len(individual)):
            individual[i] = individual[i] / total
    else:
        # Si tous les poids sont négatifs, réinitialiser
        individual[:] = [1.0/len(individual)] * len(individual)
    
    return individual,

def evaluate_portfolio_fitness(individual, X_train, y_train):
    """
    Fonction de fitness : tracking error (RMSE).
    
    Args:
        individual: Vecteur de poids (individu)
        X_train: Matrice des rendements sectoriels
        y_train: Vecteur des rendements thématiques
    
    Returns:
        tracking_error: RMSE (à minimiser)
    """
    weights = np.array(individual)
    
    # Calculer les rendements prédits
    predicted = X_train @ weights
    
    # Calculer le RMSE (tracking error)
    mse = np.mean((y_train - predicted) ** 2)
    rmse = np.sqrt(mse)
    
    return (rmse,)

def optimize_portfolio_genetic(X_train, y_train, ngen=100, pop_size=100, 
                               cxpb=0.5, mutpb=0.2, verbose=False):
    """
    Optimise les poids du portefeuille avec un algorithme génétique.
    
    Args:
        X_train: Matrice des rendements sectoriels
        y_train: Vecteur des rendements thématiques
        ngen: Nombre de générations
        pop_size: Taille de la population
        cxpb: Probabilité de croisement
        mutpb: Probabilité de mutation
        verbose: Afficher les statistiques
    
    Returns:
        best_weights: Meilleurs poids trouvés
        best_fitness: Fitness du meilleur individu
        log: Historique de l'évolution
    """
    n_assets = X_train.shape[1]
    
    # Créer le setup de l'algorithme génétique
    toolbox = create_genetic_algorithm_setup(n_assets, ngen, pop_size, cxpb, mutpb)
    
    # Définir la fonction d'évaluation avec les données
    def evaluate(individual):
        return evaluate_portfolio_fitness(individual, X_train, y_train)
    
    toolbox.register("evaluate", evaluate)
    
    # Créer la population initiale
    population = toolbox.population(n=pop_size)
    
    # Statistiques
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Hall of Fame (meilleurs individus)
    hof = tools.HallOfFame(1)
    
    # Exécuter l'algorithme génétique
    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=cxpb, mutpb=mutpb, ngen=ngen,
        stats=stats, halloffame=hof, verbose=verbose
    )
    
    best_weights = np.array(hof[0])
    best_fitness = hof[0].fitness.values[0]
    
    return best_weights, best_fitness, logbook

def walk_forward_genetic_tracking(
    X, y, initial_train_size, window_size, ngen=50, pop_size=50,
    cxpb=0.5, mutpb=0.2, rebalance_every=1, verbose=False):

    """
    Walk-forward validation avec algorithme génétique et fenêtre glissante.
    
    Args:
        X: Données features (rendements sectoriels)
        y: Target (rendement thématique)
        initial_train_size: Index où commence la période de test
        window_size: Taille de la fenêtre glissante
        ngen: Nombre de générations par optimisation
        pop_size: Taille de la population
        cxpb: Probabilité de croisement
        mutpb: Probabilité de mutation
        rebalance_every: Fréquence de rebalancement
        verbose: Afficher les statistiques
    
    Returns:
        all_weights: Liste des poids optimaux à chaque rebalancement
        predicted_returns: Rendements prédits sur la période de test
        tracking_error: Tracking error pour la période de test
        fitness_history: Historique des fitness à chaque période
    """
    n_months = len(y)
    
    if initial_train_size < window_size:
        raise ValueError(f"initial_train_size ({initial_train_size}) doit être >= window_size ({window_size})")
    
    all_weights = []
    predicted_returns = []
    fitness_history = []
    
    print(f"Optimisation avec algorithme génétique (window={window_size}, pop={pop_size}, gen={ngen})...")
    
    # Boucle sur les mois de test
    for t in range(initial_train_size, n_months):
        if (t - initial_train_size) % 10 == 0:
            print(f"  Période {t - initial_train_size + 1}/{n_months - initial_train_size}")
        
        # Fenêtre glissante
        window_start = max(0, t - window_size)
        X_train = X[window_start:t]
        y_train = y[window_start:t]
        
        # Optimisation avec algorithme génétique
        best_weights, best_fitness, logbook = optimize_portfolio_genetic(
            X_train, y_train,
            ngen=ngen,
            pop_size=pop_size,
            cxpb=cxpb,
            mutpb=mutpb,
            verbose=False
        )
        
        all_weights.append(best_weights)
        fitness_history.append({
            'fitness': best_fitness,
            'generation': len(logbook)
        })
        
        # Prédiction pour le mois t
        X_test_t = X[t]
        y_pred_t = np.dot(X_test_t, best_weights)
        predicted_returns.append(y_pred_t)
    
    # Calculer le tracking error
    predicted_returns = np.array(predicted_returns)
    actual_returns = np.array(y[initial_train_size:])
    tracking_error = np.sqrt(np.mean((actual_returns - predicted_returns) ** 2))
    
    print(f"✓ Optimisation terminée. Tracking Error: {tracking_error:.4%}")
    
    return all_weights, predicted_returns, tracking_error, fitness_history