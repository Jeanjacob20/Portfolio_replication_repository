# Portfolio Replication - Réplication de Portefeuilles Thématiques

## 📋 Description

Ce projet vise à développer et comparer des modèles d'optimisation pour répliquer les performances de portefeuilles thématiques via des portefeuilles sectoriels. 

Le projet implémente et évalue cinq modèles d'optimisation différents :
- **Régression linéaire contrainte** (simple et avec walk-forward)
- **Régression Ridge** avec régularisation L2
- **Mean-Variance Tracking** basé sur l'optimisation de la variance
- **Algorithme génétique** Pour une méthode d'optimisation stochastique 

## 🚀 Installation

### Prérequis
- Python 3.9 ou supérieur
- pip ou conda

### Installation des dépendances

```bash
pip install -r requirements.txt
```

Les principales dépendances incluent :
- `pandas`, `numpy` : manipulation de données
- `cvxpy` : optimisation convexe
- `scikit-learn` : outils de machine learning
- `deap` : algorithmes génétiques
- `matplotlib`, `seaborn` : visualisation

## 📁 Structure du projet

```plaintext
Portfolio-replication/
├── data/                           # Données brutes (CSV)
│   └── raw/                       
│       ├── msci_acwi_imi_thematics_daily_returns_202007_202508.csv  
│       ├── msci_acwi_imi_sectors_daily_returns_202007_202508.csv   
│       └── msci_acwi_imi_daily_returns_202007_202508.csv            
│
├── src/                            # Code source modulaire
│   ├── __init__.py                 # Package principal
│   ├── data_processing.py          # Chargement et transformation des données
│   ├── evaluation.py               # Métriques d'évaluation (RMSE, stabilité, turnover)
│   └── models/                     # Stocke les fonctions de réplications
│       ├── __init__.py             # Exports des fonctions de réplications
│       └── optimization.py         # Implémentation des 5 modèles
│
├── results/                        # Résultats et outputs
│   ├── figures/                    # Graphiques générés
│   │   ├── boxplot_rmse_comparaison.png          
│   │   └── turnover_par_methode.png               
│   └── tables/                     # Tableaux de résultats (CSV)
│       ├── evaluation_all_models_all_portfolios.csv  
│       └── statistiques_descriptives_rmse_stabilite.csv  
│
├── docs/                           # Documentation
│   └── Sujet_Technique_Stage_Quant.pdf   # Document technique de référence
│
├── evaluate_all_ptf.py             # Script principal d'évaluation
│
├── Notebooks d'analyse :
│   ├── Model_selection.ipynb       # Explication des modèles utilisés
│   ├── Statistics_all_portfolios.ipynb   # Statistiques sur les méthodes
│   ├── Statistics_Specific_portfolio.ipynb   # Analyse détaillée pour un portefeuille spécifique
│   ├── Statistiques_descriptives.ipynb   # Statistiques descriptives générales
│   └── UNACHIVED_ML_model.ipynb    # Propositions de Modèles ML non aboutis
│
├── requirements.txt                # Dépendances Python
└── README.md                                     
```

### `Statistics_all_portfolios.ipynb`
Analyse comparative de tous les portefeuilles thématiques :
- Chargement et visualisation des résultats du CSV principal
- Statistiques descriptives sur les RMSE par modèle
- Statistiques descriptives sur la stabilité (MAD, Turnover)
- Visualisations comparatives (boxplots, scatter plots)
- Identification du meilleur modèle par portefeuille

### `Statistics_Specific_portfolio.ipynb`
Analyse détaillée d'un portefeuille thématique spécifique :
- Exécution de tous les modèles sur un portefeuille choisi
- Visualisation des poids optimaux au fil du temps
- Comparaison des rendements prédits vs réels
- Analyse de la stabilité des poids pour chaque modèle

### `Statistiques_descriptives.ipynb`
Calcul et présentation des statistiques descriptives :
- Moyennes, médianes, quartiles des métriques
- Distribution des RMSE et de la stabilité
- Comparaisons inter-modèles

### `UNACHIVED_ML_model.ipynb`
Notebook de développement pour modèles de machine learning non finalisés (work in progress).

## 🎯 Script principal

### `evaluate_all_ptf.py`
Script Python autonome qui évalue tous les modèles sur tous les portefeuilles thématiques.

**Utilisation :**
```bash
python evaluate_all_ptf.py
```

**Fonctionnalités :**
- Charge automatiquement les données depuis `data/raw/`
- Transforme les rendements journaliers en rendements mensuels
- Évalue les 5 modèles sur chaque portefeuille thématique (33 portefeuilles)
- Calcule les métriques de performance (RMSE) et de stabilité (MAD, Turnover)
- Génère deux fichiers CSV dans `results/tables/` :
  - `evaluation_all_models_all_portfolios.csv` : Résultats détaillés par portefeuille
  - `statistiques_descriptives_rmse_stabilite.csv` : Statistiques agrégées par modèle

**Paramètres configurables :**
- `split_ratio` : Ratio train/test (défaut: 0.7)
- `window_size_linear` : Taille de fenêtre pour modèles linéaires (défaut: 12 mois)
- `window_size_advanced` : Taille de fenêtre pour modèles avancés (défaut: 18 mois)

## 📊 Résultats

### Fichiers générés

**`results/tables/evaluation_all_models_all_portfolios.csv`**
Dataframe complete avec pour chaque portefeuille et chaque modèle :
- RMSE out-of-sample
- Métriques de stabilité (MAD, Turnover, Volatilité des poids)
- Paramètres spécifiques (ex: alpha Ridge)

**`results/tables/statistiques_descriptives_rmse_stabilite.csv`**
Statistiques descriptives agrégées à partir de evaluation_all_models_all_portfolios.csv:
- Moyennes, médianes, min, max, écart-type
- Quartiles (Q1, Q3)
- Nombre d'observations valides

**`results/figures/`**
Graphiques de visualisation :
- Comparaisons de RMSE entre modèles
- Analyse du turnover par méthode
- Scatter plots RMSE vs stabilité

## 🔬 Utilisation des modules

### Exemple : Utilisation d'un modèle

```python
from src.data_processing import load_returns_data, compose_monthly_returns
from src.models import walk_forward_rebalancing_sliding_window
from src.evaluation import calculate_weight_stability

# Charger les données
thematic_df = load_returns_data('data/raw/msci_acwi_imi_thematics_daily_returns_202007_202508.csv')
sector_df = load_returns_data('data/raw/msci_acwi_imi_sectors_daily_returns_202007_202508.csv')

# Transformer en rendements mensuels
thematic_m = compose_monthly_returns(thematic_df)
sector_m = compose_monthly_returns(sector_df)

# Préparer les données
X = sector_m.drop(columns='date').values
y = thematic_m['MSCI ACWI IMI Digital Economy'].values

# Exécuter le modèle walk-forward
weights, predictions, rmse = walk_forward_rebalancing_sliding_window(
    X, y, initial_train_size=42, window_size=12, rebalance_every=1
)

# Calculer la stabilité des poids
stability = calculate_weight_stability(weights)
print(f"RMSE: {rmse:.4%}")
print(f"Turnover moyen: {stability['mean_turnover']:.4f}")
```

## 📚 Documentation technique

Le document `docs/Sujet_Technique_Stage_Quant.pdf` contient la formulation mathématique complète du problème d'optimisation et les spécifications techniques du projet.

## 🔍 Modèles implémentés

### 1. Régression Linéaire Contrainte (MSE)
Minimise l'erreur quadratique moyenne sous contraintes. Deux variantes :
- **Simple** : Entraînement unique sur période in-sample
- **Walk-Forward** : Rebalancement mensuel avec fenêtre glissante

### 2. Régression Ridge
Ajoute une pénalisation L2 pour gérer la multicolinéarité. Le paramètre alpha est optimisé via validation croisée.

### 3. Mean-Variance Tracking
Optimise la variance de l'erreur de tracking plutôt que l'erreur moyenne. Utilise l'estimateur Ledoit-Wolf pour la matrice de covariance.

### 4. Algorithme Génétique
Méthode métaheuristique stochastique explorant l'espace des solutions via sélection, croisement et mutation. Référence : Andriosopoulos & Nomikos (2014).

## 📝 Notes

- Les portefeuilles thématiques avec historiques incomplets ont été exclus (6 sur 40)
- Les données couvrent la période de juillet 2020 à août 2025
- Les rendements sont transformés en mensuels composés pour l'analyse

