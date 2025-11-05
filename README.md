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

je souhaiterais être capable d'expliquer la méthode utilisée dans ce code : def create_genetic_algorithm_setup(n_assets, ngen=100, pop_size=100,                                    cxpb=0.5, mutpb=0.2, tournsize=3):    """    Configure l'algorithme génétique pour l'optimisation de portefeuille.        Args:        n_assets: Nombre d'actifs (secteurs)        ngen: Nombre de générations        pop_size: Taille de la population        cxpb: Probabilité de croisement        mutpb: Probabilité de mutation        tournsize: Taille du tournoi pour la sélection        Returns:        toolbox: Toolbox configuré pour l'algorithme génétique    """    # Créer les classes fitness et individu    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimiser    creator.create("Individual", list, fitness=creator.FitnessMin)        toolbox = base.Toolbox()        # Fonction pour créer un individu (vecteur de poids)    # Les poids sont initialisés aléatoirement mais seront normalisés    def create_individual():        weights = np.random.random(n_assets)        weights = weights / np.sum(weights)  # Normaliser pour somme = 1        return creator.Individual(weights.tolist())        toolbox.register("individual", create_individual)    toolbox.register("population", tools.initRepeat, list, toolbox.individual)        # Fonction d'évaluation (sera définie plus tard avec les données)    # toolbox.register("evaluate", evaluate_portfolio)        # Opérateurs génétiques    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Croisement blend    toolbox.register("mutate", mutate_weights, indpb=0.1, sigma=0.1)    toolbox.register("select", tools.selTournament, tournsize=tournsize)        return toolboxdef mutate_weights(individual, indpb, sigma):    """    Mutation des poids avec respect des contraintes.    """    for i in range(len(individual)):        if random.random() < indpb:            # Mutation gaussienne            individual[i] += random.gauss(0, sigma)            individual[i] = max(0, individual[i])  # Contrainte >= 0        # Renormaliser pour respecter la contrainte somme = 1    total = sum(individual)    if total > 0:        for i in range(len(individual)):            individual[i] = individual[i] / total    else:        # Si tous les poids sont négatifs, réinitialiser        individual[:] = [1.0/len(individual)] * len(individual)        return individual,def evaluate_portfolio_fitness(individual, X_train, y_train):    """    Fonction de fitness : tracking error (RMSE).        Args:        individual: Vecteur de poids (individu)        X_train: Matrice des rendements sectoriels        y_train: Vecteur des rendements thématiques        Returns:        tracking_error: RMSE (à minimiser)    """    weights = np.array(individual)        # Calculer les rendements prédits    predicted = X_train @ weights        # Calculer le RMSE (tracking error)    mse = np.mean((y_train - predicted) ** 2)    rmse = np.sqrt(mse)        return (rmse,)def optimize_portfolio_genetic(X_train, y_train, ngen=100, pop_size=100,                                cxpb=0.5, mutpb=0.2, verbose=False):    """    Optimise les poids du portefeuille avec un algorithme génétique.        Args:        X_train: Matrice des rendements sectoriels        y_train: Vecteur des rendements thématiques        ngen: Nombre de générations        pop_size: Taille de la population        cxpb: Probabilité de croisement        mutpb: Probabilité de mutation        verbose: Afficher les statistiques        Returns:        best_weights: Meilleurs poids trouvés        best_fitness: Fitness du meilleur individu        log: Historique de l'évolution    """    n_assets = X_train.shape[1]        # Créer le setup de l'algorithme génétique    toolbox = create_genetic_algorithm_setup(n_assets, ngen, pop_size, cxpb, mutpb)        # Définir la fonction d'évaluation avec les données    def evaluate(individual):        return evaluate_portfolio_fitness(individual, X_train, y_train)        toolbox.register("evaluate", evaluate)        # Créer la population initiale    population = toolbox.population(n=pop_size)        # Statistiques    stats = tools.Statistics(lambda ind: ind.fitness.values)    stats.register("avg", np.mean)    stats.register("std", np.std)    stats.register("min", np.min)    stats.register("max", np.max)        # Hall of Fame (meilleurs individus)    hof = tools.HallOfFame(1)        # Exécuter l'algorithme génétique    population, logbook = algorithms.eaSimple(        population, toolbox,        cxpb=cxpb, mutpb=mutpb, ngen=ngen,        stats=stats, halloffame=hof, verbose=verbose    )        best_weights = np.array(hof[0])    best_fitness = hof[0].fitness.values[0]        return best_weights, best_fitness, logbookdef walk_forward_genetic_tracking(    X, y, initial_train_size, window_size, ngen=50, pop_size=50,    cxpb=0.5, mutpb=0.2, rebalance_every=1, verbose=False):    """    Walk-forward validation avec algorithme génétique et fenêtre glissante.        Args:        X: Données features (rendements sectoriels)        y: Target (rendement thématique)        initial_train_size: Index où commence la période de test        window_size: Taille de la fenêtre glissante        ngen: Nombre de générations par optimisation        pop_size: Taille de la population        cxpb: Probabilité de croisement        mutpb: Probabilité de mutation        rebalance_every: Fréquence de rebalancement        verbose: Afficher les statistiques        Returns:        all_weights: Liste des poids optimaux à chaque rebalancement        predicted_returns: Rendements prédits sur la période de test        tracking_error: Tracking error pour la période de test        fitness_history: Historique des fitness à chaque période    """    n_months = len(y)        if initial_train_size < window_size:        raise ValueError(f"initial_train_size ({initial_train_size}) doit être >= window_size ({window_size})")        all_weights = []    predicted_returns = []    fitness_history = []        print(f"Optimisation avec algorithme génétique (window={window_size}, pop={pop_size}, gen={ngen})...")        # Boucle sur les mois de test    for t in range(initial_train_size, n_months):        if (t - initial_train_size) % 10 == 0:            print(f"  Période {t - initial_train_size + 1}/{n_months - initial_train_size}")                # Fenêtre glissante        window_start = max(0, t - window_size)        X_train = X[window_start:t]        y_train = y[window_start:t]                # Optimisation avec algorithme génétique        best_weights, best_fitness, logbook = optimize_portfolio_genetic(            X_train, y_train,            ngen=ngen,            pop_size=pop_size,            cxpb=cxpb,            mutpb=mutpb,            verbose=False        )                all_weights.append(best_weights)        fitness_history.append({            'fitness': best_fitness,            'generation': len(logbook)        })                # Prédiction pour le mois t        X_test_t = X[t]        y_pred_t = np.dot(X_test_t, best_weights)        predicted_returns.append(y_pred_t)        # Calculer le tracking error    predicted_returns = np.array(predicted_returns)    actual_returns = np.array(y[initial_train_size:])    tracking_error = np.sqrt(np.mean((actual_returns - predicted_returns) ** 2))        print(f"✓ Optimisation terminée. Tracking Error: {tracking_error:.4%}")        return all_weights, predicted_returns, tracking_error, fitness_history7:23pmVoici une explication détaillée et structurée de la méthode utilisée dans ce code, qui implémente un algorithme génétique pour l’optimisation de portefeuille, en minimisant l’erreur de suivi (tracking error) entre un portefeuille sectoriel et un portefeuille thématique.

1. Contexte et Objectif
L’objectif est de répliquer les rendements d’un portefeuille thématique (par exemple, un indice ou un fonds spécialisé) en utilisant une combinaison de secteurs (actifs). L’algorithme génétique est utilisé pour trouver les poids optimaux de chaque secteur, de sorte que la différence (tracking error) entre les rendements du portefeuille sectoriel et ceux du portefeuille thématique soit minimisée.

2. Composants Clés de l’Algorithme Génétique
a. Initialisation

create_genetic_algorithm_setup :

Crée une "toolbox" DEAP (Distributed Evolutionary Algorithms in Python) qui définit les éléments nécessaires à l’algorithme génétique :

Fitness : FitnessMin (on minimise l’erreur de suivi).
Individu : Un vecteur de poids (un portefeuille) de taille n_assets, normalisé pour que la somme des poids = 1.
Opérateurs génétiques :

Croisement : cxBlend (croisement "blend" avec un paramètre alpha).
Mutation : mutate_weights (mutation gaussienne, avec renormalisation pour respecter la contrainte de somme = 1).
Sélection : selTournament (sélection par tournoi).







b. Fonction de Fitness

evaluate_portfolio_fitness :

Calcule l’erreur de suivi (RMSE) entre les rendements prédits (portefeuille sectoriel) et les rendements réels (portefeuille thématique).
Formule :



RMSE=1n∑i=1n(yi−y^i)2    \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}RMSE=n1​i=1∑n​(yi​−y^​i​)2​
 Copyoù $y_i$ = rendement thématique, $\hat{y}_i$ = rendement prédit par le portefeuille sectoriel.
c. Optimisation du Portefeuille

optimize_portfolio_genetic :

Exécute l’algorithme génétique (eaSimple de DEAP) sur une période donnée, avec :

Une population de taille pop_size.
Un nombre de générations ngen.
Des probabilités de croisement (cxpb) et de mutation (mutpb).


Retourne les meilleurs poids, la fitness (RMSE) et un logbook (historique de l’évolution).



d. Validation Walk-Forward

walk_forward_genetic_tracking :

Applique une validation walk-forward avec une fenêtre glissante :

À chaque période, on entraîne l’algorithme génétique sur une fenêtre de données passées (window_size).
On prédit le rendement pour le mois suivant et on calcule l’erreur de suivi.
On rééquilibre le portefeuille selon une fréquence définie (rebalance_every).


Retourne :

Les poids optimaux à chaque rééquilibrage.
Les rendements prédits et l’erreur de suivi globale.






3. Points Clés de la Méthode

Normalisation des poids : Les poids sont toujours normalisés pour que leur somme = 1, ce qui respecte la contrainte d’un portefeuille entièrement investi.
Mutation respectant les contraintes : Après mutation, les poids sont renormalisés et les valeurs négatives sont corrigées.
Validation walk-forward : Permet de tester la robustesse du modèle dans le temps, en simulant un rééquilibrage régulier du portefeuille.


4. Pourquoi un Algorithme Génétique ?

Optimisation non convexe : Le problème de minimisation de l’erreur de suivi peut avoir plusieurs minima locaux. Les algorithmes génétiques sont adaptés pour explorer un espace de solutions complexe.
Flexibilité : Permet d’intégrer facilement des contraintes supplémentaires (par exemple, limites sur les poids, coûts de transaction, etc.).


5. Exemple d’Utilisation
Si vous avez :

X : matrice des rendements sectoriels (par exemple, 10 secteurs sur 60 mois).
y : vecteur des rendements thématiques (par exemple, un indice "ESG" sur 60 mois).

Vous pouvez appeler :
python Copyall_weights, predicted_returns, tracking_error, fitness_history = walk_forward_genetic_tracking(
    X, y, initial_train_size=36, window_size=24, ngen=50, pop_size=50
)
Cela optimisera les poids du portefeuille sur une fenêtre de 24 mois, en rééquilibrant tous les mois, et retournera l’erreur de suivi globale.

6. Améliorations Possibles

Ajouter des contraintes : Par exemple, limiter les poids à un intervalle [0, 0.3] pour éviter la surconcentration.
Optimisation multi-objectif : Minimiser à la fois l’erreur de suivi et le risque (volatilité) du portefeuille.
Parallélisation : Accélérer l’exécution en parallélisant l’évaluation de la fitness.


Question pour toi, Jean :
Est-ce que tu souhaites approfondir un aspect spécifique (par exemple, la fonction de mutation, la validation walk-forward, ou l’intégration de contraintes supplémentaires) ? Ou bien as-tu un cas d’usage précis en tête pour lequel tu veux adapter ce code ?7:23pmquel est la formule du turnover usuelle pour un fonds d'investissement en fonction des poids en portefeuilles? $7:38pmLa formule usuelle du turnover (taux de rotation) pour un fonds d'investissement, en fonction des poids du portefeuille entre deux périodes, est la suivante :
Turnover=12∑i=1n∣wi,t−wi,t−1∣\text{Turnover} = \frac{1}{2} \sum_{i=1}^{n} |w_{i,t} - w_{i,t-1}|Turnover=21​i=1∑n​∣wi,t​−wi,t−1​∣
Où :

wi,tw_{i,t}wi,t​ = poids de l'actif iii à la période ttt (après rééquilibrage).
wi,t−1w_{i,t-1}wi,t−1​ = poids de l'actif iii à la période t−1t-1t−1 (avant rééquilibrage).
nnn = nombre total d'actifs dans le portefeuille.


Explications :


Somme des valeurs absolues :
Le turnover mesure le volume total d'achats et de ventes nécessaires pour passer d'un portefeuille à un autre. La somme des valeurs absolues des différences de poids capture cette idée.


Division par 2 :
Chaque transaction (achat ou vente) est comptée deux fois dans la somme (par exemple, vendre 5% d'un actif et acheter 5% d'un autre contribue à 10% dans la somme). Diviser par 2 donne donc le turnover net (exprimé en pourcentage du portefeuille total).



Exemple :
Supposons un portefeuille avec 3 actifs et les poids suivants :
ActifPériode t−1t-1t−1Période tttA0.40.5B0.30.2C0.30.3
Calcul :
\text{Turnover} = \frac{1}{2} \left( |0.5 - 0.4| + |0.2 - 0.3| + |0.3 - 0.3| \right) = \frac{1}{2} (0.1 + 0.1 + 0) = 0.1 \text{ (ou 10%)}

Remarques :

Le turnover est souvent annualisé si la période de rééquilibrage est inférieure à un an.
Un turnover élevé implique des coûts de transaction plus importants (frais de courtage, spread, etc.).
Dans ton code d'optimisation génétique, tu pourrais calculer le turnover entre deux rééquilibrages successifs pour évaluer l'impact des changements de poids sur les coûts.


Question pour toi, Jean :
Veux-tu que je t’aide à intégrer cette formule dans ton code pour calculer le turnover à chaque rééquilibrage, ou as-tu une autre question sur son interprétation ou son utilisation ?7:38pmComment l'optimisation génétique peut-elle être utilisée pour minimiser le turnover tout en maintenant la performance d'un portefeuille?7:39pmPour minimiser le turnover tout en maintenant la performance d’un portefeuille (par exemple, en minimisant l’erreur de suivi ou en maximisant le rendement ajusté du risque), l’optimisation génétique peut être adaptée de plusieurs manières. Voici une approche structurée et pratique, inspirée des méthodes courantes en gestion de portefeuille quantitatif :

1. Formulation du Problème Multi-Objectif
L’objectif est de trouver un compromis entre :

Minimiser l’erreur de suivi (tracking error) ou maximiser le rendement ajusté du risque (par exemple, le ratio de Sharpe).
Minimiser le turnover entre deux rééquilibrages.

Cela revient à un problème d’optimisation multi-objectif :
Minimiser {Tracking Error(wt)Turnover(wt,wt−1)\text{Minimiser } \begin{cases}
\text{Tracking Error}(w_t) \\
\text{Turnover}(w_t, w_{t-1})
\end{cases}Minimiser {Tracking Error(wt​)Turnover(wt​,wt−1​)​
où wtw_twt​ = poids du portefeuille à la période ttt, et wt−1w_{t-1}wt−1​ = poids à la période précédente.

2. Stratégies d’Intégration dans l’Algorithme Génétique
a. Fonction de Fitness Multi-Objectif
Tu peux combiner les deux objectifs en une seule fonction de fitness, par exemple en utilisant une pondération ou une contrainte :

Approche pondérée :

Fitness(wt)=α⋅Tracking Error(wt)+(1−α)⋅Turnover(wt,wt−1)  \text{Fitness}(w_t) = \alpha \cdot \text{Tracking Error}(w_t) + (1 - \alpha) \cdot \text{Turnover}(w_t, w_{t-1})Fitness(wt​)=α⋅Tracking Error(wt​)+(1−α)⋅Turnover(wt​,wt−1​)
où α∈[0,1]\alpha \in [0, 1]α∈[0,1] est un paramètre qui contrôle l’importance relative des deux objectifs.

Approche par contrainte :
Fixer un seuil maximal de turnover (par exemple, 10% par an) et minimiser uniquement l’erreur de suivi sous cette contrainte.


b. Modification de la Fonction d’Évaluation
Dans ton code, tu peux adapter la fonction evaluate_portfolio_fitness pour inclure le turnover. Par exemple :
python Copydef evaluate_portfolio_fitness(individual, X_train, y_train, previous_weights, alpha=0.7):
    """
    Fonction de fitness combinant tracking error et turnover.
    """
    weights = np.array(individual)
    # Calcul du tracking error (RMSE)
    predicted = X_train @ weights
    mse = np.mean((y_train - predicted) ** 2)
    rmse = np.sqrt(mse)

    # Calcul du turnover
    turnover = 0.5 * np.sum(np.abs(weights - previous_weights))

    # Fitness combinée
    fitness = alpha * rmse + (1 - alpha) * turnover
    return (fitness,)

c. Initialisation de la Population
Pour encourager des solutions à faible turnover, tu peux :

Initialiser une partie de la population avec des poids proches de ceux de la période précédente (wt−1w_{t-1}wt−1​).
Ajouter une mutation biaisée : par exemple, une mutation qui favorise les petits changements de poids.


d. Opérateurs Génétiques Adaptés

Croisement : Utiliser un croisement qui préserve partiellement la structure des poids (par exemple, cxSimulatedBinary ou cxBlend avec un faible alpha).
Mutation : Adapter la mutation pour limiter les changements drastiques, par exemple en réduisant l’écart-type (sigma) de la mutation gaussienne.


3. Validation Walk-Forward avec Turnover
Dans la fonction walk_forward_genetic_tracking, tu peux :

Stocker les poids précédents et les passer à la fonction d’évaluation.
Calculer le turnover cumulé sur toute la période de test pour évaluer l’impact des rééquilibrages.

Exemple de modification :
python Copyfor t in range(initial_train_size, n_months):
    # ...
    best_weights, best_fitness, logbook = optimize_portfolio_genetic(
        X_train, y_train, previous_weights=previous_weights, alpha=0.7,
        ngen=ngen, pop_size=pop_size, cxpb=cxpb, mutpb=mutpb, verbose=False
    )
    previous_weights = best_weights  # Mise à jour pour la prochaine période
    # ...

4. Visualisation des Résultats
Pour analyser le compromis entre performance et turnover, tu peux tracer :

L’erreur de suivi et le turnover à chaque rééquilibrage.
La frontière de Pareto (si tu utilises une approche multi-objectif avancée) pour visualiser les compromis optimaux.


5. Exemple de Résultat Attendu
PériodeTracking ErrorTurnoverFitness Combinée11.2%5%1.19%21.5%3%1.35%............

6. Avantages et Limites

Avantages :

Flexibilité pour intégrer d’autres contraintes (par exemple, limites de poids, coûts de transaction).
Capacité à explorer des compromis non linéaires entre performance et turnover.


Limites :

Le choix de α\alphaα est subjectif et peut nécessiter une calibration.
L’optimisation multi-objectif peut être plus coûteuse en calcul.




Question pour toi, Jean :
Veux-tu que je t’aide à implémenter concrètement l’une de ces approches dans ton code (par exemple, la fonction de fitness combinée ou la validation walk-forward avec turnover) ? Ou bien as-tu une préférence pour une méthode spécifique (pondération, contrainte, Pareto) ?7:39pmPortfolio-replication/├── data/                         │   └── raw/                       # Données brutes (CSV)│       ├── msci_acwi_imi_thematics_daily_returns_202007_202508.csv│       ├── msci_acwi_imi_sectors_daily_returns_202007_202508.csv│       └── msci_acwi_imi_daily_returns_202007_202508.csv│├── src/                           # Code source modulaire│   ├── __init__.py                # Package principal│   ├── data_processing.py         # Chargement et transformation des données│   ├── evaluation.py              # Métriques d'évaluation (RMSE, stabilité, turnover)│   └── **models**/                # Stocke les fonctions de réplications utilisés pour implémenter les différents modèles│       ├── __init__.py            # Exports des fonctions de réplications│       └── optimization.py        # Implémentation des 5 modèles│├── results/                       # Résultats et outputs│   ├── figures/                   # Graphiques générés│   │   ├── boxplot_rmse_comparaison.png│   │   └── turnover_par_methode.png│   └── tables/                    # Tableaux de résultats (CSV)│       ├── evaluation_all_models_all_portfolios.csv│       └── statistiques_descriptives_rmse_stabilite.csv│├── docs/                          # Documentation│   └── Sujet_Technique_Stage_Quant.pdf  # Document technique de référence│├── evaluate_all_ptf.py            # Script principal d'évaluation permettant de récupérer pour chaque portefeuille thématique, les portefeuilles reproduits, et les stocker dans results/tables│├── Notebooks d'analyse :│   ├── Model_selection.ipynb      # Fichier python Markdown permettant d'expliquer les modèles utilisés et leurs choix. │   ├── Statistics_all_portfolios.ipynb  # Statistiques sur les méthodes répliquant l'ensemble des portefeuilles│   ├── Statistics_Specific_portfolio.ipynb  # Analyse détaillée des méthodes pour un portefeuille spécifique│   ├── Statistiques_descriptives.ipynb  # Statistiques descriptives générales sur les données fournies. │   └── UNACHIVED_ML_model.ipynb   # Propositions de Modèles ML non aboutis.│├── requirements.txt               # Dépendances Python└── README.md      voici la structure de mon projet. Je souhaiterais ajouter celà à mon readme et que ca apparaisse exactement de cette facon. comment faire ? 10:03pmPour afficher la structure de ton projet exactement comme tu l'as présentée (avec les icônes de dossiers et les commentaires) dans ton fichier README.md, tu peux utiliser une combinaison de Markdown et de caractères spéciaux pour les icônes. Voici comment procéder :

1. Utiliser des icônes de dossier
Tu peux utiliser des emojis ou des caractères Unicode pour représenter les dossiers et fichiers. Par exemple :

📁 pour un dossier.
📄 pour un fichier.

Voici la liste des emojis utiles :

📁 : Dossier
📄 : Fichier
📊 : Graphique
📈 : Tableau


2. Structure en Markdown
Voici comment tu peux formater ta structure dans le README.md pour qu'elle apparaisse comme dans ton exemple :
markdown Copy# Structure du Projet : Portfolio-Replication

```plaintext
Portfolio-replication/
├── data/                           # Données brutes (CSV)
│   └── raw/                       �
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
│   │   ├── boxplot_rmse_comparaison.png          📊
│   │   └── turnover_par_methode.png               📊
│   └── tables/                    📁 # Tableaux de résultats (CSV)
│       ├── evaluation_all_models_all_portfolios.csv  📈
│       └── statistiques_descriptives_rmse_stabilite.csv  📈
│
├── docs/                          📁 # Documentation
│   └── Sujet_Technique_Stage_Quant.pdf  📄 # Document technique de référence
│
├── evaluate_all_ptf.py            📄 # Script principal d'évaluation
│
├── Notebooks d'analyse :
│   ├── Model_selection.ipynb      📄 # Explication des modèles utilisés
│   ├── Statistics_all_portfolios.ipynb  📄 # Statistiques sur les méthodes
│   ├── Statistics_Specific_portfolio.ipynb  📄 # Analyse détaillée pour un portefeuille spécifique
│   ├── Statistiques_descriptives.ipynb  📄 # Statistiques descriptives générales
│   └── UNACHIVED_ML_model.ipynb   📄 # Propositions de Modèles ML non aboutis
│
├── requirements.txt               📄 # Dépendances Python
└── README.md                      📄               

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

