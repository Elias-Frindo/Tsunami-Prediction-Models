# Prédiction des tsunamis — Random Forest vs XGBoost

Projet d'apprentissage automatique visant à **prédire si un séisme a déclenché un tsunami** (`tsunami` = 1) ou non (`tsunami` = 0) à partir de ses caractéristiques (magnitude, intensités, profondeur, localisation, qualité de la mesure, date). Il s'agit d'un problème de **classification binaire** sur 782 séismes de magnitude ≥ 6.5 (2001–2022). Deux modèles d'ensemble d'arbres de décision, **Random Forest** et **XGBoost**, sont entraînés et comparés.

## Structure du projet

```
.
├── earthquake_data_tsunami.csv      # Données (782 lignes, 13 colonnes)
├── Prédiction_tsunami.ipynb            # Analyse complète : exploration, entraînement, comparaison
├── modele_final_tsunami.joblib      # Modèle sauvegardé (généré à l'exécution du notebook)
└── README.md
```

## Données

Fichier : `earthquake_data_tsunami.csv` — aucune valeur manquante, aucun doublon.

| Variable | Description |
|---|---|
| `magnitude` | Magnitude du séisme (de 6.5 à 9.1) |
| `cdi` | Intensité ressentie, déclarée par les habitants |
| `mmi` | Intensité instrumentale (échelle de Mercalli modifiée) |
| `sig` | Score de significativité calculé par l'USGS |
| `nst` | Nombre de stations sismiques ayant enregistré l'événement |
| `dmin` | Distance à la station la plus proche |
| `gap` | Écart azimutal entre stations |
| `depth` | Profondeur du foyer (km) |
| `latitude`, `longitude` | Épicentre |
| `Year`, `Month` | Date du séisme |
| **`tsunami`** | **Variable cible** : 1 si un tsunami a été déclenché, 0 sinon (≈ 39 % de 1, ≈ 61 % de 0) |

## Méthodologie

Le notebook `Serie_2_tsunami.ipynb` suit une démarche en 13 sections :

1. Importation des bibliothèques et des données
2. Nettoyage et analyse exploratoire : équilibre des classes, statistiques, valeurs aberrantes (règle de l'IQR), corrélations, distribution des variables selon la classe, carte des séismes, proportion de tsunamis par année
3. Séparation X / y (pas de standardisation : les modèles à base d'arbres n'en ont pas besoin)
4. Séparation entraînement / test (80 % / 20 %, stratifiée)
5. Modèle de référence (baseline : toujours la classe majoritaire)
6. **Random Forest** : hyperparamètres par défaut, puis optimisés (`RandomizedSearchCV`, validation croisée 5-fold, score = AUC ROC)
7. **XGBoost** : hyperparamètres par défaut, puis optimisés (mêmes modalités)
8. Comparaison : exactitude, précision, rappel, F1, AUC ROC, AUC PR, matrices de confusion, courbes ROC et précision-rappel, importance des variables (impureté/gain et permutation), validation croisée stratifiée 10-fold
9. Seuil de décision : compromis entre faux négatifs et fausses alertes
10. Robustesse : le modèle apprend-il la géophysique ou l'historique des mesures ? (jeux de variables réduits et validation temporelle)
11. Interprétation
12. Prédiction pour un nouveau séisme
13. Sauvegarde du modèle final avec `joblib`

## Résultats

Validation croisée stratifiée 10-fold sur l'ensemble des données :

| Modèle | AUC ROC | Exactitude | F1 |
|---|---|---|---|
| Baseline (classe majoritaire) | 0.500 | 0.611 | 0.000 |
| Random Forest (optimisé) | 0.971 | 0.936 | 0.923 |
| XGBoost (optimisé) | 0.971 | 0.926 | 0.908 |

* **Random Forest et XGBoost sont à égalité :** l'écart d'AUC (0.9712 contre 0.9707) est très inférieur à la variation entre plis (≈ 0.017). Random Forest est très légèrement meilleur en exactitude et en F1 ; c'est lui que le notebook retient automatiquement.
* **Seuil de décision :** pour un système d'alerte, manquer un vrai tsunami est plus grave qu'une fausse alerte. Avec Random Forest sur le jeu de test, un seuil de 0.3 supprime tous les faux négatifs au prix de 12 fausses alertes (contre 2 faux négatifs et 7 fausses alertes au seuil de 0.5).

### Mise en garde : le rôle de la variable `Year`

La variable la plus importante n'est ni la magnitude ni la profondeur, mais **`Year`** : la retirer fait chuter l'AUC d'environ 0.23. La proportion de tsunamis passe de 39 % globalement à 75 % après 2018, ce qui reflète vraisemblablement l'évolution de la collecte des données plutôt qu'un phénomène géophysique. Les scores ci-dessus surestiment donc la capacité réelle du modèle :

| Jeu de variables (AUC ROC, CV 10-fold) | Random Forest | XGBoost |
|---|---|---|
| Toutes les variables | 0.971 | 0.971 |
| Sans la date (`Year`, `Month`) | 0.943 | 0.944 |
| Géophysiques seulement (`magnitude`, `depth`, `latitude`, `longitude`, `cdi`, `mmi`, `sig`) | 0.819 | 0.810 |

En validation temporelle (entraînement sur les années ≤ 2018, test sur les années récentes), l'AUC est d'environ 0.85–0.87 avec toutes les variables et d'environ 0.81–0.83 avec les seules variables géophysiques. La capacité réelle à prédire un tsunami est donc plutôt de l'ordre de **0.8 à 0.85 d'AUC**.

## Limites

* Petit échantillon (782 séismes) et magnitudes ≥ 6.5 uniquement.
* Biais d'échantillonnage manifeste lié à `Year`.
* Les variables `cdi`, `mmi` et `sig` sont mesurées **après** l'événement : le modèle classe a posteriori, il ne constitue pas un système de prévision.
* Un seul jeu de test ; les hyperparamètres sont choisis sur le jeu d'entraînement, alors que la validation croisée 10-fold utilise toutes les données (légère optimisme).
* Pour un usage réel, il faudrait des données plus riches (type de faille, distance à la côte, bathymétrie, mécanisme au foyer).

## Installation et utilisation

Prérequis : Python 3.10 ou supérieur.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib jupyter
```

Le notebook a été exécuté avec `scikit-learn` 1.8 et `xgboost` 3.4.

Placez `earthquake_data_tsunami.csv` dans le même dossier que le notebook, puis :

```bash
jupyter notebook
```

Ouvrez `Serie_2_tsunami.ipynb` et exécutez toutes les cellules (`Run All`). Les graines aléatoires sont fixées (`RANDOM_STATE = 42`) : les résultats sont reproductibles.

### Réutiliser le modèle sauvegardé

```python
import joblib
import pandas as pd

saved = joblib.load("modele_final_tsunami.joblib")
model, features = saved["model"], saved["features"]

# magnitude, cdi, mmi, sig, nst, dmin, gap, depth, latitude, longitude, Year, Month
new_quake = pd.DataFrame([[7.2, 6, 6, 800, 150, 1.5, 20, 30, -20.0, -70.0, 2022, 11]],
                         columns=features)
proba = model.predict_proba(new_quake)[0, 1]
print(f"Probabilité de tsunami : {proba:.1%}")
```
