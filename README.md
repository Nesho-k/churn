# Prédiction de Résiliation Client – Pipeline ML déployé sur AWS

Projet de Data Science appliqué au secteur télécom : prédiction du churn client par XGBoost, avec comparaison de modèles, optimisation du seuil de décision, tuning via Optuna, tracking MLflow, et déploiement complet sur AWS ECS Fargate.

---

## Contexte

**Point de départ** : le dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), un dataset de référence dans le domaine télécom avec 7 043 clients réels et 21 variables (profil, services souscrits, contrat, facturation).

**Problème métier** : un client qui résilie coûte plus cher à remplacer qu'à retenir. Dans ce contexte, manquer un churner (faux négatif) est plus coûteux que contacter un client fidèle (faux positif). Cette contrainte oriente directement les choix de modélisation : **le recall est la métrique prioritaire**.

**Objectif** : construire un pipeline ML complet, de l'analyse exploratoire jusqu'au modèle en production, accessible via une API REST et une interface web.

---

## Compétences démontrées

| Compétence | Ce qui est fait dans ce projet |
|---|---|
| **Analyse exploratoire** | Identification des variables corrélées au churn, analyse du déséquilibre de classes (27% de churners), corrélations |
| **Machine Learning** | Comparaison de 3 modèles (RandomForest, LightGBM, XGBoost), gestion du déséquilibre (`scale_pos_weight`), tuning du seuil de décision, optimisation via Optuna |
| **MLOps** | Tracking MLflow (paramètres, métriques, artefacts), pipeline reproductible entraînement → production |
| **Déploiement Cloud** | Conteneurisation Docker, CI/CD GitHub Actions → Docker Hub, déploiement AWS ECS Fargate + ALB |
| **API & Interface** | API REST FastAPI, interface web Streamlit |

---

## Architecture

Le pipeline part du CSV brut Telco et passe d'abord par une étape de preprocessing : validation des données (Great Expectations), encodage des variables catégorielles et alignement des colonnes entre entraînement et production.

Le modèle est ensuite entraîné (XGBoost, sélectionné après comparaison avec RandomForest et LightGBM), avec gestion du déséquilibre de classes, tuning du seuil de décision à 0.35, et optimisation des hyperparamètres via Optuna sur 30 essais. Chaque run est tracké dans MLflow.

Une fois le modèle sérialisé, il est servi de deux façons : une API REST via FastAPI et une interface web via Streamlit. L'ensemble est conteneurisé avec Docker et déployé sur AWS ECS Fargate derrière un ALB.

---

## Machine Learning : détail des choix

C'est le cœur du projet. Chaque décision est motivée par la contrainte métier.

### Déséquilibre de classes

Le dataset contient **27% de churners**. Sans traitement, un modèle naïf prédirait "Non-churn" pour tout le monde et obtiendrait 73% de précision globale, sans jamais détecter un seul churner. Deux mécanismes sont utilisés :

- `class_weight='balanced'` pour RandomForest
- `scale_pos_weight = n_non_churners / n_churners` pour XGBoost

### Comparaison des modèles

Trois modèles ont été comparés à seuil de décision identique (0.35) :

| Modèle | Recall (churners) |
|---|---|
| RandomForest | 71,7 % |
| Réseaux de neurones | 69.4 % |
| **XGBoost** | **91,7 %** |

XGBoost est sélectionné pour sa capacité à gérer le déséquilibre de classes via `scale_pos_weight` et ses meilleures performances en recall.

### Tuning du seuil de décision

Le seuil par défaut (0.5) est sous-optimal dans ce contexte. Un seuil plus bas (0.35) permet de détecter plus de churners au prix d'un léger surcoût de faux positifs — compromis acceptable selon la contrainte métier.

### Optimisation des hyperparamètres (Optuna)

30 essais Optuna sur les hyperparamètres clés (`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`).

**Meilleur résultat** : Recall = **91,71 %** (trial 25)

### Insights EDA

- Le type de contrat est le principal facteur de churn : les clients avec contrat mensuel sont beaucoup plus susceptibles de résilier (`Contract_Two year` : corrélation -0.30 avec le churn)
- L'ancienneté est fortement négativement corrélée au churn (`tenure` : -0.35) — plus un client reste longtemps, moins il est susceptible de partir

---

## MLOps & Tracking

Chaque entraînement est tracké via **MLflow** :
- Paramètres : hyperparamètres du modèle, seuil, taille du jeu de test
- Métriques : recall, precision, F1, ROC-AUC, temps d'entraînement
- Artefacts : modèle sérialisé, `feature_columns.txt`, `preprocessing.pkl`

La cohérence entre entraînement et production est garantie : les transformations appliquées aux données (encodage, ordre des colonnes) sont strictement identiques via les artefacts MLflow chargés au moment de la prédiction.

---

## Stack technique

| Couche | Technologie |
|---|---|
| Machine Learning | XGBoost, scikit-learn, Optuna |
| Tracking | MLflow |
| Validation données | Great Expectations |
| API | FastAPI, Pydantic |
| Interface | Streamlit |
| Conteneurisation | Docker |
| CI/CD | GitHub Actions → Docker Hub |
| Cloud | AWS ECS Fargate, ALB |

---

## Dataset

- **Source** : [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- **Volume** : 7 043 clients, 21 variables
- **Cible** : variable `Churn` (Yes/No) — 26,5% de churners

---

## Application en ligne

Interface Streamlit déployée sur AWS ECS Fargate :
**http://telco-alb-101861819.eu-north-1.elb.amazonaws.com/**

---

## Installation locale

### Prérequis
- Python 3.11+
- Docker Desktop

### Lancement

```bash
# 1. Cloner le repo
git clone https://github.com/Nesho-k/churn.git
cd churn

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le pipeline d'entraînement
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn

# 5. Lancer l'interface Streamlit
streamlit run src/app/streamlit_app.py

# 6. Ou lancer l'API FastAPI
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

---

## Structure du projet

Le code source est dans `src/`, organisé en trois parties : `app/` pour l'API FastAPI et l'interface Streamlit, `features/` pour le feature engineering, `serving/` pour le chargement du modèle et la prédiction.

Les scripts d'entraînement sont dans `scripts/`, avec `run_pipeline.py` comme point d'entrée principal. Les données brutes et traitées sont dans `data/`, et les expériences MLflow dans `mlruns/` (non versionné). Le CI/CD est géré par `.github/workflows/ci.yml`.

---

## Auteur

**Nesho Kanthakumar**
Étudiant en Data Science
[GitHub](https://github.com/Nesho-k) · [LinkedIn](https://www.linkedin.com/in/nesho-kanthakumar-6354512a6/)
