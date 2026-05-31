# Prédiction de Résiliation Client – Pipeline ML end-to-end sur Google Cloud Run

Projet de Data Science appliqué au secteur télécom : prédiction du churn client par comparaison de 3 modèles (XGBoost, Random Forest, Neural Net PyTorch), optimisation des hyperparamètres via Optuna, tracking MLflow, et déploiement complet sur Google Cloud Run via CI/CD GitHub Actions.

---

## Contexte

**Point de départ** : le dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), un dataset de référence dans le domaine télécom avec 7 043 clients et 26 variables brutes (profil, services souscrits, contrat, facturation).

**Problème métier** : un client qui résilie coûte plus cher à remplacer qu'à retenir. Manquer un churner (faux négatif) est plus coûteux que contacter un client fidèle (faux positif). Cette contrainte oriente directement les choix de modélisation : **le recall est la métrique prioritaire**.

**Objectif** : construire un pipeline ML complet, de la validation des données jusqu'au modèle en production, accessible via une API REST et une interface web.

---

## Compétences démontrées

| Compétence | Ce qui est fait dans ce projet |
|---|---|
| **Machine Learning** | Comparaison de 3 modèles (XGBoost, Random Forest, Neural Net PyTorch), gestion du déséquilibre de classes, optimisation F2-score via Optuna (20 trials/modèle) |
| **MLOps** | 60 runs MLflow trackés (params, métriques, artefacts), Model Registry, 3-level model loading en production |
| **Feature Engineering** | BINARY_MAP déterministe, get_dummies drop_first, reindex(FEATURE_COLS) — cohérence train/serve garantie |
| **Déploiement Cloud** | Conteneurisation Docker, CI/CD GitHub Actions → Artifact Registry → Cloud Run (GCP) |
| **API & Interface** | API REST FastAPI + Swagger auto-généré, interface web Streamlit Community Cloud |

---

## Architecture

```
CSV brut → Validation (23 checks) → Preprocessing → Feature Engineering
→ Split 60/20/20 → Optuna (20 trials × 3 modèles) → MLflow Tracking
→ Meilleur run → Model Registry → inference.py → FastAPI → Cloud Run
                                                          ↑
                                              Streamlit Community Cloud
```

---

## Machine Learning : détail des choix

### Déséquilibre de classes

Le dataset contient **26.5% de churners**. Chaque modèle gère le déséquilibre différemment :

- `scale_pos_weight = n_non_churners / n_churners` pour XGBoost
- `class_weight = {0: 1, 1: ratio}` pour Random Forest
- `BCEWithLogitsLoss(pos_weight=ratio)` pour le Neural Net PyTorch

### Comparaison des 3 modèles (test set — seuil 0.5)

| Modèle | Recall | Precision | F1 | ROC AUC | Val F2 |
|---|---|---|---|---|---|
| **XGBoost** ✅ | **0.807** | 0.503 | 0.620 | **0.848** | 0.737 |
| Random Forest | 0.805 | 0.516 | 0.629 | 0.843 | 0.734 |
| Neural Net (PyTorch) | 0.561 | 0.597 | 0.579 | 0.815 | 0.735 |

XGBoost retenu : meilleur recall et ROC AUC, artefacts MLflow complets pour chaque trial.

### Optuna + MLflow — comment ils sont liés

**Chaque trial Optuna = 1 run MLflow** :

1. Optuna suggère des hyperparamètres (algorithme TPE)
2. Le modèle est entraîné sur le train set
3. Évaluation sur le validation set → F2-score (β=2, favorise le recall)
4. MLflow logge params + métriques + artefact dans le même run
5. Le meilleur run est marqué `is_best=True`

20 trials × 3 modèles = **60 runs trackés** dans l'experiment `Telco Churn Comparison`.

Pourquoi F2 et pas recall seul ? Le recall seul pousse le modèle à tout prédire "churn" (precision → 0). F2 maintient un équilibre utile tout en favorisant le recall.

### Seuil de décision

Le seuil est **fixé à 0.5 pendant la recherche Optuna** (inclure le seuil dans Optuna le poussait toujours vers 0.25 — recall=0.97, precision=0.33). Le seuil est optimisé séparément sur le validation set après sélection du meilleur modèle.

| Seuil | Recall | Precision | F1 | F2 |
|---|---|---|---|---|
| 0.30 | 0.88 | 0.44 | 0.58 | 0.75 |
| 0.40 | 0.82 | 0.49 | 0.61 | 0.74 |
| **0.50** | **0.77** | **0.52** | **0.62** | **0.74** |
| 0.60 | 0.70 | 0.57 | 0.63 | 0.68 |

---

## MLflow : local vs production

**En local** :
- Backend : `./mlruns/` (file store)
- UI : `mlflow ui` → http://127.0.0.1:5000
- Meilleur run → enregistré dans le Model Registry → promu en "Production"

**En production (Cloud Run)** :
- `inference.py` charge le modèle en 3 niveaux :
  1. MLflow Model Registry : `models:/churn-model/Production`
  2. Chemin Docker : `/app/model` (modèle copié à la construction de l'image)
  3. Fallback local : `./mlruns/*/artifacts/model`

Changer de modèle en production = promouvoir un nouveau run dans le Registry + rebuild Docker.

---

## Stack technique

| Couche | Technologie |
|---|---|
| Machine Learning | XGBoost, scikit-learn, PyTorch |
| Hyperparameter tuning | Optuna (20 trials/modèle, algorithme TPE) |
| Tracking | MLflow (params, métriques, artefacts, Model Registry) |
| Validation données | 23 checks custom (schéma, business rules, ranges numériques) |
| API | FastAPI, Pydantic, Swagger |
| Interface | Streamlit Community Cloud |
| Conteneurisation | Docker (python:3.11-slim) |
| CI/CD | GitHub Actions → Artifact Registry → Cloud Run |
| Cloud | Google Cloud Run (scale-to-zero, 1Gi RAM) |

---

## Feature Engineering

- **26 colonnes brutes** → **20 colonnes** après preprocessing (suppression customerID, conversion types)
- **20 colonnes** → **30 features** après feature engineering :
  - 5 binaires (Yes/No, Male/Female) → encodage déterministe via `BINARY_MAP`
  - 10 multi-catégories → `pd.get_dummies(drop_first=True)` → 21 nouvelles colonnes
  - bool → int (compatibilité XGBoost/sklearn)
- `FEATURE_COLS` sauvegardé à l'entraînement → `reindex(FEATURE_COLS, fill_value=0)` en production pour garantir l'ordre et gérer les catégories absentes

---

## Dataset

- **Source** : [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- **Volume** : 7 043 clients, 26 variables brutes
- **Cible** : variable `Churn` (Yes/No) — 26.5% de churners
- **Split** : 60% train / 20% val / 20% test (stratifié, random_state=42)

---

## Lancement local

```bash
# 1. Cloner le repo
git clone https://github.com/Nesho-k/churn.git
cd churn

# 2. Créer l'environnement virtuel
python -m venv churn
churn\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements_local.txt

# 4. Lancer la comparaison des 3 modèles (60 runs MLflow)
python scripts/run_comparison.py --input data/raw/Telco-Customer-Churn.csv

# 5. Visualiser les runs MLflow
python -m mlflow ui

# 6. Afficher les classification reports des meilleurs modèles
python scripts/print_classification_reports.py --input data/raw/Telco-Customer-Churn.csv

# 7. Lancer l'API FastAPI
uvicorn src.app.main:app --host 0.0.0.0 --port 8080

# 8. Lancer l'interface Streamlit
streamlit run src/app/streamlit_app.py
```

---

## Structure du projet

```
src/
├── app/          # API FastAPI + interface Streamlit
├── data/         # Chargement et preprocessing
├── features/     # Feature engineering
├── models/       # Entraînement XGBoost, Random Forest, Neural Net
├── serving/      # inference.py — chargement modèle + prédiction
└── utils/        # Validation des données (23 checks)
scripts/
├── run_comparison.py           # Orchestrateur : 3 modèles × 20 trials
├── print_classification_reports.py  # Classification report des meilleurs runs
└── feature_importance.py       # Feature importance XGBoost
data/raw/                       # CSV brut Telco
.github/workflows/ci.yml        # CI/CD GitHub Actions → Cloud Run
dockerfile                      # Image Docker python:3.11-slim
requirements_local.txt          # Dépendances complètes (local)
requirements_api.txt            # Dépendances API (Docker)
requirements.txt                # Dépendances Streamlit Community Cloud
```

---

## Auteur

**Nesho Kanthakumar**
Étudiant en Data Science
[GitHub](https://github.com/Nesho-k) · [LinkedIn](https://www.linkedin.com/in/nesho-kanthakumar-6354512a6/)
