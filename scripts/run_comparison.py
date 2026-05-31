"""
Comparaison XGBoost vs Random Forest vs Neural Net
Optuna optimise les hyperparamètres sur le validation set.
Chaque trial = 1 run MLflow dans l'experiment "Telco Churn Comparison".
Le test set est touché une seule fois par modèle à la fin.

Usage:
    python scripts/run_comparison.py --input data/raw/Telco-Customer-Churn.csv
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data
from src.models.train_xgboost import run_xgboost_study
from src.models.train_random_forest import run_random_forest_study
from src.models.train_neural_net import run_neural_net_study

EXPERIMENT = "Telco Churn Comparison"


def main(args):
    # ── 1. Chargement & validation ─────────────────────────────────────────────
    print("📂 Chargement des données...")
    df = load_data(args.input)

    is_valid, failed = validate_telco_data(df)
    if not is_valid:
        raise ValueError(f"Data quality failed: {failed}")
    print("✅ Validation OK")

    # ── 2. Preprocessing & feature engineering ────────────────────────────────
    df = preprocess_data(df)
    df = build_features(df, target_col="Churn")

    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)

    # ── 3. Split train / val / test ───────────────────────────────────────────
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 60% train / 20% val / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
    )

    print(f"\n📊 Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"   Features : {X_train.shape[1]}\n")

    results = {}

    # ── 4. XGBoost ────────────────────────────────────────────────────────────
    if "xgboost" in args.models:
        print("=" * 50)
        print("🔷 XGBoost")
        print("=" * 50)
        _, metrics = run_xgboost_study(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_trials=args.n_trials, experiment_name=EXPERIMENT
        )
        results["xgboost"] = metrics

    # ── 5. Random Forest ──────────────────────────────────────────────────────
    if "random_forest" in args.models:
        print("=" * 50)
        print("🌲 Random Forest")
        print("=" * 50)
        _, metrics = run_random_forest_study(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_trials=args.n_trials, experiment_name=EXPERIMENT
        )
        results["random_forest"] = metrics

    # ── 6. Neural Net ─────────────────────────────────────────────────────────
    if "neural_net" in args.models:
        print("=" * 50)
        print("🧠 Neural Net (PyTorch)")
        print("=" * 50)
        _, metrics = run_neural_net_study(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_trials=args.n_trials, experiment_name=EXPERIMENT
        )
        results["neural_net"] = metrics

    # ── 7. Tableau récapitulatif ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 COMPARAISON FINALE (test set)")
    print("=" * 60)
    print(f"{'Modèle':<16} {'Recall':>8} {'Precision':>10} {'F1':>8} {'ROC AUC':>9}")
    print("-" * 60)
    for model_name, m in results.items():
        print(f"{model_name:<16} "
              f"{m['test_recall']:>8.3f} "
              f"{m['test_precision']:>10.3f} "
              f"{m['test_f1']:>8.3f} "
              f"{m['test_roc_auc']:>9.3f}")
    print("=" * 60)

    best_model = max(results, key=lambda k: results[k]["test_recall"])
    print(f"\n🏆 Meilleur modèle (recall) : {best_model}")
    print("\n💡 Lance 'mlflow ui' pour comparer tous les runs visuellement.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True,
                   help="Chemin vers le CSV brut")
    p.add_argument("--n_trials", type=int, default=20,
                   help="Nombre de trials Optuna par modèle")
    p.add_argument("--models", nargs="+",
                   default=["xgboost", "random_forest", "neural_net"],
                   choices=["xgboost", "random_forest", "neural_net"],
                   help="Modèles à entraîner")
    args = p.parse_args()
    main(args)
