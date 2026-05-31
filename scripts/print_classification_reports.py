"""
Charge le meilleur run MLflow pour chaque modèle et affiche le classification report.
Usage:
    python scripts/print_classification_reports.py --input data/raw/Telco-Customer-Churn.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data

EXPERIMENT = "Telco Churn Comparison"
MODEL_TYPES = ["xgboost", "random_forest", "neural_net"]


def get_best_run(client, experiment_id, model_type):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"params.model_type = '{model_type}' AND params.is_best = 'True'",
        order_by=["metrics.val_f2 DESC"],
        max_results=1,
    )
    if not runs:
        print(f"  ⚠️  Aucun run trouvé pour {model_type}")
        return None
    return runs[0]


def main(args):
    # ── Données ───────────────────────────────────────────────────────────────
    df = load_data(args.input)
    is_valid, failed = validate_telco_data(df)
    if not is_valid:
        raise ValueError(f"Data quality failed: {failed}")

    df = preprocess_data(df)
    df = build_features(df, target_col="Churn")
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT}' introuvable. Lance d'abord run_comparison.py.")

    for model_type in MODEL_TYPES:
        print("\n" + "=" * 60)
        print(f"  {model_type.upper()}")
        print("=" * 60)

        run = get_best_run(client, experiment.experiment_id, model_type)
        if run is None:
            continue

        run_id = run.info.run_id
        print(f"  Run ID : {run_id}")
        print(f"  Val F2 : {run.data.metrics.get('val_f2', 'N/A'):.3f}")

        model_uri = f"runs:/{run_id}/model"
        try:
            loaded = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            print(f"  ❌ Impossible de charger le modèle : {e}")
            continue

        preds = loaded.predict(X_test)
        if isinstance(preds, pd.DataFrame):
            preds = preds.iloc[:, 0].values
        elif hasattr(preds, "tolist"):
            preds = np.array(preds.tolist())

        preds = np.array(preds, dtype=float)
        y_pred = (preds >= 0.5).astype(int)

        print()
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    args = p.parse_args()
    main(args)