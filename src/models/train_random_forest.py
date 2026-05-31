import mlflow
import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, fbeta_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_random_forest_study(X_train, y_train, X_val, y_val, X_test, y_test,
                             n_trials: int = 20, experiment_name: str = "Telco Churn Comparison"):
    """
    Optimise les hyperparamètres Random Forest avec Optuna.
    Chaque trial Optuna = 1 run MLflow.
    Le seuil est optimisé sur le validation set.
    Le test set est touché une seule fois à la fin sur le meilleur modèle.
    """
    mlflow.set_experiment(experiment_name)
    class_weight = {0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()}

    best_val_recall = -1
    best_run_id = None

    def objective(trial):
        nonlocal best_val_recall, best_run_id

        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight":      class_weight,
            "random_state":      42,
            "n_jobs":            -1,
        }
        with mlflow.start_run():
            mlflow.log_param("model_type", "random_forest")
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params({k: v for k, v in params.items()
                               if k not in ("class_weight", "random_state", "n_jobs")})

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            proba_val = model.predict_proba(X_val)[:, 1]
            y_pred_val = (proba_val >= 0.5).astype(int)

            val_recall    = recall_score(y_val, y_pred_val)
            val_precision = precision_score(y_val, y_pred_val, zero_division=0)
            val_f1        = f1_score(y_val, y_pred_val)
            val_roc_auc   = roc_auc_score(y_val, proba_val)
            val_f2        = fbeta_score(y_val, y_pred_val, beta=2)

            mlflow.log_metric("val_recall",    val_recall)
            mlflow.log_metric("val_precision", val_precision)
            mlflow.log_metric("val_f1",        val_f1)
            mlflow.log_metric("val_roc_auc",   val_roc_auc)
            mlflow.log_metric("val_f2",        val_f2)

            run_id = mlflow.active_run().info.run_id

            if val_f2 > best_val_recall:
                best_val_recall = val_f2
                best_run_id = run_id
                mlflow.log_param("is_best", True)
            else:
                mlflow.log_param("is_best", False)

        return val_f2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params

    final_model = RandomForestClassifier(
        **best,
        class_weight=class_weight,
        random_state=42, n_jobs=-1
    )
    final_model.fit(np.vstack([X_train, X_val]),
                    np.concatenate([y_train, y_val]))

    proba_test = final_model.predict_proba(X_test)[:, 1]
    y_pred_test = (proba_test >= 0.5).astype(int)

    test_metrics = {
        "test_recall":    recall_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test, zero_division=0),
        "test_f1":        f1_score(y_test, y_pred_test),
        "test_roc_auc":   roc_auc_score(y_test, proba_test),
    }

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metrics(test_metrics)
        mlflow.sklearn.log_model(final_model, artifact_path="model")

    print(f"\n✅ Random Forest — Meilleur trial #{study.best_trial.number}")
    print(f"   Val F2: {best_val_recall:.3f}")
    for k, v in test_metrics.items():
        print(f"   {k}: {v:.3f}")

    return final_model, test_metrics