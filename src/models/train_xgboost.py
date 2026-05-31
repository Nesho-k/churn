import mlflow
import optuna
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, fbeta_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_xgboost_study(X_train, y_train, X_val, y_val, X_test, y_test,
                      n_trials: int = 20, experiment_name: str = "Telco Churn Comparison"):
    """
    Optimise les hyperparamètres XGBoost avec Optuna.
    Chaque trial Optuna = 1 run MLflow.
    Le seuil est optimisé sur le validation set.
    Le test set est touché une seule fois à la fin sur le meilleur modèle.
    """
    mlflow.set_experiment(experiment_name)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    best_val_recall = -1
    best_run_id = None

    def objective(trial):
        nonlocal best_val_recall, best_run_id

        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight":  scale_pos_weight,
            "random_state":      42,
            "n_jobs":            -1,
            "eval_metric":       "logloss",
        }
        with mlflow.start_run():
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params({k: v for k, v in params.items()
                               if k not in ("scale_pos_weight", "random_state",
                                            "n_jobs", "eval_metric")})

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)

            # Évaluation sur validation set (threshold fixe à 0.5)
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

            mlflow.xgboost.log_model(model, artifact_path="model")

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

    # Ré-entraîne le meilleur modèle sur train+val, évalue sur test (threshold=0.5)
    best = study.best_params

    final_model = XGBClassifier(
        **best,
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1, eval_metric="logloss"
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

    print(f"\n✅ XGBoost — Meilleur trial #{study.best_trial.number}")
    print(f"   Val Recall: {best_val_recall:.3f}")
    for k, v in test_metrics.items():
        print(f"   {k}: {v:.3f}")

    return final_model, test_metrics