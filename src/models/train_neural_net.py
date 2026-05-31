import mlflow
import optuna
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, fbeta_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ChurnNet(nn.Module):
    """
    Réseau simple : entrée (n_features) → couche cachée → sortie (1 neurone sigmoid).
    Architecture volontairement légère pour données tabulaires.
    """
    def __init__(self, n_features: int, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def _to_tensor(X, y=None):
    X_t = torch.tensor(X.values if hasattr(X, "values") else X, dtype=torch.float32)
    if y is not None:
        y_t = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.float32)
        return X_t, y_t
    return X_t


def _train_model(model, X_train_t, y_train_t, lr, batch_size, epochs, pos_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(weight=None)

    # Gestion déséquilibre : on surpondère la classe positive dans la loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    # Patch : on recalcule la sortie sans sigmoid pour BCEWithLogitsLoss
    model_raw = nn.Sequential(*list(model.net.children())[:-1])  # sans sigmoid

    dataset = TensorDataset(X_train_t, y_train_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_raw.train()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model_raw(X_batch).squeeze(1)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    return model_raw


def run_neural_net_study(X_train, y_train, X_val, y_val, X_test, y_test,
                         n_trials: int = 20, experiment_name: str = "Telco Churn Comparison"):
    """
    Optimise les hyperparamètres du réseau de neurones avec Optuna.
    Chaque trial Optuna = 1 run MLflow.
    Le seuil est optimisé sur le validation set.
    Le test set est touché une seule fois à la fin sur le meilleur modèle.
    """
    mlflow.set_experiment(experiment_name)

    n_features   = X_train.shape[1]
    pos_weight   = float((y_train == 0).sum() / (y_train == 1).sum())

    X_train_t, y_train_t = _to_tensor(X_train, y_train)
    X_val_t,   y_val_t   = _to_tensor(X_val,   y_val)
    X_test_t             = _to_tensor(X_test)

    best_val_recall  = -1
    best_run_id      = None
    best_params_ever = None

    def objective(trial):
        nonlocal best_val_recall, best_run_id, best_params_ever

        hidden_size = trial.suggest_int("hidden_size", 16, 128)
        dropout     = trial.suggest_float("dropout", 0.0, 0.5)
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])
        epochs      = trial.suggest_int("epochs", 20, 100)

        with mlflow.start_run():
            mlflow.log_param("model_type",   "neural_net")
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_param("n_features",   n_features)
            mlflow.log_param("hidden_size",  hidden_size)
            mlflow.log_param("dropout",      dropout)
            mlflow.log_param("lr",           lr)
            mlflow.log_param("batch_size",   batch_size)
            mlflow.log_param("epochs",       epochs)

            model_raw = _train_model(
                ChurnNet(n_features, hidden_size, dropout),
                X_train_t, y_train_t,
                lr=lr, batch_size=batch_size, epochs=epochs,
                pos_weight=pos_weight
            )

            model_raw.eval()
            with torch.no_grad():
                logits_val = model_raw(X_val_t).squeeze(1)
                proba_val  = torch.sigmoid(logits_val).numpy()

            y_pred_val = (proba_val >= 0.5).astype(int)

            val_recall    = recall_score(y_val_t.numpy(), y_pred_val)
            val_precision = precision_score(y_val_t.numpy(), y_pred_val, zero_division=0)
            val_f1        = f1_score(y_val_t.numpy(), y_pred_val)
            val_roc_auc   = roc_auc_score(y_val_t.numpy(), proba_val)
            val_f2        = fbeta_score(y_val_t.numpy(), y_pred_val, beta=2)

            mlflow.log_metric("val_recall",    val_recall)
            mlflow.log_metric("val_precision", val_precision)
            mlflow.log_metric("val_f1",        val_f1)
            mlflow.log_metric("val_roc_auc",   val_roc_auc)
            mlflow.log_metric("val_f2",        val_f2)

            run_id = mlflow.active_run().info.run_id

            if val_f2 > best_val_recall:
                best_val_recall  = val_f2
                best_run_id      = run_id
                best_params_ever = {
                    "hidden_size": hidden_size, "dropout": dropout,
                    "lr": lr, "batch_size": batch_size, "epochs": epochs,
                }
                mlflow.log_param("is_best", True)
            else:
                mlflow.log_param("is_best", False)

        return val_f2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    bp = best_params_ever
    X_full_t = torch.cat([X_train_t, X_val_t])
    y_full_t = torch.cat([y_train_t, y_val_t])

    final_raw = _train_model(
        ChurnNet(n_features, bp["hidden_size"], bp["dropout"]),
        X_full_t, y_full_t,
        lr=bp["lr"], batch_size=bp["batch_size"], epochs=bp["epochs"],
        pos_weight=pos_weight
    )

    final_raw.eval()
    with torch.no_grad():
        logits_test = final_raw(X_test_t).squeeze(1)
        proba_test  = torch.sigmoid(logits_test).numpy()

    y_pred_test = (proba_test >= 0.5).astype(int)

    test_metrics = {
        "test_recall":    recall_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test, zero_division=0),
        "test_f1":        f1_score(y_test, y_pred_test),
        "test_roc_auc":   roc_auc_score(y_test, proba_test),
    }

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metrics(test_metrics)
        mlflow.pytorch.log_model(final_raw, artifact_path="model")

    print(f"\n✅ Neural Net — Meilleur trial #{study.best_trial.number}")
    print(f"   Architecture : {n_features} → {bp['hidden_size']} → 1")
    print(f"   Val F2       : {best_val_recall:.3f}")
    for k, v in test_metrics.items():
        print(f"   {k}: {v:.3f}")

    return final_raw, test_metrics
