import sys
sys.path.append(".")

import pandas as pd
import mlflow
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

model_uri = "runs:/c717ab6c985c4285b4348ee99c6123c8/model"
model = mlflow.xgboost.load_model(model_uri)

df = load_data("data/raw/Telco-Customer-Churn.csv")
df = preprocess_data(df)
df = build_features(df, target_col="Churn")
for c in df.select_dtypes(include=["bool"]).columns:
    df[c] = df[c].astype(int)
feature_names = df.drop(columns=["Churn"]).columns.tolist()

fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
print(fi.head(15))
