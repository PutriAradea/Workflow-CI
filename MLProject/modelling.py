import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "stroke_preprocessed.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# parameter
max_iter = 1000
class_weight = "balanced"

# model tracking
with mlflow.start_run():

    # Train Model
    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # log parameter
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("class_weight", class_weight)

    # log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    # classification report
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    report_path = os.path.join(
        BASE_DIR,
        "classification_report.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    mlflow.log_artifact(report_path)

    # confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax
    )

    plt.tight_layout()

    cm_path = os.path.join(
        BASE_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    print("Training completed successfully!")

print("MLflow run completed.")