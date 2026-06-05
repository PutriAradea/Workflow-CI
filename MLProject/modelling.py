import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

# load data
BASE_DIR = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE_DIR, "stroke_preprocessed.csv"))

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# parameter
max_iter = 1000
class_weight = "balanced"

# mlflow training
with mlflow.start_run():

    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log params
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("class_weight", class_weight)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Classification report
    report = classification_report(y_test, y_pred)
    report_path = os.path.join(BASE_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    mlflow.log_artifact(report_path)

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    print("TRAINING DONE OK")