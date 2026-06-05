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

# Load dataset
df = pd.read_csv(
    "stroke_preprocessed.csv"
)

# Split fitur dan target
X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Nama eksperimen
mlflow.set_experiment("Stroke_Prediction")

with mlflow.start_run():

    # Parameter model
    max_iter = 1000
    class_weight = "balanced"

    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight
    )

    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Manual logging parameter
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("class_weight", class_weight)

    # Manual logging metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Simpan model
    mlflow.sklearn.log_model(model, "model")

    # Artefak 1: Classification Report
    report = classification_report(y_test, y_pred)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    # Artefak 2: Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")