import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


DATA_PATH = os.path.join("data", "processed", "cleaned_disasters.csv")
MODEL_PATH = os.path.join("models", "best_model.joblib")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.joblib")
REPORT_DIR = "reports"


def load_artifacts():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Processed dataset not found. Run src/preprocess.py first.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Best model not found. Run src/train_model.py first.")

    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError("Label encoder not found. Run src/train_model.py first.")

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    return df, model, label_encoder


def evaluate():
    os.makedirs(REPORT_DIR, exist_ok=True)

    df, model, label_encoder = load_artifacts()

    X = df.drop(columns=["severity_class"])
    y = df["severity_class"]

    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    y_pred = model.predict(X_test)

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    macro_f1 = f1_score(y_test_labels, y_pred_labels, average="macro")
    weighted_f1 = f1_score(y_test_labels, y_pred_labels, average="weighted")

    report = classification_report(y_test_labels, y_pred_labels)

    print("\nFinal Model Evaluation")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(report)

    metrics_path = os.path.join(REPORT_DIR, "final_model_metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Final Model Evaluation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1-score: {macro_f1:.4f}\n")
        f.write(f"Weighted F1-score: {weighted_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax, values_format="d")
    plt.title("Confusion Matrix - Final LightGBM Model")
    plt.tight_layout()

    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"\nSaved final metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    evaluate()