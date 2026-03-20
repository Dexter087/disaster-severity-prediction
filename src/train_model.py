import os
import joblib
import pandas as pd

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


DATA_PATH = os.path.join("data", "processed", "cleaned_disasters.csv")
MODEL_DIR = "models"
REPORT_DIR = "reports"


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Processed dataset not found at {DATA_PATH}. "
            "Please run src/preprocess.py first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded processed dataset: {df.shape}")
    return df


def split_features_target(df):
    target_col = "severity_class"

    if target_col not in df.columns:
        raise ValueError("severity_class column not found in processed dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("\nCategorical columns:")
    print(categorical_cols)

    print("\nNumerical columns:")
    print(numerical_cols)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_cols),
            ("numerical", numerical_pipeline, numerical_cols),
        ]
    )

    return preprocessor


def get_models():
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            random_state=42,
            n_jobs=-1
        ),
    }

    return models


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print(f"Model: {name}")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = load_data()
    X, y = split_features_target(df)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(f"\nTraining size: {X_train.shape}")
    print(f"Testing size: {X_test.shape}")

    preprocessor = build_preprocessor(X_train)
    models = get_models()

    results = []
    best_model = None
    best_model_name = None
    best_macro_f1 = -1

    for name, clf in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )

        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        y_test_labels = label_encoder.inverse_transform(y_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)

        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        macro_f1 = f1_score(y_test_labels, y_pred_labels, average="macro")
        weighted_f1 = f1_score(y_test_labels, y_pred_labels, average="weighted")

        print("\n" + "=" * 60)
        print(f"Model: {name}")
        print("=" * 60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")
        print(f"Weighted F1-score: {weighted_f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_labels))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test_labels, y_pred_labels))

        result = {
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

        results.append(result)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model = pipeline
            best_model_name = name

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="macro_f1", ascending=False)

    results_path = os.path.join(REPORT_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)

    best_model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")

    joblib.dump(best_model, best_model_path)
    joblib.dump(label_encoder, label_encoder_path)

    print("\n" + "=" * 60)
    print("Model comparison:")
    print(results_df)
    print("=" * 60)

    print(f"\nBest model: {best_model_name}")
    print(f"Best macro F1-score: {best_macro_f1:.4f}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved label encoder to: {label_encoder_path}")
    print(f"Saved model comparison report to: {results_path}")


if __name__ == "__main__":
    train_models()