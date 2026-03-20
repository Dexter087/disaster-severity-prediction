import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


MODEL_PATH = os.path.join("models", "best_model.joblib")
REPORT_DIR = "reports"


def get_feature_names(preprocessor):
    feature_names = []

    categorical_transformer = preprocessor.named_transformers_["categorical"]
    encoder = categorical_transformer.named_steps["encoder"]

    categorical_cols = preprocessor.transformers_[0][2]
    numerical_cols = preprocessor.transformers_[1][2]

    encoded_categorical_names = encoder.get_feature_names_out(categorical_cols)

    feature_names.extend(encoded_categorical_names)
    feature_names.extend(numerical_cols)

    return feature_names


def plot_feature_importance():
    os.makedirs(REPORT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Best model not found. Run src/train_model.py first.")

    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        raise ValueError("This model does not support feature_importances_.")

    feature_names = get_feature_names(preprocessor)
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = importance_df.sort_values(by="importance", ascending=False)

    importance_csv_path = os.path.join(REPORT_DIR, "feature_importance.csv")
    importance_df.to_csv(importance_csv_path, index=False)

    top_features = importance_df.head(20).copy()

    top_features["feature_clean"] = (
        top_features["feature"]
        .str.replace("disaster_subtype_", "", regex=False)
        .str.replace("disaster_subgroup_", "", regex=False)
        .str.replace("disaster_type_", "", regex=False)
        .str.replace("magnitude_scale_", "", regex=False)
        .str.replace("region_", "", regex=False)
        .str.replace("decade_", "", regex=False)
    )

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature_clean"], top_features["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances - Final Model")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = os.path.join(REPORT_DIR, "feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("Top 20 features:")
    print(top_features)

    print(f"\nSaved feature importance CSV to: {importance_csv_path}")
    print(f"Saved feature importance plot to: {plot_path}")


if __name__ == "__main__":
    plot_feature_importance()