import os
import joblib
import pandas as pd
import warnings


MODEL_PATH = os.path.join("models", "best_model.joblib")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.joblib")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_disasters.csv")


warnings.filterwarnings("ignore", message="X does not have valid feature names")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Best model not found. Run src/train_model.py first.")

    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError("Label encoder not found. Run src/train_model.py first.")

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    return model, label_encoder


def load_reference_data():
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed dataset not found. Run src/preprocess.py first.")

    return pd.read_csv(PROCESSED_DATA_PATH)


def calculate_decade(year):
    year = int(year)
    return f"{year // 10 * 10}s"


def calculate_frequency_features(event_data, reference_df):
    country = event_data.get("country", "Unknown")
    disaster_type = event_data.get("disaster_type", "Unknown")

    country_frequency = reference_df["country"].value_counts().get(country, 0)
    disaster_type_frequency = reference_df["disaster_type"].value_counts().get(disaster_type, 0)

    event_data["country_disaster_frequency"] = country_frequency
    event_data["disaster_type_frequency"] = disaster_type_frequency

    return event_data


def prepare_event(event_data):
    reference_df = load_reference_data()

    if "decade" not in event_data:
        event_data["decade"] = calculate_decade(event_data["start_year"])

    if "disaster_duration_years" not in event_data:
        event_data["disaster_duration_years"] = max(
            int(event_data["end_year"]) - int(event_data["start_year"]),
            0
        )

    if "magnitude" not in event_data or pd.isna(event_data["magnitude"]):
        event_data["magnitude"] = 0
        event_data["magnitude_missing"] = 1
    else:
        event_data["magnitude_missing"] = 0

    event_data = calculate_frequency_features(event_data, reference_df)

    expected_columns = [
        "country",
        "region",
        "disaster_group",
        "disaster_subgroup",
        "disaster_type",
        "disaster_subtype",
        "start_year",
        "start_month",
        "end_year",
        "magnitude",
        "magnitude_scale",
        "decade",
        "disaster_duration_years",
        "country_disaster_frequency",
        "disaster_type_frequency",
        "magnitude_missing",
    ]

    final_event = {}

    for col in expected_columns:
        final_event[col] = event_data.get(col, "Unknown")

    return pd.DataFrame([final_event])


def predict_single_event(event_data):
    model, label_encoder = load_model()

    input_df = prepare_event(event_data)

    prediction_encoded = model.predict(input_df)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)

    prediction_proba = model.predict_proba(input_df)[0]

    probability_df = pd.DataFrame(
        {
            "severity_class": label_encoder.classes_,
            "probability": prediction_proba,
        }
    ).sort_values(by="probability", ascending=False)

    return prediction_label[0], probability_df


if __name__ == "__main__":
    sample_event = {
        "country": "India",
        "region": "Asia",
        "disaster_group": "Natural",
        "disaster_subgroup": "Hydrological",
        "disaster_type": "Flood",
        "disaster_subtype": "Flood (General)",
        "start_year": 2024,
        "start_month": 7,
        "end_year": 2024,
        "magnitude": 1000,
        "magnitude_scale": "Km2",
    }

    prediction, probabilities = predict_single_event(sample_event)

    print("\nSample Disaster Event Prediction")
    print("=" * 60)
    print(f"Predicted severity: {prediction}")

    print("\nClass probabilities:")
    print(probabilities)