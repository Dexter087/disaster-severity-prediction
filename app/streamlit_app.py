import os
import sys
import pandas as pd
import streamlit as st

# Allows Streamlit to import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_single_event


st.set_page_config(
    page_title="Disaster Severity Prediction",
    page_icon="🌍",
    layout="centered"
)


st.title("🌍 Disaster Severity Prediction")
st.write(
    "This app predicts whether a disaster event is likely to have "
    "**Low Impact**, **Medium Impact**, or **High Impact** using basic disaster details."
)

st.info(
    "The model uses only pre-impact features such as location, disaster type, year, "
    "month, magnitude, and magnitude scale. Final impact values such as deaths, affected "
    "population, and damage are not used as input features."
)


st.sidebar.header("Disaster Event Input")


country = st.sidebar.text_input("Country", value="India")

region = st.sidebar.selectbox(
    "Region",
    [
        "Africa",
        "Americas",
        "Asia",
        "Europe",
        "Oceania",
        "Unknown"
    ],
    index=2
)

disaster_group = st.sidebar.selectbox(
    "Disaster Group",
    [
        "Natural",
        "Technological",
        "Unknown"
    ],
    index=0
)

disaster_subgroup = st.sidebar.selectbox(
    "Disaster Subgroup",
    [
        "Hydrological",
        "Meteorological",
        "Geophysical",
        "Climatological",
        "Biological",
        "Extra-terrestrial",
        "Unknown"
    ],
    index=0
)

disaster_type = st.sidebar.selectbox(
    "Disaster Type",
    [
        "Flood",
        "Storm",
        "Earthquake",
        "Drought",
        "Epidemic",
        "Landslide",
        "Extreme temperature",
        "Wildfire",
        "Volcanic activity",
        "Mass movement (dry)",
        "Unknown"
    ],
    index=0
)

disaster_subtype = st.sidebar.text_input(
    "Disaster Subtype",
    value="Flood (General)"
)

start_year = st.sidebar.number_input(
    "Start Year",
    min_value=1900,
    max_value=2100,
    value=2024,
    step=1
)

start_month = st.sidebar.number_input(
    "Start Month",
    min_value=1,
    max_value=12,
    value=7,
    step=1
)

end_year = st.sidebar.number_input(
    "End Year",
    min_value=1900,
    max_value=2100,
    value=2024,
    step=1
)

magnitude = st.sidebar.number_input(
    "Magnitude",
    min_value=0.0,
    value=1000.0,
    step=1.0
)

magnitude_scale = st.sidebar.selectbox(
    "Magnitude Scale",
    [
        "Km2",
        "Richter",
        "Kph",
        "m3",
        "Vaccinated",
        "Affected",
        "Unknown"
    ],
    index=0
)


event_data = {
    "country": country,
    "region": region,
    "disaster_group": disaster_group,
    "disaster_subgroup": disaster_subgroup,
    "disaster_type": disaster_type,
    "disaster_subtype": disaster_subtype,
    "start_year": int(start_year),
    "start_month": int(start_month),
    "end_year": int(end_year),
    "magnitude": float(magnitude),
    "magnitude_scale": magnitude_scale,
}


st.subheader("Input Summary")

input_df = pd.DataFrame([event_data])
st.dataframe(input_df, use_container_width=True)


if st.button("Predict Severity"):
    try:
        prediction, probabilities = predict_single_event(event_data)

        st.subheader("Prediction Result")

        if prediction == "High Impact":
            st.error(f"Predicted Severity: {prediction}")
        elif prediction == "Medium Impact":
            st.warning(f"Predicted Severity: {prediction}")
        else:
            st.success(f"Predicted Severity: {prediction}")

        st.subheader("Class Probabilities")

        probabilities_display = probabilities.copy()
        probabilities_display["probability"] = probabilities_display["probability"].round(4)

        st.dataframe(probabilities_display, use_container_width=True)

        st.bar_chart(
            probabilities.set_index("severity_class")["probability"]
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.write(e)


st.markdown("---")

st.write(
    "This project is a tabular machine learning system for disaster analytics. "
    "It follows the project plan of building a multi-class classifier for low, medium, "
    "and high disaster impact prediction using EM-DAT disaster records."
)