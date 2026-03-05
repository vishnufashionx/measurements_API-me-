import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Output columns from training
ALL_COLUMNS = [
    "Sleeves Length", "Shoulder Width", "Chest Around", "Waist", "Neck",
    "Torso Length", "Bicep Around", "Leg Length", "Hip", "Thigh", "Wrist",
    "Rise", "Breast Point",
]

# Relevant measurements per gender
MALE_COLUMNS = [
    "Sleeves Length", "Shoulder Width", "Chest Around", "Waist", "Neck",
    "Torso Length", "Bicep Around", "Leg Length", "Hip", "Thigh", "Wrist", "Rise",
]
FEMALE_COLUMNS = [
    "Sleeves Length", "Shoulder Width", "Chest Around", "Waist",
    "Leg Length", "Hip", "Thigh", "Breast Point",
]

# Test samples: Name, Height(cm), Weight(kg), Age, Actual Chest(cm), Actual Waist(cm), Actual Hip(cm)
TEST_SAMPLES = [
    {"Name": "NIHAL",    "Height": 175, "Weight": 90,  "Age": 25, "Chest": 106, "Waist": 99,  "Hip": 112},
    {"Name": "SAFUAN",   "Height": 172, "Weight": 64,  "Age": 25, "Chest": 90,  "Waist": 83,  "Hip": 97},
    {"Name": "ABILASH",  "Height": 167, "Weight": 70,  "Age": 25, "Chest": 96,  "Waist": 91,  "Hip": 102},
    {"Name": "SABUJI",   "Height": 174, "Weight": 83,  "Age": 25, "Chest": 103, "Waist": 88,  "Hip": 106},
    {"Name": "MATHEW",   "Height": 181, "Weight": 100, "Age": 25, "Chest": 113, "Waist": 104, "Hip": 120},
    {"Name": "ABHIJITH", "Height": 168, "Weight": 52,  "Age": 25, "Chest": 86,  "Waist": 67,  "Hip": 88},
]


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras", compile=False)
    scaler = joblib.load("scaler.pkl")
    return model, scaler


def predict_single(model, scaler, height_cm, weight_kg, gender, age):
    height_mm = height_cm * 10
    weight_10g = int(weight_kg * 10)
    gender_code = 1.0 if gender == "Male" else 0.0

    raw = np.array([[height_mm, weight_10g, gender_code, float(age)]])
    scaled = scaler.transform(raw)
    pred = model.predict(scaled, verbose=0)[0]
    return pred


def predictor_tab(model, scaler):
    st.header("Input Parameters")
    col1, col2 = st.columns(2)

    with col1:
        height_cm = st.number_input("Height (cm)", min_value=140.0, max_value=210.0, value=170.0, step=0.5)
        gender = st.radio("Gender", ["Male", "Female"])

    with col2:
        weight_kg = st.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=70.0, step=0.5)
        age = st.number_input("Age", min_value=17, max_value=65, value=25, step=1)

    if st.button("Predict Measurements", type="primary", use_container_width=True):
        pred = predict_single(model, scaler, height_cm, weight_kg, gender, age)
        relevant_cols = MALE_COLUMNS if gender == "Male" else FEMALE_COLUMNS

        st.header("Predicted Measurements")
        results = []
        for col in relevant_cols:
            idx = ALL_COLUMNS.index(col)
            val_mm = pred[idx]
            val_cm = val_mm / 10.0
            results.append({"Measurement": col, "Value (cm)": round(val_cm, 1), "Value (mm)": round(val_mm, 1)})
        st.table(results)


def test_tab(model, scaler):
    st.header("Test Samples: Prediction Accuracy")
    st.markdown(
        "Each male sample is predicted using height, weight, gender, and age. "
        "Chest, Waist, and Hip predictions (cm) are compared against actual measurements. "
        "**Error = |Predicted - Actual|**."
    )

    chest_idx = ALL_COLUMNS.index("Chest Around")
    waist_idx = ALL_COLUMNS.index("Waist")
    hip_idx = ALL_COLUMNS.index("Hip")

    rows = []
    for sample in TEST_SAMPLES:
        pred = predict_single(
            model, scaler,
            sample["Height"], sample["Weight"], "Male", sample["Age"]
        )

        pred_chest = round(pred[chest_idx] / 10.0, 1)
        pred_waist = round(pred[waist_idx] / 10.0, 1)
        pred_hip = round(pred[hip_idx] / 10.0, 1)

        err_chest = round(abs(pred_chest - sample["Chest"]), 1)
        err_waist = round(abs(pred_waist - sample["Waist"]), 1)
        err_hip = round(abs(pred_hip - sample["Hip"]), 1)
        avg_err = round((err_chest + err_waist + err_hip) / 3.0, 1)

        acc_chest = round((1 - err_chest / sample["Chest"]) * 100, 1)
        acc_waist = round((1 - err_waist / sample["Waist"]) * 100, 1)
        acc_hip = round((1 - err_hip / sample["Hip"]) * 100, 1)
        avg_acc = round((acc_chest + acc_waist + acc_hip) / 3.0, 1)

        rows.append({
            "Name": sample["Name"],
            "Height": sample["Height"],
            "Weight": sample["Weight"],
            "Age": sample["Age"],
            "Chest (Actual)": sample["Chest"],
            "Chest (Pred)": pred_chest,
            "Chest Err": err_chest,
            "Chest Acc%": acc_chest,
            "Waist (Actual)": sample["Waist"],
            "Waist (Pred)": pred_waist,
            "Waist Err": err_waist,
            "Waist Acc%": acc_waist,
            "Hip (Actual)": sample["Hip"],
            "Hip (Pred)": pred_hip,
            "Hip Err": err_hip,
            "Hip Acc%": acc_hip,
            "Avg Error": avg_err,
            "Avg Acc%": avg_acc,
        })

    df = pd.DataFrame(rows)

    # Color code errors: green < 3cm, yellow 3-5cm, red > 5cm
    def color_errors(val):
        if isinstance(val, (int, float)):
            if val <= 3.0:
                return "color: #22c55e; font-weight: bold"
            elif val <= 5.0:
                return "color: #eab308; font-weight: bold"
            else:
                return "color: #ef4444; font-weight: bold"
        return ""

    # Color code accuracy: green >= 97%, yellow 95-97%, red < 95%
    def color_accuracy(val):
        if isinstance(val, (int, float)):
            if val >= 97.0:
                return "color: #22c55e; font-weight: bold"
            elif val >= 95.0:
                return "color: #eab308; font-weight: bold"
            else:
                return "color: #ef4444; font-weight: bold"
        return ""

    err_cols = ["Chest Err", "Waist Err", "Hip Err", "Avg Error"]
    acc_cols = ["Chest Acc%", "Waist Acc%", "Hip Acc%", "Avg Acc%"]
    styled = (
        df.style
        .applymap(color_errors, subset=err_cols)
        .applymap(color_accuracy, subset=acc_cols)
        .format(precision=1)
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary stats
    avg_chest_err = round(df["Chest Err"].mean(), 1)
    avg_waist_err = round(df["Waist Err"].mean(), 1)
    avg_hip_err = round(df["Hip Err"].mean(), 1)
    overall_avg = round(df["Avg Error"].mean(), 1)
    avg_chest_acc = round(df["Chest Acc%"].mean(), 1)
    avg_waist_acc = round(df["Waist Acc%"].mean(), 1)
    avg_hip_acc = round(df["Hip Acc%"].mean(), 1)
    overall_acc = round(df["Avg Acc%"].mean(), 1)

    st.markdown(f"""
    **Summary across all samples:**
    | Metric | Error | Accuracy |
    |---|---|---|
    | Chest | {avg_chest_err} cm | {avg_chest_acc}% |
    | Waist | {avg_waist_err} cm | {avg_waist_acc}% |
    | Hip | {avg_hip_err} cm | {avg_hip_acc}% |
    | **Overall** | **{overall_avg} cm** | **{overall_acc}%** |
    """)


def model_info_tab():
    st.header("Model Details")

    st.subheader("Training Data")
    st.markdown("""
    | Dataset | Subjects | Population |
    |---|---|---|
    | ANSUR II | 6,068 (4,082 M + 1,986 F) | US Military |
    | BodyM | 2,497 (1,418 M + 1,079 F) | General Public |
    | **Total** | **8,565** | |
    """)

    st.subheader("Architecture")
    st.markdown("""
    ```
    Input(4) → Dense(64) → Dense(128) → Dense(256) → Dense(256) → Dense(128) → Dense(64) → Dense(13)
    ```
    - Activation: ReLU with He uniform initialization
    - Loss: Masked Huber (robust to outliers, handles partial outputs)
    - Optimizer: Adam (lr=0.005)
    """)

    st.subheader("Accuracy (MAE in cm)")
    accuracy_data = {
        "Dataset": ["ANSUR II Male", "ANSUR II Female", "BodyM Male", "BodyM Female"],
        "MAE (cm)": [1.89, 2.49, 1.97, 1.94],
    }
    st.table(pd.DataFrame(accuracy_data))

    st.subheader("Inputs")
    st.markdown("""
    | Input | Range | Units |
    |---|---|---|
    | Height | 140 – 210 | cm |
    | Weight | 40 – 150 | kg |
    | Gender | Male / Female | - |
    | Age | 17 – 65 | years |
    """)


def main():
    st.set_page_config(page_title="Body Measurement Predictor", layout="wide")
    st.title("Body Measurement Predictor")

    model, scaler = load_model()

    st.caption("Trained on ANSUR II + BodyM datasets | Wider network + Huber loss | ~2 cm avg error")

    tab1, tab2, tab3 = st.tabs(["Predict", "Test Samples", "Model Info"])

    with tab1:
        predictor_tab(model, scaler)

    with tab2:
        test_tab(model, scaler)

    with tab3:
        model_info_tab()


if __name__ == "__main__":
    main()
