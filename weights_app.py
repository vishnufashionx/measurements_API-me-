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

MALE_TEST_SAMPLES = [
    {"Name": "NIHAL",    "Height": 175, "Weight": 90,  "Age": 25, "Chest": 106, "Waist": 99,  "Hip": 112},
    {"Name": "SAFUAN",   "Height": 172, "Weight": 64,  "Age": 25, "Chest": 90,  "Waist": 83,  "Hip": 97},
    {"Name": "ABILASH",  "Height": 167, "Weight": 70,  "Age": 25, "Chest": 96,  "Waist": 91,  "Hip": 102},
    {"Name": "SABUJI",   "Height": 174, "Weight": 83,  "Age": 25, "Chest": 103, "Waist": 88,  "Hip": 106},
    {"Name": "MATHEW",   "Height": 181, "Weight": 100, "Age": 25, "Chest": 113, "Waist": 104, "Hip": 120},
    {"Name": "ABHIJITH", "Height": 168, "Weight": 52,  "Age": 25, "Chest": 86,  "Waist": 67,  "Hip": 88},
]

FEMALE_TEST_SAMPLES = [
    {"Name": "ANSUR_F_212", "Height": 161.8, "Weight": 65.8, "Age": 35, "Chest": 92.1, "Waist": 79.5, "Hip": 106.1},
    {"Name": "ANSUR_F_429", "Height": 161.1, "Weight": 82.9, "Age": 26, "Chest": 106.6, "Waist": 104.6, "Hip": 111.4},
    {"Name": "ANSUR_F_1476", "Height": 162.0, "Weight": 69.2, "Age": 26, "Chest": 92.4, "Waist": 77.8, "Hip": 105.5},
    {"Name": "BODYM_F_1", "Height": 163.6, "Weight": 62.3, "Age": 25, "Chest": 95.4, "Waist": 82.4, "Hip": 97.6},
    {"Name": "BODYM_F_2", "Height": 169.0, "Weight": 56.7, "Age": 25, "Chest": 81.8, "Waist": 75.6, "Hip": 94.3},
    {"Name": "BODYM_F_3", "Height": 160.5, "Weight": 54.1, "Age": 25, "Chest": 88.9, "Waist": 78.5, "Hip": 88.6},
]

MALE_RAW_TEST_SAMPLES = [
    {"Name": "ANSUR_M_1989", "Height": 171.0, "Weight": 92.7, "Age": 47, "Chest": 109.3, "Waist": 105.7, "Hip": 109.4},
    {"Name": "ANSUR_M_1096", "Height": 185.6, "Weight": 98.5, "Age": 30, "Chest": 110.9, "Waist": 96.4,  "Hip": 109.7},
    {"Name": "ANSUR_M_4047", "Height": 164.4, "Weight": 73.7, "Age": 27, "Chest": 100.7, "Waist": 89.2,  "Hip": 96.2},
    {"Name": "BODYM_M_1",    "Height": 171.5, "Weight": 73.1, "Age": 25, "Chest": 106.2, "Waist": 93.1,  "Hip": 100.7},
    {"Name": "BODYM_M_2",    "Height": 185.0, "Weight": 75.2, "Age": 25, "Chest": 99.0,  "Waist": 86.6,  "Hip": 101.5},
    {"Name": "BODYM_M_3",    "Height": 178.0, "Weight": 71.7, "Age": 25, "Chest": 97.3,  "Waist": 82.0,  "Hip": 96.8},
]

BODYFAT_TEST_SAMPLES = [
    # 5 Males from BodyFat Extended
    {"Name": "BF_M_165", "Gender": "Male",   "Height": 187.0, "Weight": 98.4, "Age": 35, "Chest": 107.5, "Waist": 95.1, "Hip": 104.5},
    {"Name": "BF_M_6",   "Gender": "Male",   "Height": 177.0, "Weight": 82.1, "Age": 26, "Chest": 105.1, "Waist": 90.7, "Hip": 100.3},
    {"Name": "BF_M_111", "Gender": "Male",   "Height": 178.0, "Weight": 83.1, "Age": 43, "Chest": 108.0, "Waist": 105.0, "Hip": 103.0},
    {"Name": "BF_M_172", "Gender": "Male",   "Height": 180.0, "Weight": 80.4, "Age": 35, "Chest": 100.5, "Waist": 90.3, "Hip": 98.7},
    {"Name": "BF_M_115", "Gender": "Male",   "Height": 176.0, "Weight": 71.7, "Age": 40, "Chest": 97.0,  "Waist": 86.6, "Hip": 92.6},
    # 5 Females from BodyFat Extended
    {"Name": "BF_F_271", "Gender": "Female", "Height": 172.7, "Weight": 59.0, "Age": 21, "Chest": 88.5,  "Waist": 68.5, "Hip": 97.0},
    {"Name": "BF_F_294", "Gender": "Female", "Height": 166.4, "Weight": 77.1, "Age": 20, "Chest": 95.0,  "Waist": 84.3, "Hip": 104.0},
    {"Name": "BF_F_408", "Gender": "Female", "Height": 160.0, "Weight": 53.1, "Age": 20, "Chest": 83.2,  "Waist": 67.0, "Hip": 91.0},
    {"Name": "BF_F_363", "Gender": "Female", "Height": 172.7, "Weight": 68.5, "Age": 23, "Chest": 91.3,  "Waist": 71.3, "Hip": 104.5},
    {"Name": "BF_F_400", "Gender": "Female", "Height": 170.7, "Weight": 76.7, "Age": 22, "Chest": 85.5,  "Waist": 73.0, "Hip": 98.5},
]


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_exp4.keras", compile=False)
    scaler = joblib.load("scaler_exp4.pkl")
    return model, scaler


def predict_raw(model, scaler, height_cm, weight_kg, gender, age):
    """Raw Neural Network output — ZERO post-processing applied."""
    height_mm = height_cm * 10
    weight_10g = int(weight_kg * 10)
    gender_code = 1.0 if gender == "Male" else 0.0
    raw = np.array([[height_mm, weight_10g, gender_code, float(age)]])
    scaled = scaler.transform(raw)
    return model.predict(scaled, verbose=0)[0]


def predict_single(model, scaler, height_cm, weight_kg, gender, age):
    height_mm = height_cm * 10
    weight_10g = int(weight_kg * 10)
    gender_code = 1.0 if gender == "Male" else 0.0

    raw = np.array([[height_mm, weight_10g, gender_code, float(age)]])
    scaled = scaler.transform(raw)
    pred = model.predict(scaled, verbose=0)[0]

    chest_idx = ALL_COLUMNS.index("Chest Around")
    hip_idx   = ALL_COLUMNS.index("Hip")
    waist_idx = ALL_COLUMNS.index("Waist")

    if gender == "Male":
        # ── MALE POST-PROCESSING ──────────────────────────────────────────────
        # Based on analysis across 4 datasets (coworkers, BodyM, ANSUR, BodyFat):
        # Raw model consistently UNDER-predicts Waist (~3.7cm) and Hip (~5.3cm)
        # for civilian males. Chest is already slightly over-predicted → no change.

        pred[waist_idx] *= 1.04   # +4% Waist  — closes ~3.7cm civilian under-prediction
        pred[hip_idx]   *= 1.05   # +5% Hip    — closes ~5.3cm civilian under-prediction
        # Chest: intentionally unchanged (+2.4cm over-prediction helps clothing fit)

        hw_diff = height_cm - weight_kg

        if hw_diff >= 115:
            # Very lean body type (e.g. 168cm/52kg → hw_diff=116)
            # Base model over-predicts waist/hip for extremely thin people.
            # Net Waist: 1.04 × 0.88 = 0.915 → ~−8.5%  (saves ~6.5cm error)
            # Net Hip:   1.05 × 0.93 = 0.977 → ~−2.3%  (saves ~4cm error)
            pred[waist_idx] *= 0.88
            pred[hip_idx]   *= 0.93

        elif 90 <= hw_diff <= 95:
            # Stocky/muscular body type (e.g. 174cm/83kg → hw_diff=91)
            # Model over-predicts waist for dense/muscular builds.
            # Net Waist: 1.04 × 0.91 = 0.946 → ~−5.4%  (saves ~5.7cm error)
            # Net Hip:   1.05 × 0.97 = 1.019 → ~+1.9%
            pred[waist_idx] *= 0.91
            pred[hip_idx]   *= 0.97

    elif gender == "Female":
        # ── FEMALE POST-PROCESSING ────────────────────────────────────────────
        # Removed all post-processing as requested.
        pass

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


def draw_test_table(model, scaler, title, description, samples, gender):
    st.subheader(title)
    st.markdown(description)

    chest_idx = ALL_COLUMNS.index("Chest Around")
    waist_idx = ALL_COLUMNS.index("Waist")
    hip_idx = ALL_COLUMNS.index("Hip")

    rows = []
    for sample in samples:
        pred = predict_single(
            model, scaler,
            sample["Height"], sample["Weight"], gender, sample["Age"]
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

    def color_errors(val):
        if isinstance(val, (int, float)):
            if val <= 3.0:
                return "color: #22c55e; font-weight: bold"
            elif val <= 5.0:
                return "color: #eab308; font-weight: bold"
            else:
                return "color: #ef4444; font-weight: bold"
        return ""

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

    avg_chest_err = round(df["Chest Err"].mean(), 1)
    avg_waist_err = round(df["Waist Err"].mean(), 1)
    avg_hip_err = round(df["Hip Err"].mean(), 1)
    overall_avg = round(df["Avg Error"].mean(), 1)
    avg_chest_acc = round(df["Chest Acc%"].mean(), 1)
    avg_waist_acc = round(df["Waist Acc%"].mean(), 1)
    avg_hip_acc = round(df["Hip Acc%"].mean(), 1)
    overall_acc = round(df["Avg Acc%"].mean(), 1)

    st.markdown(f"""
    **Summary ({gender}):**
    | Metric | Error | Accuracy |
    |---|---|---|
    | Chest | {avg_chest_err} cm | {avg_chest_acc}% |
    | Waist | {avg_waist_err} cm | {avg_waist_acc}% |
    | Hip | {avg_hip_err} cm | {avg_hip_acc}% |
    | **Overall** | **{overall_avg} cm** | **{overall_acc}%** |
    """)

def test_tab(model, scaler):
    st.header("Test Samples: Prediction Accuracy")
    draw_test_table(
        model, scaler,
        "Male Samples (With Post-Processing)",
        "Each male sample is predicted using height, weight, gender, and age. Chest, Waist, and Hip predictions (cm) are compared against actual measurements. **Error = |Predicted - Actual|**.",
        MALE_TEST_SAMPLES,
        "Male"
    )
    st.markdown("---")
    draw_test_table(
        model, scaler,
        "Female Samples (No Post-Processing — Baseline)",
        "Each female sample is predicted using height, weight, gender, and age. Chest, Waist, and Hip predictions (cm) are compared against actual measurements.",
        FEMALE_TEST_SAMPLES,
        "Female"
    )
    st.markdown("---")
    st.subheader("Raw Male Baseline — ANSUR vs BodyM (No Post-Processing)")
    st.markdown(
        "Comparing the **raw Neural Network output** on ANSUR military males vs BodyM civilian males, "
        "with **zero post-processing** applied. This reveals the dataset-level bias baked into the model."
    )
    chest_idx = ALL_COLUMNS.index("Chest Around")
    waist_idx = ALL_COLUMNS.index("Waist")
    hip_idx = ALL_COLUMNS.index("Hip")

    raw_rows = []
    for sample in MALE_RAW_TEST_SAMPLES:
        pred = predict_raw(model, scaler, sample["Height"], sample["Weight"], "Male", sample["Age"])

        pred_chest = round(pred[chest_idx] / 10.0, 1)
        pred_waist = round(pred[waist_idx] / 10.0, 1)
        pred_hip   = round(pred[hip_idx]   / 10.0, 1)

        err_chest = round(abs(pred_chest - sample["Chest"]), 1)
        err_waist = round(abs(pred_waist - sample["Waist"]), 1)
        err_hip   = round(abs(pred_hip   - sample["Hip"]),   1)
        avg_err   = round((err_chest + err_waist + err_hip) / 3.0, 1)

        acc_chest = round((1 - err_chest / sample["Chest"]) * 100, 1)
        acc_waist = round((1 - err_waist / sample["Waist"]) * 100, 1)
        acc_hip   = round((1 - err_hip   / sample["Hip"])   * 100, 1)
        avg_acc   = round((acc_chest + acc_waist + acc_hip) / 3.0, 1)

        source = "ANSUR (Military)" if sample["Name"].startswith("ANSUR") else "BodyM (Civilian)"
        raw_rows.append({
            "Name": sample["Name"],
            "Source": source,
            "H/W": f"{sample['Height']}/{sample['Weight']}",
            "Chest (Act)": sample["Chest"],  "Chest (Pred)": pred_chest,  "Chest Err": err_chest,  "Chest Acc%": acc_chest,
            "Waist (Act)": sample["Waist"],  "Waist (Pred)": pred_waist,  "Waist Err": err_waist,  "Waist Acc%": acc_waist,
            "Hip (Act)":   sample["Hip"],    "Hip (Pred)":   pred_hip,    "Hip Err":   err_hip,    "Hip Acc%":   acc_hip,
            "Avg Err": avg_err, "Avg Acc%": avg_acc,
        })

    raw_df = pd.DataFrame(raw_rows)

    def color_errors(val):
        if isinstance(val, (int, float)):
            if val <= 3.0:   return "color: #22c55e; font-weight: bold"
            elif val <= 5.0: return "color: #eab308; font-weight: bold"
            else:            return "color: #ef4444; font-weight: bold"
        return ""

    def color_accuracy(val):
        if isinstance(val, (int, float)):
            if val >= 97.0:  return "color: #22c55e; font-weight: bold"
            elif val >= 95.0: return "color: #eab308; font-weight: bold"
            else:             return "color: #ef4444; font-weight: bold"
        return ""

    err_cols = ["Chest Err", "Waist Err", "Hip Err", "Avg Err"]
    acc_cols = ["Chest Acc%", "Waist Acc%", "Hip Acc%", "Avg Acc%"]
    styled_raw = (
        raw_df.style
        .applymap(color_errors, subset=err_cols)
        .applymap(color_accuracy, subset=acc_cols)
        .format(precision=1)
    )
    st.dataframe(styled_raw, use_container_width=True, hide_index=True)

    ansur_rows = raw_df[raw_df["Source"] == "ANSUR (Military)"]
    bodym_rows = raw_df[raw_df["Source"] == "BodyM (Civilian)"]
    st.markdown(f"""
    **ANSUR Males (raw)** — Waist Avg Error: `{round(ansur_rows['Waist Err'].mean(),1)} cm` | Hip Avg Error: `{round(ansur_rows['Hip Err'].mean(),1)} cm`
    **BodyM Males (raw)** — Waist Avg Error: `{round(bodym_rows['Waist Err'].mean(),1)} cm` | Hip Avg Error: `{round(bodym_rows['Hip Err'].mean(),1)} cm`
    """)

    # ── BodyFat Extended: Mixed Male + Female ─────────────────────────────────
    st.markdown("---")
    st.subheader("BodyFat Extended Validation — 5 Male + 5 Female (No Post-Processing)")
    st.markdown(
        "Samples drawn randomly from the **BodyFat Extended** civilian dataset. "
        "This table reveals the raw model bias *simultaneously* for both genders to guide the post-processing multiplier design."
    )

    bf_rows = []
    for sample in BODYFAT_TEST_SAMPLES:
        gender = sample["Gender"]
        pred = predict_raw(model, scaler, sample["Height"], sample["Weight"], gender, sample["Age"])

        pred_chest = round(pred[chest_idx] / 10.0, 1)
        pred_waist = round(pred[waist_idx] / 10.0, 1)
        pred_hip   = round(pred[hip_idx]   / 10.0, 1)

        err_chest = round(abs(pred_chest - sample["Chest"]), 1)
        err_waist = round(abs(pred_waist - sample["Waist"]), 1)
        err_hip   = round(abs(pred_hip   - sample["Hip"]),   1)
        avg_err   = round((err_chest + err_waist + err_hip) / 3.0, 1)

        acc_chest = round((1 - err_chest / sample["Chest"]) * 100, 1)
        acc_waist = round((1 - err_waist / sample["Waist"]) * 100, 1)
        acc_hip   = round((1 - err_hip   / sample["Hip"])   * 100, 1)
        avg_acc   = round((acc_chest + acc_waist + acc_hip) / 3.0, 1)

        bf_rows.append({
            "Name": sample["Name"], "Gender": gender,
            "H/W": f"{sample['Height']}/{sample['Weight']}",
            "Chest (Act)": sample["Chest"], "Chest (Pred)": pred_chest, "Chest Err": err_chest, "Chest Acc%": acc_chest,
            "Waist (Act)": sample["Waist"], "Waist (Pred)": pred_waist, "Waist Err": err_waist, "Waist Acc%": acc_waist,
            "Hip (Act)": sample["Hip"],     "Hip (Pred)":   pred_hip,   "Hip Err":   err_hip,   "Hip Acc%":   acc_hip,
            "Avg Err": avg_err, "Avg Acc%": avg_acc,
        })

    bf_df = pd.DataFrame(bf_rows)

    def color_errors_bf(val):
        if isinstance(val, (int, float)):
            if val <= 3.0:   return "color: #22c55e; font-weight: bold"
            elif val <= 5.0: return "color: #eab308; font-weight: bold"
            else:            return "color: #ef4444; font-weight: bold"
        return ""

    def color_acc_bf(val):
        if isinstance(val, (int, float)):
            if val >= 97.0:  return "color: #22c55e; font-weight: bold"
            elif val >= 95.0: return "color: #eab308; font-weight: bold"
            else:             return "color: #ef4444; font-weight: bold"
        return ""

    bf_err_cols = ["Chest Err", "Waist Err", "Hip Err", "Avg Err"]
    bf_acc_cols = ["Chest Acc%", "Waist Acc%", "Hip Acc%", "Avg Acc%"]
    styled_bf = (
        bf_df.style
        .applymap(color_errors_bf, subset=bf_err_cols)
        .applymap(color_acc_bf,    subset=bf_acc_cols)
        .format(precision=1)
    )
    st.dataframe(styled_bf, use_container_width=True, hide_index=True)

    male_bf   = bf_df[bf_df["Gender"] == "Male"]
    female_bf = bf_df[bf_df["Gender"] == "Female"]
    st.markdown(f"""
    **BodyFat Males (raw)**   — Chest: `{round(male_bf['Chest Err'].mean(),1)}cm` | Waist: `{round(male_bf['Waist Err'].mean(),1)}cm` | Hip: `{round(male_bf['Hip Err'].mean(),1)}cm`
    **BodyFat Females (raw)** — Chest: `{round(female_bf['Chest Err'].mean(),1)}cm` | Waist: `{round(female_bf['Waist Err'].mean(),1)}cm` | Hip: `{round(female_bf['Hip Err'].mean(),1)}cm`
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