"""
Experiment: Compare 3 input configurations for body measurement prediction.
  Baseline: stature, weightkg, Gender, shape  (4 inputs) - reference MAE only
  Approach 1: stature, weightkg, Gender, Age  (4 inputs)
  Approach 2: stature, weightkg, Gender       (3 inputs)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from keras.layers import Dense
from keras import Model, Input
from keras.optimizers import Adam

DATA_DIR = "dataset"
dsm_orig = pd.read_csv(f"{DATA_DIR}/ANSUR II MALE Public.csv", encoding="ISO-8859-1")
dsf_orig = pd.read_csv(f"{DATA_DIR}/ANSUR II FEMALE Public.csv", encoding="ISO-8859-1")

# ---- Helper: build y targets (same logic as train.py) ----
def build_targets(dsm, dsf):
    ds_ym_raw = dsm.iloc[:, list(range(1, 75)) + list(range(76, 91)) + [92, 93]]
    ds_ym = ds_ym_raw.loc[:, [
        "sleevelengthspinewrist", "biacromialbreadth", "chestcircumference",
        "waistcircumference", "neckcircumferencebase", "waistbacklength",
        "bicepscircumferenceflexed", "functionalleglength", "buttockcircumference",
        "thighcircumference", "wristcircumference"
    ]].copy()
    ds_ym.columns = [
        "Sleeves Length", "Shoulder Width", "Chest Around", "Waist", "Neck",
        "Torso Length", "Bicep Around", "Leg Length", "Hip", "Thigh", "Wrist"
    ]
    ds_ym["Rise"] = ds_ym_raw.crotchlengthomphalion - ds_ym_raw.crotchlengthposterioromphalion
    ds_ym["Breast Point"] = (
        ds_ym_raw.verticaltrunkcircumferenceusa
        - ds_ym_raw.crotchlengthomphalion
        - (ds_ym_raw.cervicaleheight - ds_ym_raw.waistheightomphalion)
        - (ds_ym_raw.chestheight - ds_ym_raw.waistheightomphalion)
    )

    ds_yf_raw = dsf.iloc[:, list(range(1, 75)) + list(range(76, 91)) + [92, 93]]
    ds_yf = ds_yf_raw.loc[:, [
        "sleevelengthspinewrist", "biacromialbreadth", "chestcircumference",
        "waistcircumference", "neckcircumferencebase", "waistbacklength",
        "bicepscircumferenceflexed", "functionalleglength", "buttockcircumference",
        "thighcircumference", "wristcircumference"
    ]].copy()
    ds_yf.columns = [
        "Sleeves Length", "Shoulder Width", "Chest Around", "Waist", "Neck",
        "Torso Length", "Bicep Around", "Leg Length", "Hip", "Thigh", "Wrist"
    ]
    ds_yf["Rise"] = ds_yf_raw.crotchlengthomphalion - ds_yf_raw.crotchlengthposterioromphalion
    ds_yf["Breast Point"] = (
        ds_yf_raw.verticaltrunkcircumferenceusa
        - ds_yf_raw.crotchlengthomphalion
        - (ds_yf_raw.cervicaleheight - ds_yf_raw.waistheightomphalion)
        - (ds_yf_raw.chestheight - ds_yf_raw.waistheightomphalion)
    )
    return ds_ym, ds_yf, ds_yf_raw

# ---- Helper: female shape filter (returns indices to remove) ----
def get_female_remove_indices(ds_yf):
    # Temporarily rename for shape logic
    yf = ds_yf.copy()
    yf.columns = [
        "Sleeves Length", "Shoulder Width", "bust", "waist", "Neck",
        "Torso Length", "Bicep Around", "Leg Length", "hip", "Thigh", "Wrist",
        "Rise", "Breast Point"
    ]
    rem_ind = []
    for i in range(yf.shape[0]):
        bust = yf.bust.iloc[i]
        hip = yf.hip.iloc[i]
        waist = yf.waist.iloc[i]
        cond1 = (bust - hip) <= 25 and (hip - bust) < 91 and ((bust - waist) >= 230 or (hip - waist) >= 250)
        cond2 = (bust - hip) > 25 and (bust - hip) < 250 and (bust - waist) >= 230
        cond3 = (hip - bust) >= 91 and (hip - bust) < 250 and (hip - waist) >= 230
        hour = cond1 or cond2 or cond3
        tri = (hip - bust) >= 91 and (hip - waist) < 230
        invtri = (bust - hip) >= 91 and (bust - waist) < 230
        rect = (hip - bust) < 91 and (bust - hip) < 91 and (bust - waist) < 230 and (hip - waist) < 250
        if not (hour or tri or invtri or rect):
            rem_ind.append(i)
    return rem_ind

# ---- Helper: build shape column (for baseline reference) ----
def build_shape_col(ds_Xm, ds_ym, ds_Xf, ds_yf):
    ds_Xm = ds_Xm.copy()
    ds_Xm["shape"] = ""
    for i in range(ds_ym.shape[0]):
        chest = ds_ym["Chest Around"].iloc[i]
        waist = ds_ym["Waist"].iloc[i]
        hip = ds_ym["Hip"].iloc[i]
        if waist >= chest and waist >= hip:
            ds_Xm.loc[ds_Xm.index[i], "shape"] = "oval"
        elif (chest - hip) >= 91 and (chest - waist) >= 230:
            ds_Xm.loc[ds_Xm.index[i], "shape"] = "inverted_triangle"
        elif (chest - hip) >= 25 and (chest - hip) < 91 and (chest - waist) >= 100:
            ds_Xm.loc[ds_Xm.index[i], "shape"] = "trapezoid"
        elif (hip - chest) >= 91:
            ds_Xm.loc[ds_Xm.index[i], "shape"] = "triangle"
        else:
            ds_Xm.loc[ds_Xm.index[i], "shape"] = "rectangle"

    # Female shapes
    yf = ds_yf.copy()
    yf.columns = [
        "Sleeves Length", "Shoulder Width", "bust", "waist", "Neck",
        "Torso Length", "Bicep Around", "Leg Length", "hip", "Thigh", "Wrist",
        "Rise", "Breast Point"
    ]
    ds_Xf = ds_Xf.copy()
    ds_Xf["shape"] = ""
    for i in range(yf.shape[0]):
        bust = yf.bust.iloc[i]
        hip = yf.hip.iloc[i]
        waist = yf.waist.iloc[i]
        cond1 = (bust - hip) <= 25 and (hip - bust) < 91 and ((bust - waist) >= 230 or (hip - waist) >= 250)
        cond2 = (bust - hip) > 25 and (bust - hip) < 250 and (bust - waist) >= 230
        cond3 = (hip - bust) >= 91 and (hip - bust) < 250 and (hip - waist) >= 230
        if cond1 or cond2 or cond3:
            ds_Xf.loc[i, "shape"] = "hourglass"
        elif (hip - bust) >= 91 and (hip - waist) < 230:
            ds_Xf.loc[i, "shape"] = "triangle"
        elif (bust - hip) >= 91 and (bust - waist) < 230:
            ds_Xf.loc[i, "shape"] = "inverted_triangle"
        else:
            ds_Xf.loc[i, "shape"] = "rectangle"

    shape_encoder = OrdinalEncoder()
    combined_shapes = pd.concat([ds_Xm["shape"], ds_Xf["shape"]], ignore_index=True)
    shape_encoder.fit(combined_shapes.to_numpy().reshape(-1, 1))
    ds_Xm["shape"] = shape_encoder.transform(ds_Xm["shape"].to_numpy().reshape(-1, 1)).flatten()
    ds_Xf["shape"] = shape_encoder.transform(ds_Xf["shape"].to_numpy().reshape(-1, 1)).flatten()
    return ds_Xm, ds_Xf

# ---- Model definition (same architecture) ----
def def_model(input_dims):
    inputs = Input(shape=(input_dims,))
    x = Dense(20, activation="relu", kernel_initializer="he_uniform")(inputs)
    x = Dense(50, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(100, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(100, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(50, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(25, activation="relu", kernel_initializer="he_uniform")(x)
    output = Dense(13, kernel_initializer="he_uniform", name="reg")(x)
    return Model(inputs=inputs, outputs=output)

# ---- Train and evaluate ----
def train_and_eval(ds_X, ds_y, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Input columns: {list(ds_X.columns)}")
    print(f"{'='*60}")

    # Encode gender
    ds_X = ds_X.copy()
    ds_X["Gender"] = OrdinalEncoder().fit_transform(
        ds_X["Gender"].to_numpy().reshape(-1, 1)
    ).flatten()

    # Combine, shuffle, split
    ds = pd.concat([ds_X.reset_index(drop=True), ds_y.reset_index(drop=True)], axis=1)
    ds = ds.sample(frac=1).reset_index(drop=True)

    n_inputs = ds_X.shape[1]
    X = ds.iloc[:, :n_inputs].to_numpy()
    y = ds.iloc[:, n_inputs:].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, Targets: {y.shape[1]}")

    # Find gender column index (it's always at index 2 in our feature set)
    gender_col = 2

    model = def_model(n_inputs)
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=[tf.keras.losses.MeanSquaredError()],
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )

    model.fit(
        x=X_train, y=y_train,
        batch_size=64, epochs=60,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", min_delta=1e-4, patience=10,
                verbose=1, mode="auto", restore_best_weights=True
            ),
        ],
        verbose=0,
    )

    # Split test by gender
    gender_max = np.max(X[:, gender_col])
    gender_min = np.min(X[:, gender_col])

    X_test_m = X_test[np.where(X_test[:, gender_col] == gender_max)]
    X_test_f = X_test[np.where(X_test[:, gender_col] == gender_min)]
    y_test_m = y_test[np.where(X_test[:, gender_col] == gender_max)]
    y_test_f = y_test[np.where(X_test[:, gender_col] == gender_min)]

    preds_m = model.predict(X_test_m, verbose=0)
    preds_f = model.predict(X_test_f, verbose=0)

    mae_m = mean_absolute_error(y_test_m, preds_m)
    mae_f = mean_absolute_error(y_test_f, preds_f)

    print(f"\n  >> Male MAE:   {mae_m:.2f} mm")
    print(f"  >> Female MAE: {mae_f:.2f} mm")
    return mae_m, mae_f


# ===========================================================================
# Build targets
# ===========================================================================
ds_ym, ds_yf, _ = build_targets(dsm_orig, dsf_orig)

# Remove female outliers (same as train.py)
rem_ind = get_female_remove_indices(ds_yf)
print(f"Removing {len(rem_ind)} female outlier(s)")
ds_yf = ds_yf.drop(index=rem_ind).reset_index(drop=True)

# Prepare base X columns
ds_Xm_base = dsm_orig.iloc[:, [75, 91, 94]].copy()  # stature, weightkg, Gender
ds_Xf_base = dsf_orig.iloc[:, [75, 91, 94]].copy()
ds_Xf_base = ds_Xf_base.drop(index=rem_ind).reset_index(drop=True)

# Restore y column names consistently
ds_yf.columns = [
    "Sleeves Length", "Shoulder Width", "Chest Around", "Waist", "Neck",
    "Torso Length", "Bicep Around", "Leg Length", "Hip", "Thigh", "Wrist",
    "Rise", "Breast Point"
]

# ===========================================================================
# BASELINE: stature, weightkg, Gender, shape
# ===========================================================================
ds_Xm_bl, ds_Xf_bl = build_shape_col(ds_Xm_base.copy(), ds_ym, ds_Xf_base.copy(), ds_yf)
ds_X_bl = pd.concat([ds_Xm_bl, ds_Xf_bl], ignore_index=True)
ds_y_bl = pd.concat([ds_ym, ds_yf], ignore_index=True)
mae_m_bl, mae_f_bl = train_and_eval(ds_X_bl, ds_y_bl, "BASELINE: stature + weightkg + Gender + shape")

# ===========================================================================
# APPROACH 1: stature, weightkg, Gender, Age
# ===========================================================================
ds_Xm_a1 = dsm_orig.iloc[:, [75, 91, 94]].copy()
ds_Xm_a1["Age"] = dsm_orig.iloc[:, 104].values

ds_Xf_a1 = dsf_orig.iloc[:, [75, 91, 94]].copy()
ds_Xf_a1["Age"] = dsf_orig.iloc[:, 104].values
ds_Xf_a1 = ds_Xf_a1.drop(index=rem_ind).reset_index(drop=True)

ds_X_a1 = pd.concat([ds_Xm_a1, ds_Xf_a1], ignore_index=True)
ds_y_a1 = pd.concat([ds_ym, ds_yf], ignore_index=True)
mae_m_a1, mae_f_a1 = train_and_eval(ds_X_a1, ds_y_a1, "APPROACH 1: stature + weightkg + Gender + Age")

# ===========================================================================
# APPROACH 2: stature, weightkg, Gender (3 inputs)
# ===========================================================================
ds_X_a2 = pd.concat([ds_Xm_base, ds_Xf_base], ignore_index=True)
ds_y_a2 = pd.concat([ds_ym, ds_yf], ignore_index=True)
mae_m_a2, mae_f_a2 = train_and_eval(ds_X_a2, ds_y_a2, "APPROACH 2: stature + weightkg + Gender (no 4th input)")

# ===========================================================================
# COMPARISON TABLE
# ===========================================================================
print("\n" + "=" * 70)
print("  COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Approach':<50} {'Male MAE':>10} {'Female MAE':>10}")
print("-" * 70)
print(f"{'Baseline (shape)  [ref: ~19.3 / ~20.3]':<50} {mae_m_bl:>10.2f} {mae_f_bl:>10.2f}")
print(f"{'Approach 1 (Age instead of shape)':<50} {mae_m_a1:>10.2f} {mae_f_a1:>10.2f}")
print(f"{'Approach 2 (drop 4th input entirely)':<50} {mae_m_a2:>10.2f} {mae_f_a2:>10.2f}")
print("-" * 70)
print("All MAE values in mm.")
print()
