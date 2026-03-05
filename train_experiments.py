"""
Run 4 training experiments on combined ANSUR II + BodyM data to find the
best approach. Compares results in a final table.

Saves per experiment: model_exp{N}.keras, scaler_exp{N}.pkl
Does NOT touch: model.keras, scaler.pkl, model_combined.keras, scaler_combined.pkl
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from keras.layers import Dense, BatchNormalization, Dropout
from keras import Model, Input
from keras.optimizers import Adam

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_COLUMNS = [
    "Sleeves Length", "Shoulder Width", "Chest Around", "Waist", "Neck",
    "Torso Length", "Bicep Around", "Leg Length", "Hip", "Thigh", "Wrist",
    "Rise", "Breast Point",
]

BODYM_MAP = {
    "shoulder-breadth": 1, "chest": 2, "waist": 3, "bicep": 6,
    "hip": 8, "thigh": 9, "wrist": 10,
}

FEMALE_INDICES = [0, 1, 2, 3, 7, 8, 9, 12]
BODYM_INDICES = list(BODYM_MAP.values())

# ─────────────────────────────────────────────────────────────────────────────
# Custom losses
# ─────────────────────────────────────────────────────────────────────────────
@tf.function
def masked_mse_loss(y_true, y_pred):
    mask = tf.math.is_finite(y_true)
    y_true_safe = tf.where(mask, y_true, 0.0)
    y_pred_safe = tf.where(mask, y_pred, 0.0)
    sq_err = tf.square(y_true_safe - y_pred_safe)
    masked_sq_err = tf.where(mask, sq_err, 0.0)
    n_valid = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
    return tf.reduce_sum(masked_sq_err) / n_valid


@tf.function
def masked_mae_metric(y_true, y_pred):
    mask = tf.math.is_finite(y_true)
    y_true_safe = tf.where(mask, y_true, 0.0)
    y_pred_safe = tf.where(mask, y_pred, 0.0)
    abs_err = tf.abs(y_true_safe - y_pred_safe)
    masked_abs_err = tf.where(mask, abs_err, 0.0)
    n_valid = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
    return tf.reduce_sum(masked_abs_err) / n_valid


@tf.function
def masked_huber_loss(y_true, y_pred):
    delta = 50.0  # mm — errors above 5cm treated linearly
    mask = tf.math.is_finite(y_true)
    y_true_safe = tf.where(mask, y_true, 0.0)
    y_pred_safe = tf.where(mask, y_pred, 0.0)
    diff = tf.abs(y_true_safe - y_pred_safe)
    huber = tf.where(diff <= delta, 0.5 * tf.square(diff), delta * (diff - 0.5 * delta))
    masked_huber = tf.where(mask, huber, 0.0)
    n_valid = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
    return tf.reduce_sum(masked_huber) / n_valid


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (identical to train_combined.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    DATA_DIR = "dataset"
    dsm = pd.read_csv(f"{DATA_DIR}/ANSUR II MALE Public.csv", encoding="ISO-8859-1")
    dsf = pd.read_csv(f"{DATA_DIR}/ANSUR II FEMALE Public.csv", encoding="ISO-8859-1")

    # Male
    ds_Xm = dsm.loc[:, ["stature", "weightkg", "Gender", "Age"]].copy()
    ds_ym_raw = dsm.iloc[:, list(range(1, 75)) + list(range(76, 91)) + [92, 93]]
    ds_ym = ds_ym_raw.loc[:, [
        "sleevelengthspinewrist", "biacromialbreadth", "chestcircumference",
        "waistcircumference", "neckcircumferencebase", "waistbacklength",
        "bicepscircumferenceflexed", "functionalleglength", "buttockcircumference",
        "thighcircumference", "wristcircumference"
    ]]
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

    # Female
    ds_Xf = dsf.loc[:, ["stature", "weightkg", "Gender", "Age"]].copy()
    ds_yf_raw = dsf.iloc[:, list(range(1, 75)) + list(range(76, 91)) + [92, 93]]
    ds_yf = ds_yf_raw.loc[:, [
        "sleevelengthspinewrist", "biacromialbreadth", "chestcircumference",
        "waistcircumference", "neckcircumferencebase", "waistbacklength",
        "bicepscircumferenceflexed", "functionalleglength", "buttockcircumference",
        "thighcircumference", "wristcircumference"
    ]]
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

    # Combine ANSUR II
    ansur_X = pd.concat([ds_Xm, ds_Xf], ignore_index=True)
    ansur_y = pd.concat([ds_ym, ds_yf], ignore_index=True)
    ansur_X["Gender"] = OrdinalEncoder().fit_transform(
        ansur_X["Gender"].to_numpy().reshape(-1, 1)
    ).flatten()
    ansur_X_arr = ansur_X.to_numpy(dtype=np.float64)
    ansur_y_arr = ansur_y.to_numpy(dtype=np.float64)

    # Load BodyM
    BODYM_DIR = "bodym_dataset"
    frames = []
    for split in ["train", "testA", "testB"]:
        hwg = pd.read_csv(f"{BODYM_DIR}/{split}_hwg.csv")
        meas = pd.read_csv(f"{BODYM_DIR}/{split}_measurements.csv")
        frames.append(hwg.merge(meas, on="subject_id", how="inner"))
    bodym_df = pd.concat(frames, ignore_index=True)
    bodym_df = bodym_df[
        (bodym_df["height_cm"] >= 140) & (bodym_df["height_cm"] <= 210) &
        (bodym_df["weight_kg"] >= 40) & (bodym_df["weight_kg"] <= 150)
    ].reset_index(drop=True)

    bodym_X_arr = np.column_stack([
        bodym_df["height_cm"].values * 10.0,
        bodym_df["weight_kg"].values * 10.0,
        np.where(bodym_df["gender"].values == "male", 1.0, 0.0),
        np.full(len(bodym_df), 25.0),
    ])
    bodym_y_arr = np.full((len(bodym_df), 13), np.nan, dtype=np.float64)
    for bodym_col, out_idx in BODYM_MAP.items():
        bodym_y_arr[:, out_idx] = bodym_df[bodym_col].values * 10.0

    # Combine
    X_all = np.vstack([ansur_X_arr, bodym_X_arr])
    y_all = np.vstack([ansur_y_arr, bodym_y_arr])
    source_all = np.concatenate([
        np.zeros(len(ansur_X_arr), dtype=int),
        np.ones(len(bodym_X_arr), dtype=int),
    ])

    print(f"ANSUR II: {len(ansur_X_arr)}, BodyM: {len(bodym_X_arr)}, Combined: {len(X_all)}")

    return X_all, y_all, source_all, ansur_X_arr, ansur_y_arr


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────
def build_standard_model():
    """Same architecture as current model."""
    inputs = Input(shape=(4,))
    x = Dense(20, activation="relu", kernel_initializer="he_uniform")(inputs)
    x = Dense(50, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(100, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(100, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(50, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(25, activation="relu", kernel_initializer="he_uniform")(x)
    output = Dense(13, kernel_initializer="he_uniform")(x)
    return Model(inputs=inputs, outputs=output)


def build_batchnorm_dropout_model():
    """Standard architecture + BatchNorm + Dropout."""
    inputs = Input(shape=(4,))
    x = Dense(20, kernel_initializer="he_uniform")(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(50, kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(100, kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(100, kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(50, kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = Dense(25, kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    output = Dense(13, kernel_initializer="he_uniform")(x)
    return Model(inputs=inputs, outputs=output)


def build_wide_model():
    """Wider architecture: 64→128→256→256→128→64→13."""
    inputs = Input(shape=(4,))
    x = Dense(64, activation="relu", kernel_initializer="he_uniform")(inputs)
    x = Dense(128, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(256, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(256, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(128, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(64, activation="relu", kernel_initializer="he_uniform")(x)
    output = Dense(13, kernel_initializer="he_uniform")(x)
    return Model(inputs=inputs, outputs=output)


def build_wide_batchnorm_dropout_model():
    """Wider architecture + BatchNorm + Dropout."""
    inputs = Input(shape=(4,))

    x = Dense(64, kernel_initializer="he_uniform")(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(128, kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(256, kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(256, kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(128, kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = Dense(64, kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    output = Dense(13, kernel_initializer="he_uniform")(x)
    return Model(inputs=inputs, outputs=output)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, src_test, X_all, label=""):
    """Return dict with ANSUR Male/Female MAE and BodyM Male/Female MAE (in cm)."""
    results = {}

    # ANSUR II test
    ansur_mask = src_test == 0
    X_a = X_test[ansur_mask]
    y_a = y_test[ansur_mask]
    male_mask = X_a[:, 2] == np.max(X_all[:, 2])
    female_mask = X_a[:, 2] == np.min(X_all[:, 2])

    preds_m = model.predict(X_a[male_mask], verbose=0)
    preds_f = model.predict(X_a[female_mask], verbose=0)
    y_m = y_a[male_mask]
    y_f = y_a[female_mask]

    results["ANSUR_M"] = round(mean_absolute_error(y_m[:, :12], preds_m[:, :12]) / 10, 2)
    results["ANSUR_F"] = round(mean_absolute_error(y_f[:, FEMALE_INDICES], preds_f[:, FEMALE_INDICES]) / 10, 2)

    # BodyM test
    bodym_mask = src_test == 1
    X_b = X_test[bodym_mask]
    y_b = y_test[bodym_mask]
    male_mask_b = X_b[:, 2] == np.max(X_all[:, 2])
    female_mask_b = X_b[:, 2] == np.min(X_all[:, 2])

    preds_bm = model.predict(X_b[male_mask_b], verbose=0)
    preds_bf = model.predict(X_b[female_mask_b], verbose=0)
    y_bm = y_b[male_mask_b]
    y_bf = y_b[female_mask_b]

    # BodyM male
    mae_list = []
    for idx in BODYM_INDICES:
        valid = np.isfinite(y_bm[:, idx])
        if valid.sum() > 0:
            mae_list.append(mean_absolute_error(y_bm[valid, idx], preds_bm[valid, idx]))
    results["BodyM_M"] = round(np.mean(mae_list) / 10, 2)

    # BodyM female
    mae_list = []
    for idx in BODYM_INDICES:
        valid = np.isfinite(y_bf[:, idx])
        if valid.sum() > 0:
            mae_list.append(mean_absolute_error(y_bf[valid, idx], preds_bf[valid, idx]))
    results["BodyM_F"] = round(np.mean(mae_list) / 10, 2)

    results["AVG"] = round(np.mean([results["ANSUR_M"], results["ANSUR_F"],
                                     results["BodyM_M"], results["BodyM_F"]]), 2)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    X_raw, y_all, source_all, ansur_X_raw, ansur_y_raw = load_data()

    # Fit scaler on combined data (once)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_raw)

    # Also scale ANSUR-only subset (for two-phase training)
    ansur_X_scaled = scaler.transform(ansur_X_raw)

    # Split combined
    X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
        X_all, y_all, source_all, test_size=0.5, random_state=42
    )

    # Split ANSUR-only (for phase 1 of two-phase experiments)
    ansur_X_tr, ansur_X_te, ansur_y_tr, ansur_y_te = train_test_split(
        ansur_X_scaled, ansur_y_raw, test_size=0.5, random_state=42
    )

    # Sample weights for Experiment 2
    sample_weights = np.where(src_train == 0, 1.5, 1.0)

    print(f"\nTrain: {len(X_train)} (ANSUR: {np.sum(src_train==0)}, BodyM: {np.sum(src_train==1)})")
    print(f"Test:  {len(X_test)} (ANSUR: {np.sum(src_test==0)}, BodyM: {np.sum(src_test==1)})")

    all_results = {}

    # ═════════════════════════════════════════════════════════════════════════
    # Experiment 1: Two-Phase Training (Pretrain ANSUR + Finetune Combined)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Two-Phase (Pretrain ANSUR II → Finetune Combined)")
    print("=" * 70)

    model1 = build_standard_model()

    # Phase 1: ANSUR II only, standard MSE
    print("\n--- Phase 1: ANSUR II only (lr=0.01, 40 epochs) ---")
    model1.compile(optimizer=Adam(learning_rate=0.01),
                   loss=tf.keras.losses.MeanSquaredError(),
                   metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    model1.fit(ansur_X_tr, ansur_y_tr, batch_size=64, epochs=40,
               validation_data=(ansur_X_te, ansur_y_te),
               callbacks=[tf.keras.callbacks.EarlyStopping(
                   monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
               )], verbose=0)
    print(f"Phase 1 done. Val loss: {model1.evaluate(ansur_X_te, ansur_y_te, verbose=0)[0]:.1f}")

    # Phase 2: Combined, masked MSE, lower LR
    print("\n--- Phase 2: Combined data (lr=0.001, 40 epochs) ---")
    model1.compile(optimizer=Adam(learning_rate=0.001),
                   loss=masked_mse_loss, metrics=[masked_mae_metric])
    model1.fit(X_train, y_train, batch_size=64, epochs=40,
               validation_data=(X_test, y_test),
               callbacks=[tf.keras.callbacks.EarlyStopping(
                   monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
               )], verbose=0)

    all_results["Exp1: Two-Phase"] = evaluate_model(model1, X_test, y_test, src_test, X_all)
    model1.save("model_exp1.keras")
    print(f"Results: {all_results['Exp1: Two-Phase']}")

    # ═════════════════════════════════════════════════════════════════════════
    # Experiment 2: Sample Weights + BatchNorm + Dropout
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Sample Weights + BatchNorm + Dropout")
    print("=" * 70)

    model2 = build_batchnorm_dropout_model()
    model2.compile(optimizer=Adam(learning_rate=0.005),
                   loss=masked_mse_loss, metrics=[masked_mae_metric])
    model2.fit(X_train, y_train, batch_size=64, epochs=100,
               validation_data=(X_test, y_test),
               sample_weight=sample_weights,
               callbacks=[tf.keras.callbacks.EarlyStopping(
                   monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
               )], verbose=0)

    all_results["Exp2: BN+Dropout+Weights"] = evaluate_model(model2, X_test, y_test, src_test, X_all)
    model2.save("model_exp2.keras")
    print(f"Results: {all_results['Exp2: BN+Dropout+Weights']}")

    # ═════════════════════════════════════════════════════════════════════════
    # Experiment 3: Wider Network + Huber Loss
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Wider Network + Huber Loss")
    print("=" * 70)

    model3 = build_wide_model()
    model3.compile(optimizer=Adam(learning_rate=0.005),
                   loss=masked_huber_loss, metrics=[masked_mae_metric])
    model3.fit(X_train, y_train, batch_size=64, epochs=100,
               validation_data=(X_test, y_test),
               callbacks=[tf.keras.callbacks.EarlyStopping(
                   monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
               )], verbose=0)

    all_results["Exp3: Wide+Huber"] = evaluate_model(model3, X_test, y_test, src_test, X_all)
    model3.save("model_exp3.keras")
    print(f"Results: {all_results['Exp3: Wide+Huber']}")

    # ═════════════════════════════════════════════════════════════════════════
    # Experiment 4: Best Combination (Two-Phase + Wide + BN + Dropout)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Two-Phase + Wide + BN + Dropout")
    print("=" * 70)

    model4 = build_wide_batchnorm_dropout_model()

    # Phase 1: ANSUR II only
    print("\n--- Phase 1: ANSUR II only (lr=0.005, 50 epochs) ---")
    model4.compile(optimizer=Adam(learning_rate=0.005),
                   loss=tf.keras.losses.MeanSquaredError(),
                   metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    model4.fit(ansur_X_tr, ansur_y_tr, batch_size=64, epochs=50,
               validation_data=(ansur_X_te, ansur_y_te),
               callbacks=[tf.keras.callbacks.EarlyStopping(
                   monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
               )], verbose=0)
    print(f"Phase 1 done. Val loss: {model4.evaluate(ansur_X_te, ansur_y_te, verbose=0)[0]:.1f}")

    # Phase 2: Combined, masked MSE
    print("\n--- Phase 2: Combined data (lr=0.0005, 60 epochs) ---")
    model4.compile(optimizer=Adam(learning_rate=0.0005),
                   loss=masked_mse_loss, metrics=[masked_mae_metric])
    model4.fit(X_train, y_train, batch_size=64, epochs=60,
               validation_data=(X_test, y_test),
               sample_weight=sample_weights,
               callbacks=[tf.keras.callbacks.EarlyStopping(
                   monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
               )], verbose=0)

    all_results["Exp4: TwoPhase+Wide+BN"] = evaluate_model(model4, X_test, y_test, src_test, X_all)
    model4.save("model_exp4.keras")
    print(f"Results: {all_results['Exp4: TwoPhase+Wide+BN']}")

    # ═════════════════════════════════════════════════════════════════════════
    # Save scaler (same for all experiments)
    # ═════════════════════════════════════════════════════════════════════════
    for i in range(1, 5):
        joblib.dump(scaler, f"scaler_exp{i}.pkl")

    # ═════════════════════════════════════════════════════════════════════════
    # Final comparison table
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL COMPARISON (MAE in cm)")
    print("=" * 70)

    # Add baselines for reference
    all_results["Baseline (combined v1)"] = {
        "ANSUR_M": 2.1, "ANSUR_F": 2.8, "BodyM_M": 2.8, "BodyM_F": 2.4, "AVG": 2.53
    }
    all_results["Original (ANSUR only)"] = {
        "ANSUR_M": 2.0, "ANSUR_F": 2.1, "BodyM_M": 3.8, "BodyM_F": 3.6, "AVG": 2.88
    }

    df = pd.DataFrame(all_results).T
    df = df[["ANSUR_M", "ANSUR_F", "BodyM_M", "BodyM_F", "AVG"]]
    df.columns = ["ANSUR Male", "ANSUR Female", "BodyM Male", "BodyM Female", "Overall AVG"]
    df = df.sort_values("Overall AVG")
    print(df.to_string())

    # Highlight best
    best = df.index[0]
    print(f"\n★ Best overall: {best} (avg MAE = {df.iloc[0]['Overall AVG']:.2f} cm)")


if __name__ == "__main__":
    main()
