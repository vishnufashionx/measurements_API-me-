"""
Train body measurement model on combined ANSUR II + BodyM datasets.
Uses a masked MSE loss so BodyM samples (which have only 7 of 13 measurements)
contribute gradients only for their available outputs.

Saves:
  - model_combined.keras
  - scaler_combined.pkl
  - log_combined.csv
  - loss_curves_combined.png
  - scatter_plots_combined.png

Original model.keras and scaler.pkl are NOT touched.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from keras.layers import Dense
from keras import Model, Input
from keras.optimizers import Adam

# Check GPU
device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    print("GPU device not found")
else:
    print(f"Found GPU at: {device_name}, tf version: {tf.__version__}")

# =============================================================================
# 1. Output column definitions
# =============================================================================
OUTPUT_COLUMNS = [
    "Sleeves Length", "Shoulder Width", "Chest Around", "Waist", "Neck",
    "Torso Length", "Bicep Around", "Leg Length", "Hip", "Thigh", "Wrist",
    "Rise", "Breast Point",
]

# BodyM -> our model output mapping (indices into OUTPUT_COLUMNS)
BODYM_MAP = {
    "shoulder-breadth": 1,   # Shoulder Width
    "chest":            2,   # Chest Around
    "waist":            3,   # Waist
    "bicep":            6,   # Bicep Around
    "hip":              8,   # Hip
    "thigh":            9,   # Thigh
    "wrist":            10,  # Wrist
}

# =============================================================================
# 2. Load ANSUR II (same as train.py)
# =============================================================================
DATA_DIR = "dataset"
dsm = pd.read_csv(f"{DATA_DIR}/ANSUR II MALE Public.csv", encoding="ISO-8859-1")
dsf = pd.read_csv(f"{DATA_DIR}/ANSUR II FEMALE Public.csv", encoding="ISO-8859-1")

# --- Male ---
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

# --- Female ---
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

# Encode gender
ansur_X["Gender"] = OrdinalEncoder().fit_transform(
    ansur_X["Gender"].to_numpy().reshape(-1, 1)
).flatten()

# Convert to arrays (all in mm / 10g units already)
ansur_X_arr = ansur_X.to_numpy(dtype=np.float64)
ansur_y_arr = ansur_y.to_numpy(dtype=np.float64)

# Track which samples are ANSUR II
ansur_source = np.zeros(len(ansur_X_arr), dtype=int)  # 0 = ANSUR II

print(f"ANSUR II: {ansur_X_arr.shape[0]} samples")

# =============================================================================
# 3. Load BodyM dataset
# =============================================================================
BODYM_DIR = "bodym_dataset"
bodym_hwg_frames, bodym_meas_frames = [], []

for split in ["train", "testA", "testB"]:
    hwg = pd.read_csv(f"{BODYM_DIR}/{split}_hwg.csv")
    meas = pd.read_csv(f"{BODYM_DIR}/{split}_measurements.csv")
    merged = hwg.merge(meas, on="subject_id", how="inner")
    bodym_hwg_frames.append(merged)

bodym_df = pd.concat(bodym_hwg_frames, ignore_index=True)

# Filter to valid range
bodym_df = bodym_df[
    (bodym_df["height_cm"] >= 140) & (bodym_df["height_cm"] <= 210) &
    (bodym_df["weight_kg"] >= 40) & (bodym_df["weight_kg"] <= 150)
].reset_index(drop=True)

print(f"BodyM (after filtering): {len(bodym_df)} samples")

# Build BodyM input array: [height_mm, weight_10g, gender, age]
bodym_height_mm = bodym_df["height_cm"].values * 10.0
bodym_weight_10g = bodym_df["weight_kg"].values * 10.0
bodym_gender = np.where(bodym_df["gender"].values == "male", 1.0, 0.0)
bodym_age = np.full(len(bodym_df), 25.0)  # default age

bodym_X_arr = np.column_stack([bodym_height_mm, bodym_weight_10g, bodym_gender, bodym_age])

# Build BodyM output array: (N, 13) with NaN for missing measurements
bodym_y_arr = np.full((len(bodym_df), 13), np.nan, dtype=np.float64)

for bodym_col, out_idx in BODYM_MAP.items():
    bodym_y_arr[:, out_idx] = bodym_df[bodym_col].values * 10.0  # cm -> mm

bodym_source = np.ones(len(bodym_X_arr), dtype=int)  # 1 = BodyM

# =============================================================================
# 4. Combine, shuffle, scale, split
# =============================================================================
X_all = np.vstack([ansur_X_arr, bodym_X_arr])
y_all = np.vstack([ansur_y_arr, bodym_y_arr])
source_all = np.concatenate([ansur_source, bodym_source])

print(f"Combined: {X_all.shape[0]} samples (ANSUR II: {ansur_X_arr.shape[0]}, BodyM: {bodym_X_arr.shape[0]})")

# Standardize
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

# Split
X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
    X_all, y_all, source_all, test_size=0.5, random_state=42
)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"  Train ANSUR: {np.sum(src_train == 0)}, Train BodyM: {np.sum(src_train == 1)}")
print(f"  Test  ANSUR: {np.sum(src_test == 0)}, Test  BodyM: {np.sum(src_test == 1)}")

# =============================================================================
# 5. Custom masked MSE loss
# =============================================================================
@tf.function
def masked_mse_loss(y_true, y_pred):
    """MSE that ignores NaN targets (for partial-output samples)."""
    mask = tf.math.is_finite(y_true)
    y_true_safe = tf.where(mask, y_true, 0.0)
    y_pred_safe = tf.where(mask, y_pred, 0.0)
    sq_err = tf.square(y_true_safe - y_pred_safe)
    masked_sq_err = tf.where(mask, sq_err, 0.0)
    n_valid = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
    return tf.reduce_sum(masked_sq_err) / n_valid


@tf.function
def masked_mae(y_true, y_pred):
    """MAE that ignores NaN targets."""
    mask = tf.math.is_finite(y_true)
    y_true_safe = tf.where(mask, y_true, 0.0)
    y_pred_safe = tf.where(mask, y_pred, 0.0)
    abs_err = tf.abs(y_true_safe - y_pred_safe)
    masked_abs_err = tf.where(mask, abs_err, 0.0)
    n_valid = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
    return tf.reduce_sum(masked_abs_err) / n_valid

# =============================================================================
# 6. Define model (same architecture)
# =============================================================================
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

# =============================================================================
# 7. Train
# =============================================================================
input_dims = X_all.shape[1]
lr = 0.01
opt = Adam(learning_rate=lr)
batch_size = 64
epochs = 60

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10,
        verbose=1, mode="auto", restore_best_weights=True
    ),
    keras.callbacks.CSVLogger(filename="log_combined.csv"),
]

model = def_model(input_dims)
model.compile(optimizer=opt, loss=masked_mse_loss, metrics=[masked_mae])
model.summary()

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1,
)

# =============================================================================
# 8. Evaluate — split by dataset source and gender
# =============================================================================

# --- ANSUR II test subset ---
ansur_mask = src_test == 0
X_test_ansur = X_test[ansur_mask]
y_test_ansur = y_test[ansur_mask]

# Split ANSUR by gender
ansur_male_mask = X_test_ansur[:, 2] == np.max(X_all[:, 2])
ansur_female_mask = X_test_ansur[:, 2] == np.min(X_all[:, 2])

preds_ansur_m = model.predict(X_test_ansur[ansur_male_mask], verbose=0)
preds_ansur_f = model.predict(X_test_ansur[ansur_female_mask], verbose=0)
y_ansur_m = y_test_ansur[ansur_male_mask]
y_ansur_f = y_test_ansur[ansur_female_mask]

print("\n" + "=" * 70)
print("ANSUR II TEST SET — Per-measurement MAE (mm)")
print("=" * 70)

# Male (all 13 outputs except Breast Point)
print("\nMale:")
for i in range(12):
    mae = mean_absolute_error(y_ansur_m[:, i], preds_ansur_m[:, i])
    print(f"  {OUTPUT_COLUMNS[i]:20s}: {mae:.1f} mm ({mae/10:.1f} cm)")
total_m = mean_absolute_error(y_ansur_m[:, :12], preds_ansur_m[:, :12])
print(f"  {'OVERALL':20s}: {total_m:.1f} mm ({total_m/10:.1f} cm)")

# Female (common + Breast Point)
print("\nFemale:")
female_indices = [0, 1, 2, 3, 7, 8, 9, 12]
for i in female_indices:
    mae = mean_absolute_error(y_ansur_f[:, i], preds_ansur_f[:, i])
    print(f"  {OUTPUT_COLUMNS[i]:20s}: {mae:.1f} mm ({mae/10:.1f} cm)")
y_f_subset = y_ansur_f[:, female_indices]
p_f_subset = preds_ansur_f[:, female_indices]
total_f = mean_absolute_error(y_f_subset, p_f_subset)
print(f"  {'OVERALL':20s}: {total_f:.1f} mm ({total_f/10:.1f} cm)")

# --- BodyM test subset ---
bodym_mask = src_test == 1
X_test_bodym = X_test[bodym_mask]
y_test_bodym = y_test[bodym_mask]

bodym_male_mask = X_test_bodym[:, 2] == np.max(X_all[:, 2])
bodym_female_mask = X_test_bodym[:, 2] == np.min(X_all[:, 2])

preds_bodym_m = model.predict(X_test_bodym[bodym_male_mask], verbose=0)
preds_bodym_f = model.predict(X_test_bodym[bodym_female_mask], verbose=0)
y_bodym_m = y_test_bodym[bodym_male_mask]
y_bodym_f = y_test_bodym[bodym_female_mask]

print("\n" + "=" * 70)
print("BODYM TEST SET — Per-measurement MAE (mm)")
print("=" * 70)

bodym_indices = list(BODYM_MAP.values())
bodym_names = [OUTPUT_COLUMNS[i] for i in bodym_indices]

print("\nMale:")
for idx in bodym_indices:
    valid = np.isfinite(y_bodym_m[:, idx])
    if valid.sum() > 0:
        mae = mean_absolute_error(y_bodym_m[valid, idx], preds_bodym_m[valid, idx])
        print(f"  {OUTPUT_COLUMNS[idx]:20s}: {mae:.1f} mm ({mae/10:.1f} cm)")
# Overall for BodyM male
all_mae_m = []
for idx in bodym_indices:
    valid = np.isfinite(y_bodym_m[:, idx])
    if valid.sum() > 0:
        all_mae_m.append(mean_absolute_error(y_bodym_m[valid, idx], preds_bodym_m[valid, idx]))
print(f"  {'OVERALL':20s}: {np.mean(all_mae_m):.1f} mm ({np.mean(all_mae_m)/10:.1f} cm)")

print("\nFemale:")
all_mae_f = []
for idx in bodym_indices:
    valid = np.isfinite(y_bodym_f[:, idx])
    if valid.sum() > 0:
        mae = mean_absolute_error(y_bodym_f[valid, idx], preds_bodym_f[valid, idx])
        print(f"  {OUTPUT_COLUMNS[idx]:20s}: {mae:.1f} mm ({mae/10:.1f} cm)")
        all_mae_f.append(mae)
print(f"  {'OVERALL':20s}: {np.mean(all_mae_f):.1f} mm ({np.mean(all_mae_f)/10:.1f} cm)")

# =============================================================================
# 9. Plot loss curves
# =============================================================================
log_dict = pd.read_csv("log_combined.csv").to_dict(orient="list")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(log_dict["loss"], "r", label="train")
ax1.plot(log_dict["val_loss"], "g", label="test")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss (Masked MSE)")
ax1.set_title("Masked MSE Loss: Train vs Test")
ax1.legend()

ax2.plot(log_dict["masked_mae"], "r", label="train")
ax2.plot(log_dict["val_masked_mae"], "g", label="test")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Masked MAE")
ax2.set_title("Masked MAE: Train vs Test")
ax2.legend()

plt.tight_layout()
plt.savefig("loss_curves_combined.png", dpi=150)
print("\nSaved loss_curves_combined.png")

# =============================================================================
# 10. Scatter plots (ANSUR II test set, same as train.py)
# =============================================================================
fig, axes = plt.subplots(5, 3, constrained_layout=True, figsize=(20, 20))

def plot_graph(ax, i):
    if i in [0, 1, 2, 3, 7, 8, 9]:
        ax.scatter(preds_ansur_m[:, i], y_ansur_m[:, i], c="blue", label="male", alpha=0.5)
        ax.scatter(preds_ansur_f[:, i], y_ansur_f[:, i], c="orange", label="female", alpha=0.5)
    elif i == 12:
        ax.scatter(preds_ansur_f[:, i], y_ansur_f[:, i], c="orange", label="female", alpha=0.5)
    else:
        ax.scatter(preds_ansur_m[:, i], y_ansur_m[:, i], c="blue", label="male", alpha=0.5)

    all_y = np.concatenate([y_ansur_m[:, i], y_ansur_f[:, i]])
    ax.plot([np.min(all_y), np.max(all_y)], [np.min(all_y), np.max(all_y)], "k--")
    ax.set_xlabel("prediction")
    ax.set_ylabel("reality")
    ax.set_title(OUTPUT_COLUMNS[i])
    ax.legend()

k = 0
for i in range(5):
    for j in range(3):
        if k < 13:
            plot_graph(axes[i, j], k)
            k += 1
        else:
            axes[i, j].set_visible(False)

axes[4, 2].set_visible(False)

plt.savefig("scatter_plots_combined.png", dpi=150)
print("Saved scatter_plots_combined.png")

# =============================================================================
# 11. Save model (separate files — do NOT overwrite originals)
# =============================================================================
model.save("model_combined.keras")
joblib.dump(scaler, "scaler_combined.pkl")
print("\nSaved model_combined.keras")
print("Saved scaler_combined.pkl")
print("\nOriginal model.keras and scaler.pkl are UNTOUCHED.")
