import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy import where
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
# 1. Load Dataset
# =============================================================================
DATA_DIR = "dataset"
dsm = pd.read_csv(f"{DATA_DIR}/ANSUR II MALE Public.csv", encoding="ISO-8859-1")
dsf = pd.read_csv(f"{DATA_DIR}/ANSUR II FEMALE Public.csv", encoding="ISO-8859-1")

# =============================================================================
# 2. Split into X (input) and y (output) for both genders
# =============================================================================

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

# =============================================================================
# 3. Combine male + female, shuffle, and prepare final arrays
# =============================================================================
ds_X = pd.concat([ds_Xm, ds_Xf], ignore_index=True)
ds_y = pd.concat([ds_ym, ds_yf], ignore_index=True)

# Encode gender
ds_X["Gender"] = OrdinalEncoder().fit_transform(
    ds_X["Gender"].to_numpy().reshape(-1, 1)
).flatten()

# Combine, shuffle, then re-split
ds = pd.concat([ds_X, ds_y], axis=1, ignore_index=True)
ds.columns = list(ds_X.columns) + list(ds_y.columns)
ds = ds.sample(frac=1).reset_index(drop=True)

ds_X = ds.iloc[:, :4]
ds_y = ds.iloc[:, 4:]

X = ds_X.to_numpy()
y = ds_y.to_numpy(dtype=np.float64)

print(f"Total samples: {X.shape[0]}, Input features: {X.shape[1]}, Output targets: {y.shape[1]}")
print(f"Columns X: {list(ds_X.columns)}")
print(f"Columns y: {list(ds_y.columns)}")
print(f"Age range: {ds_X['Age'].min()} - {ds_X['Age'].max()}")

# =============================================================================
# 4. Standardize and split
# =============================================================================
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# =============================================================================
# 5. Define model
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
# 6. Train
# =============================================================================
input_dims = X.shape[1]
lr = 0.01
opt = Adam(learning_rate=lr)
losses = [tf.keras.losses.MeanSquaredError()]
metrics = [tf.keras.metrics.MeanAbsoluteError(name="mae")]
batch_size = 64
epochs = 60

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10,
        verbose=1, mode="auto", restore_best_weights=True
    ),
    keras.callbacks.CSVLogger(filename="log.csv"),
]

model = def_model(input_dims)
model.compile(optimizer=opt, loss=losses, metrics=metrics)
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
# 7. Evaluate
# =============================================================================

# Split test set by gender
X_test_m = X_test[where(X_test[:, 2] == np.max(X[:, 2]))]
X_test_f = X_test[where(X_test[:, 2] == np.min(X[:, 2]))]
y_test_m = y_test[where(X_test[:, 2] == np.max(X[:, 2]))]
y_test_f = y_test[where(X_test[:, 2] == np.min(X[:, 2]))]

preds_m = model.predict(X_test_m)
preds_f = model.predict(X_test_f)

# Per-measurement metrics
mse_m_total = mean_squared_error(y_test_m, preds_m)
mae_m_total = mean_absolute_error(y_test_m, preds_m)
mse_f_total = mean_squared_error(y_test_f, preds_f)
mae_f_total = mean_absolute_error(y_test_f, preds_f)

dict_test = {"Total reg losses": [mse_m_total, mae_m_total, mse_f_total, mae_f_total]}

# Male: all measurements except Breast Point (index 12)
for i in range(y.shape[1] - 1):
    mse = mean_squared_error(y_test_m[:, i], preds_m[:, i])
    mae = mean_absolute_error(y_test_m[:, i], preds_m[:, i])
    dict_test[ds_y.columns[i]] = [mse, mae]

# Female: only common measurements + Breast Point
for i in [0, 1, 2, 3, 7, 8, 9]:
    mse = mean_squared_error(y_test_f[:, i], preds_f[:, i])
    mae = mean_absolute_error(y_test_f[:, i], preds_f[:, i])
    dict_test[ds_y.columns[i]].extend([mse, mae])

# Breast Point for female
mse = mean_squared_error(y_test_f[:, 12], preds_f[:, 12])
mae = mean_absolute_error(y_test_f[:, 12], preds_f[:, 12])
dict_test[ds_y.columns[12]] = [None, None, mse, mae]

dftest = pd.DataFrame.from_dict(
    dict_test, orient="index",
    columns=["mse_male", "mae_male", "mse_female", "mae_female"]
)
print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)
print(dftest.to_string())

# =============================================================================
# 8. Plot loss curves
# =============================================================================
log_dict = pd.read_csv("log.csv").to_dict(orient="list")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(log_dict["loss"], "r", label="train")
ax1.plot(log_dict["val_loss"], "g", label="test")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss (MSE)")
ax1.set_title("MSE Loss: Train vs Test")
ax1.legend()

ax2.plot(log_dict["mae"], "r", label="train")
ax2.plot(log_dict["val_mae"], "g", label="test")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("MAE")
ax2.set_title("MAE: Train vs Test")
ax2.legend()

plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
print("\nSaved loss_curves.png")

# =============================================================================
# 9. Plot prediction vs reality scatter plots
# =============================================================================
fig, axes = plt.subplots(5, 3, constrained_layout=True, figsize=(20, 20))

def plot_graph(ax, i):
    if i in [0, 1, 2, 3, 7, 8, 9]:  # common measurements
        ax.scatter(preds_m[:, i], y_test_m[:, i], c="blue", label="male", alpha=0.5)
        ax.scatter(preds_f[:, i], y_test_f[:, i], c="orange", label="female", alpha=0.5)
    elif i == 12:  # Breast Point (female only)
        ax.scatter(preds_f[:, i], y_test_f[:, i], c="orange", label="female", alpha=0.5)
    else:  # male-only measurements
        ax.scatter(preds_m[:, i], y_test_m[:, i], c="blue", label="male", alpha=0.5)

    ax.plot(
        [np.min(y_test[:, i]), np.max(y_test[:, i])],
        [np.min(y_test[:, i]), np.max(y_test[:, i])],
        "k--",
    )
    ax.set_xlabel("prediction")
    ax.set_ylabel("reality")
    ax.set_title(ds_y.columns[i])
    ax.legend()

k = 0
for i in range(5):
    for j in range(3):
        if k < 13:
            plot_graph(axes[i, j], k)
            k += 1
        else:
            axes[i, j].set_visible(False)

# Hide the last empty subplot
axes[4, 2].set_visible(False)

plt.savefig("scatter_plots.png", dpi=150)
print("Saved scatter_plots.png")

# Save model and preprocessing artifacts
model.save("model.keras")
joblib.dump(scaler, "scaler.pkl")
print("Saved model.keras")
print("Saved scaler.pkl")
