"""
Validate the trained body measurement model against the BodyM dataset (AWS Open Data).
BodyM has 2,507 subjects with height, weight, gender, and 14 body measurements.

Column mapping (BodyM -> Our model output):
  shoulder-breadth -> Shoulder Width
  chest            -> Chest Around
  waist            -> Waist
  bicep            -> Bicep Around
  hip              -> Hip
  thigh            -> Thigh
  wrist            -> Wrist

Excluded: leg-length (BodyM measures inseam ~78cm, ANSUR II measures
functional leg length ~113cm — different measurement definitions).

Note: BodyM does not have age, so we use a default age of 25.
Note: BodyM measurements are in cm; our model outputs in mm.
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# ── Load model ──────────────────────────────────────────────────────────────
model = tf.keras.models.load_model("model.keras", compile=False)
scaler = joblib.load("scaler.pkl")

# ── Our model's output columns (in order) ──────────────────────────────────
ALL_COLUMNS = [
    "Sleeves Length", "Shoulder Width", "Chest Around", "Waist", "Neck",
    "Torso Length", "Bicep Around", "Leg Length", "Hip", "Thigh", "Wrist",
    "Rise", "Breast Point",
]

# ── Column mapping: BodyM name -> our model column name ─────────────────────
# Only include measurements that have a direct correspondence
COLUMN_MAP = {
    "shoulder-breadth": "Shoulder Width",
    "chest":            "Chest Around",
    "waist":            "Waist",
    "bicep":            "Bicep Around",
    "hip":              "Hip",
    "thigh":            "Thigh",
    "wrist":            "Wrist",
}
# Note: "leg-length" excluded — BodyM measures inseam (~78cm avg) while
# ANSUR II measures functional leg length (~113cm avg). Different definitions.

# ── Load BodyM dataset (all splits) ────────────────────────────────────────
splits = ["train", "testA", "testB"]
hwg_frames, meas_frames = [], []

for split in splits:
    hwg = pd.read_csv(f"bodym_dataset/{split}_hwg.csv")
    meas = pd.read_csv(f"bodym_dataset/{split}_measurements.csv")
    hwg_frames.append(hwg)
    meas_frames.append(meas)

all_hwg = pd.concat(hwg_frames, ignore_index=True)
all_meas = pd.concat(meas_frames, ignore_index=True)
df = all_hwg.merge(all_meas, on="subject_id", how="inner")

print(f"BodyM dataset: {len(df)} subjects ({dict(df['gender'].value_counts())})")
print(f"Height range: {df['height_cm'].min():.1f} – {df['height_cm'].max():.1f} cm")
print(f"Weight range: {df['weight_kg'].min():.1f} – {df['weight_kg'].max():.1f} kg")

# Filter to height/weight ranges the model was trained on (ANSUR II range)
df = df[(df["height_cm"] >= 140) & (df["height_cm"] <= 210)]
df = df[(df["weight_kg"] >= 40) & (df["weight_kg"] <= 150)]
print(f"After filtering to model's training range: {len(df)} subjects")

# ── Prepare inputs ──────────────────────────────────────────────────────────
DEFAULT_AGE = 25  # BodyM has no age column

df_male = df[df["gender"] == "male"].copy()
df_female = df[df["gender"] == "female"].copy()

print(f"Male: {len(df_male)}, Female: {len(df_female)}")


def predict_batch(df_subset, gender_code):
    """Run model predictions for a subset of subjects."""
    height_mm = df_subset["height_cm"].values * 10
    weight_10g = (df_subset["weight_kg"].values * 10).astype(int)
    gender = np.full(len(df_subset), gender_code)
    age = np.full(len(df_subset), float(DEFAULT_AGE))

    raw = np.column_stack([height_mm, weight_10g, gender, age])
    scaled = scaler.transform(raw)
    preds = model.predict(scaled, verbose=0)
    return preds  # shape: (N, 13), in mm


def compute_errors(df_subset, preds, label):
    """Compute MAE for each mapped measurement."""
    results = []
    for bodym_col, our_col in COLUMN_MAP.items():
        idx = ALL_COLUMNS.index(our_col)
        pred_cm = preds[:, idx] / 10.0  # mm -> cm
        actual_cm = df_subset[bodym_col].values
        errors = np.abs(pred_cm - actual_cm)
        mae = np.mean(errors)
        results.append({
            "Measurement": our_col,
            "BodyM Column": bodym_col,
            "MAE (cm)": round(mae, 2),
            "Median Error (cm)": round(np.median(errors), 2),
            "90th %ile Error (cm)": round(np.percentile(errors, 90), 2),
            "Max Error (cm)": round(np.max(errors), 2),
            "N": len(errors),
        })
    return pd.DataFrame(results)


# ── Run predictions ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PREDICTING...")
print("=" * 70)

preds_male = predict_batch(df_male, gender_code=1.0)
preds_female = predict_batch(df_female, gender_code=0.0)

# ── Male results ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MALE RESULTS (BodyM vs Model)")
print("=" * 70)
male_results = compute_errors(df_male, preds_male, "Male")
print(male_results.to_string(index=False))
print(f"\nOverall Male MAE: {male_results['MAE (cm)'].mean():.2f} cm")

# ── Female results ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FEMALE RESULTS (BodyM vs Model)")
print("=" * 70)
female_results = compute_errors(df_female, preds_female, "Female")
print(female_results.to_string(index=False))
print(f"\nOverall Female MAE: {female_results['MAE (cm)'].mean():.2f} cm")

# ── Combined summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY: MAE COMPARISON (cm)")
print("=" * 70)

summary = male_results[["Measurement", "MAE (cm)"]].rename(columns={"MAE (cm)": "Male MAE"})
summary = summary.merge(
    female_results[["Measurement", "MAE (cm)"]].rename(columns={"MAE (cm)": "Female MAE"}),
    on="Measurement", how="outer"
)
print(summary.to_string(index=False))
print(f"\nMale avg:   {summary['Male MAE'].mean():.2f} cm")
print(f"Female avg: {summary['Female MAE'].mean():.2f} cm")

# ── Per-measurement scatter: predicted vs actual ────────────────────────────
try:
    import matplotlib.pyplot as plt

    n_plots = len(COLUMN_MAP)
    cols = 4
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for i, (bodym_col, our_col) in enumerate(COLUMN_MAP.items()):
        ax = axes[i]
        idx = ALL_COLUMNS.index(our_col)

        pred_m_cm = preds_male[:, idx] / 10.0
        actual_m_cm = df_male[bodym_col].values
        pred_f_cm = preds_female[:, idx] / 10.0
        actual_f_cm = df_female[bodym_col].values

        ax.scatter(pred_m_cm, actual_m_cm, c="blue", alpha=0.3, s=10, label="male")
        ax.scatter(pred_f_cm, actual_f_cm, c="orange", alpha=0.3, s=10, label="female")

        all_vals = np.concatenate([actual_m_cm, actual_f_cm])
        mn, mx = all_vals.min(), all_vals.max()
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)

        ax.set_xlabel("Predicted (cm)")
        ax.set_ylabel("Actual (cm)")
        ax.set_title(f"{our_col}\nMAE: M={np.mean(np.abs(pred_m_cm - actual_m_cm)):.1f}, F={np.mean(np.abs(pred_f_cm - actual_f_cm)):.1f} cm")
        ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("BodyM Validation: Predicted vs Actual", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("bodym_validation.png", dpi=150)
    print("\nSaved bodym_validation.png")
except ImportError:
    print("\nmatplotlib not available, skipping plots")
