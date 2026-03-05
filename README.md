# Body Measurement Prediction from Anthropometric Data

A machine learning system that predicts 13 body measurements from just 4 inputs: **height**, **weight**, **gender**, and **age**. Built with TensorFlow/Keras, trained on two combined datasets (ANSUR II + BodyM), and served through an interactive Streamlit web UI.

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Predicted Measurements](#predicted-measurements)
- [Model Performance](#model-performance)
- [Training Experiments](#training-experiments)
- [Project Structure](#project-structure)
- [Setup and Usage](#setup-and-usage)
- [References](#references)

---

## Overview

The goal is to predict detailed body measurements for garment sizing using minimal user inputs. A user provides their height, weight, gender, and age, and the model returns measurements like chest circumference, waist, hip, sleeve length, etc.

This is useful for:
- Online clothing size recommendation
- Custom/made-to-measure garment fitting
- Virtual try-on applications
- Body measurement estimation when physical measurement isn't possible

## Datasets

The model is trained on two combined datasets for better generalization across diverse populations.

### ANSUR II (Primary Dataset)

**2012 US Army Anthropometric Survey** conducted by the Natick Soldier Research, Development and Engineering Center (NSRDEC) from October 2010 to April 2012.

| Split | Subjects | Source File |
|-------|----------|-------------|
| Male | 4,082 | `ANSUR II MALE Public.csv` |
| Female | 1,986 | `ANSUR II FEMALE Public.csv` |
| **Total** | **6,068** | |

Each subject has 93 directly-measured anthropometric dimensions (in millimeters) plus demographic variables including age. All measurements were taken by trained personnel using standardized procedures.

### BodyM (Supplementary Dataset)

**Amazon BodyM** — the first large public body measurement dataset, hosted on AWS Open Data Registry. Contains 2,505 real subjects with silhouette images paired with 14 body measurements.

| Split | Subjects | Source |
|-------|----------|--------|
| Male | 1,418 | General public |
| Female | 1,079 | General public |
| **Total** | **2,497** (after filtering to valid height/weight range) | |

BodyM provides 7 of our 13 target measurements (shoulder, chest, waist, bicep, hip, thigh, wrist). The model uses a **masked loss function** that only trains on available measurements for BodyM samples.

**Key difference**: BodyM does not include age, so a default age of 25 is used for BodyM samples during training.

### Combined Training Data

| | ANSUR II | BodyM | Combined |
|---|---|---|---|
| **Subjects** | 6,068 | 2,497 | **8,565** |
| **Population** | US Military | General Public | Mixed |
| **Measurements available** | 13/13 | 7/13 | Masked loss handles partial data |
| **Age** | 17–58 years | Default 25 | Mixed |

## Model Architecture

A fully-connected feedforward neural network with a wider architecture optimized through experimentation:

```
Input(4) -> Dense(64, ReLU) -> Dense(128, ReLU) -> Dense(256, ReLU) -> Dense(256, ReLU) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(13, Linear)
```

| Parameter | Value |
|-----------|-------|
| Total parameters | ~150K |
| Optimizer | Adam (lr=0.005) |
| Loss function | Masked Huber Loss (delta=50mm) |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping | patience=15, restore best weights |
| Train/test split | 50/50 (random_state=42) |
| Input scaling | StandardScaler on all 4 features |
| Hidden activation | ReLU with He uniform initialization |
| Output activation | Linear (regression) |

### Why Huber Loss?

Standard MSE is sensitive to outliers — a few subjects with unusual body proportions can dominate the loss and distort predictions for the majority. **Huber loss** (with delta=50mm / 5cm) behaves like MSE for small errors but switches to linear penalty for large errors, making it robust to outliers. This was especially important when combining two datasets with slightly different measurement methodologies.

### Why Masked Loss?

BodyM only has 7 of 13 measurements. The masked loss function detects `NaN` values in the target and excludes them from gradient computation. This allows the model to learn from BodyM's available measurements without being penalized for the missing ones.

### Input Features (4)

| Feature | Description | Units | Range |
|---------|-------------|-------|-------|
| Height | Stature | mm (internally) | 140–210 cm |
| Weight | Body weight | 10g units (internally) | 40–150 kg |
| Gender | Male/Female | Ordinal encoded (Female=0, Male=1) | - |
| Age | Age in years | Years | 17–65 |

## Predicted Measurements

The model outputs 13 measurements. Not all are relevant for both genders:

| # | Measurement | Male | Female | Description |
|---|-------------|------|--------|-------------|
| 1 | Sleeves Length | Yes | Yes | Spine to wrist length |
| 2 | Shoulder Width | Yes | Yes | Biacromial breadth |
| 3 | Chest Around | Yes | Yes | Chest circumference |
| 4 | Waist | Yes | Yes | Waist circumference |
| 5 | Neck | Yes | - | Neck circumference at base |
| 6 | Torso Length | Yes | - | Waist back length |
| 7 | Bicep Around | Yes | - | Biceps circumference (flexed) |
| 8 | Leg Length | Yes | Yes | Functional leg length |
| 9 | Hip | Yes | Yes | Buttock circumference |
| 10 | Thigh | Yes | Yes | Thigh circumference |
| 11 | Wrist | Yes | - | Wrist circumference |
| 12 | Rise | Yes | - | Crotch length difference (front - back) |
| 13 | Breast Point | - | Yes | Vertical distance derived from trunk measurements |

### Derived Measurements

Two measurements are computed from raw ANSUR II columns:

- **Rise** = `crotchlengthomphalion - crotchlengthposterioromphalion`
- **Breast Point** = `verticaltrunkcircumferenceusa - crotchlengthomphalion - (cervicaleheight - waistheightomphalion) - (chestheight - waistheightomphalion)`

## Model Performance

### Current Model (Wide + Huber, Combined Data)

Mean Absolute Error (MAE) across both datasets:

| Dataset | Male MAE | Female MAE |
|---------|----------|------------|
| ANSUR II (in-distribution) | 1.89 cm | 2.49 cm |
| BodyM (cross-dataset) | 1.97 cm | 1.94 cm |

### Per-Measurement Accuracy (ANSUR II Test Set)

| Measurement | Male MAE (mm) | Female MAE (mm) |
|-------------|---------------|-----------------|
| Sleeves Length | ~19 | ~18 |
| Shoulder Width | ~16 | ~19 |
| Chest Around | ~26 | ~34 |
| Waist | ~33 | ~37 |
| Neck | ~16 | - |
| Torso Length | ~17 | - |
| Bicep Around | ~15 | - |
| Leg Length | ~26 | ~24 |
| Hip | ~21 | ~24 |
| Thigh | ~29 | ~38 |
| Wrist | ~6 | - |
| Rise | ~21 | - |
| Breast Point | - | ~30 |

**Interpretation**: The model achieves ~2 cm average error across all measurements. Circumference measurements (waist, chest, hip) have higher error because they depend heavily on body fat distribution, which height and weight alone can't fully capture. Length and skeletal measurements (wrist, shoulder, sleeve) are predicted more accurately.

## Training Experiments

Four approaches were tested to find the optimal training strategy for combined data. All used the same train/test split for fair comparison:

| Experiment | ANSUR Male | ANSUR Female | BodyM Male | BodyM Female | Overall AVG |
|---|---|---|---|---|---|
| **Exp3: Wide + Huber (selected)** | **1.89 cm** | **2.49 cm** | **1.97 cm** | **1.94 cm** | **2.07 cm** |
| Exp1: Two-Phase Training | 1.95 cm | 2.55 cm | 2.80 cm | 2.53 cm | 2.46 cm |
| Exp4: Two-Phase + Wide + BN | 1.97 cm | 2.63 cm | 2.86 cm | 2.55 cm | 2.50 cm |
| Baseline: Combined v1 (MSE) | 2.10 cm | 2.80 cm | 2.80 cm | 2.40 cm | 2.53 cm |
| Exp2: BN + Dropout + Weights | 2.02 cm | 2.73 cm | 2.87 cm | 2.66 cm | 2.57 cm |
| Original: ANSUR only | 2.00 cm | 2.10 cm | 3.80 cm | 3.60 cm | 2.88 cm |

**Key findings**:
- The wider network (64→128→256→256→128→64) significantly outperforms the original narrow architecture
- Huber loss provides substantial improvement over MSE, especially for cross-dataset generalization
- Two-phase training (pretrain on ANSUR, finetune on combined) preserves ANSUR accuracy but doesn't improve BodyM as much
- BatchNorm + Dropout didn't help — the model is small enough that regularization isn't needed

## Project Structure

```
size_prediction_ML/
├── dataset/
│   ├── ANSUR II MALE Public.csv        # ANSUR II male data (4,082 subjects)
│   ├── ANSUR II FEMALE Public.csv      # ANSUR II female data (1,986 subjects)
│   ├── ANSUR II Databases Overview.pdf
│   ├── Gordon_2012_ANSURII_a611869.pdf
│   └── Hotzman_2011_ANSURIII_Measurements_a548497.pdf
├── bodym_dataset/
│   ├── train_hwg.csv                   # BodyM train split (height/weight/gender)
│   ├── train_measurements.csv          # BodyM train split (14 measurements)
│   ├── testA_hwg.csv / testA_measurements.csv
│   └── testB_hwg.csv / testB_measurements.csv
├── train.py                  # ANSUR II only training script
├── train_combined.py         # Combined ANSUR II + BodyM training (baseline)
├── train_experiments.py      # All 4 experiment variants
├── validate_bodym.py         # Cross-dataset validation against BodyM
├── app.py                    # Streamlit web UI (Predict + Test Samples + Model Info)
├── model.keras               # Current best model (Exp3: Wide + Huber)
├── scaler.pkl                # StandardScaler fitted on combined data
├── model_original_backup.keras   # Backup of original ANSUR-only model
├── scaler_original_backup.pkl    # Backup of original scaler
├── model_combined.keras      # Baseline combined model
├── model_exp{1-4}.keras      # Experiment models
├── requirements.txt          # Python dependencies
├── predAnthr.ipynb           # Original reference notebook
└── README.md
```

## Setup and Usage

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd size_prediction_ML

# Create virtual environment
python -m venv env
source env/bin/activate  # Linux/Mac
# env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

**ANSUR II only (original):**
```bash
python train.py
```

**Combined ANSUR II + BodyM (current best):**
```bash
python train_experiments.py
```

**Run all experiments:**
```bash
python train_experiments.py
```
This will run 4 experiments and print a comparison table. Each experiment saves its model as `model_exp{N}.keras`.

### Cross-Dataset Validation

```bash
python validate_bodym.py
```
Validates the current `model.keras` against the full BodyM dataset and generates scatter plots.

### Running the Web UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The UI has three tabs:
- **Predict**: Enter height, weight, gender, and age to get predicted measurements
- **Test Samples**: Compare predictions against 6 real male subjects with known measurements (shows error and accuracy %)
- **Model Info**: View training data, architecture, and accuracy details

## References

1. **ANSUR II Dataset**: Gordon, C. C., et al. (2014). 2012 Anthropometric Survey of U.S. Army Personnel: Methods and Summary Statistics. NATICK/TR-15/007.
2. **Measurer's Handbook**: Hotzman, J., et al. (2011). Measurer's Handbook: US Army and Marine Corps Anthropometric Surveys, 2010-2011. NATICK/TR-11/017.
3. **BodyM Dataset**: Satish, V., et al. (2023). BodyM: Body Measurement Dataset. AWS Open Data Registry. Available at: https://registry.opendata.aws/bodym/
4. **Huber Loss**: Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73–101.
