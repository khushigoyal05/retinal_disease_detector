"""
dataset_prep/split_dataset.py

Splits the filtered dataset into train/val/test by PATIENT, not by image.
A patient's left/right eye must land in the same split, otherwise the
model can partially "recognize the patient" instead of the disease —
that would quietly inflate test accuracy. Not part of the Pi app —
laptop-only, run once before training.
"""

import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FILTERED_DIR = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\filtered")
MANIFEST_CSV = FILTERED_DIR / "manifest.csv"
SPLIT_DIR = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\split")

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder (0.15) goes to test
RANDOM_SEED = 42  # fixed so the split is reproducible if you re-run this


def split_by_patient(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Two-stage GroupShuffleSplit: first carve out train vs (val+test),
    then split (val+test) into val vs test. Groups = patient_id, so all
    images from one patient always land in the same bucket."""
    gss1 = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)
    train_idx, rest_idx = next(gss1.split(df, groups=df["patient_id"]))
    train_df, rest_df = df.iloc[train_idx], df.iloc[rest_idx]

    val_share_of_rest = VAL_FRAC / (1 - TRAIN_FRAC)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_share_of_rest, random_state=RANDOM_SEED)
    val_idx, test_idx = next(gss2.split(rest_df, groups=rest_df["patient_id"]))
    val_df, test_df = rest_df.iloc[val_idx], rest_df.iloc[test_idx]

    return {"train": train_df, "val": val_df, "test": test_df}


def copy_split(split_name: str, split_df: pd.DataFrame):
    for class_name in split_df["class"].unique():
        (SPLIT_DIR / split_name / class_name).mkdir(parents=True, exist_ok=True)

    for _, row in split_df.iterrows():
        src = FILTERED_DIR / row["class"] / row["filename"]
        dst = SPLIT_DIR / split_name / row["class"] / row["filename"]
        shutil.copy2(src, dst)


def main():
    df = pd.read_csv(MANIFEST_CSV)
    splits = split_by_patient(df)

    print(f"{'Class':<25}{'Train':>8}{'Val':>8}{'Test':>8}")
    for class_name in sorted(df["class"].unique()):
        row = [len(splits[s][splits[s]["class"] == class_name]) for s in ("train", "val", "test")]
        print(f"{class_name:<25}{row[0]:>8}{row[1]:>8}{row[2]:>8}")

    for split_name, split_df in splits.items():
        copy_split(split_name, split_df)
        split_df.to_csv(SPLIT_DIR / f"{split_name}_manifest.csv", index=False)

    print(f"\nDone. Split folders + manifests written to {SPLIT_DIR}")


if __name__ == "__main__":
    main()