"""
dataset_prep/filter_dataset.py

One-time, local-only script. Filters ODIR-5K to single-label images for our
5 target classes, copies them into per-class folders, and writes a
manifest CSV (filename, class, patient_id) for the train/val/test split
step. Not part of the Pi app — runs on your laptop against the raw download.
"""

import shutil
from pathlib import Path
import pandas as pd

# --- adjust these three paths to match your machine ---
ODIR_CSV = Path(r"F:\capstone\dataset\full_df.csv")
ODIR_TRAIN_IMAGES = Path(r"F:\capstone\dataset\ODIR-5K\ODIR-5K\Training Images")
OUTPUT_DIR = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\filtered")

# our 5 classes -> ODIR's one-letter condition code
CLASS_CODE_MAP = {
    "Normal": "N",
    "Diabetic Retinopathy": "D",
    "Glaucoma": "G",
    "Cataract": "C",
    "AMD": "A",
}
ALL_CODE_COLUMNS = ["N", "D", "G", "C", "A", "H", "M", "O"]  # all 8 ODIR flags


def filter_single_label(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where exactly one of the 8 condition flags is set, AND
    that condition is one of our 5 target classes (drops H/M/O-only rows
    and any multi-condition / comorbid rows)."""
    flags = df[ALL_CODE_COLUMNS]
    is_single = flags.sum(axis=1) == 1
    target_codes = set(CLASS_CODE_MAP.values())
    is_target_class = flags.apply(
        lambda row: row[row == 1].index[0] in target_codes, axis=1
    )
    return df[is_single & is_target_class].copy()


def copy_into_class_folders(df: pd.DataFrame) -> list[dict]:
    """Copies each filtered image into OUTPUT_DIR/<class_name>/. Returns a
    manifest list — we capture patient_id here so the train/test split
    step doesn't have to re-derive it later (patient-level split needed
    to avoid leakage between a patient's left/right eye)."""
    for class_name in CLASS_CODE_MAP:
        (OUTPUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

    manifest = []
    for _, row in df.iterrows():
        flags = row[ALL_CODE_COLUMNS]
        code = flags[flags == 1].index[0]
        class_name = next(name for name, c in CLASS_CODE_MAP.items() if c == code)

        src = ODIR_TRAIN_IMAGES / row["filename"]
        if not src.exists():
            print(f"WARNING: missing file, skipping: {src.name}")
            continue

        dst = OUTPUT_DIR / class_name / row["filename"]
        shutil.copy2(src, dst)  # copy2 = preserve metadata, never mutate raw download

        manifest.append({
            "filename": row["filename"],
            "class": class_name,
            "patient_id": row["ID"],
        })
    return manifest


def main():
    df = pd.read_csv(ODIR_CSV)
    filtered = filter_single_label(df)

    print("Filtered counts per class:")
    print(filtered[ALL_CODE_COLUMNS].sum().loc[list(CLASS_CODE_MAP.values())])

    manifest = copy_into_class_folders(filtered)
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(OUTPUT_DIR / "manifest.csv", index=False)

    print(f"\nDone. {len(manifest)} images copied into {OUTPUT_DIR}")
    print(f"Manifest saved to {OUTPUT_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    main()