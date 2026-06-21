"""
Central configuration for the Retinal Disease Detection System.

Why this file exists: medical/embedded apps must avoid scattering magic
numbers, file paths, and thresholds across the codebase. One source of
truth makes the system easy to tune, audit, and redeploy on a different
Pi unit without touching application logic.
"""

from pathlib import Path

# Resolved dynamically so the app works regardless of where it's launched
# from (systemd service, terminal, IDE) — never hardcode an absolute path.
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
CAPTURE_DIR = DATA_DIR / "captures"
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "retinal_model.tflite"
INPUT_IMAGE_SIZE = (224, 224)        # (width, height) expected by the model
CONFIDENCE_THRESHOLD = 0.60          # below this, UI flags result as "uncertain"

# Order MUST exactly match the class order the model was trained with.
DISEASE_CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
]

CAMERA_RESOLUTION = (1640, 1232)     # Pi Camera Module still-capture resolution

# Create runtime directories the first time this module is imported.
for directory in (CAPTURE_DIR, UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)