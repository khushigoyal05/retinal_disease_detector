# Retinal Disease Detection System

An offline, touchscreen retinal disease screening device built on a Raspberry Pi 4. The device captures or accepts a fundus (retina) photo, runs on-device AI inference, and displays a diagnosis with confidence score, clinical metrics (CDR, vessel density), and a generated PDF report — all without needing internet access.

Built as a Computer Engineering capstone project.

---

## What It Does

1. **Capture or upload** a retinal image using a custom Pi Camera + 20D condensing lens rig, or upload an existing fundus photo.
2. **Preview** the image before running analysis.
3. **Run AI inference** on-device using TensorFlow Lite:
   - Disease classification (5 classes)
   - Cup-to-disc ratio (CDR) via segmentation
   - Vessel density via segmentation
4. **View results**: original image, AI-annotated image (vessel overlay + CDR circle), diagnosis, confidence, and clinical metrics.
5. **Generate a PDF report** summarizing the findings.

All processing happens locally on the Raspberry Pi — no cloud dependency.

---

## Hardware

| Component | Details |
|---|---|
| Compute | Raspberry Pi 4B |
| Camera | Pi Camera Module (v2 / HQ / v3 — exact model TBD) |
| Optics | 20D condensing lens, custom 3D-printed mount, manual focus (no autofocus motor) |
| Display | 3.5" touchscreen, 320×480, portrait orientation |
| Enclosure | Custom 3D-printed housing for Pi, camera, and lens (in revision) |

---

## Screens

- **Home** — entry point, choose capture or upload
- **Camera** — live preview, capture (burst-3-frames, auto-picks sharpest)
- **Preview** — review captured/uploaded image before analysis
- **Results** — diagnosis, confidence, CDR, vessel density, image quality score, original + annotated image, save/export

UI is built with PySide6, fixed at 320×480 portrait, hand-tuned per widget (not responsive/resizable).

---

## AI Models

Three separate TensorFlow Lite models, all trained on a laptop and run via on-device inference on the Pi.

| Model | Purpose | Architecture | Status |
|---|---|---|---|
| `retinal_model.tflite` | Disease classification (5-class) | MobileNetV2-based classifier | Working, accuracy pass pending (55.5% val accuracy) |
| `cdr_model.tflite` | Optic cup segmentation → CDR | MobileNetV2 encoder + U-Net decoder | Working, hybrid with classical disc localization |
| `vessel_model.tflite` | Vessel segmentation → vessel density | MobileNetV2 encoder + U-Net decoder | Working, trained on DRIVE dataset |

**Disease classes** (fixed index order): `0=Normal, 1=Diabetic Retinopathy, 2=Glaucoma, 3=Cataract, 4=AMD`

**Common model spec:**
- Input: `[1, 224, 224, 3]` float32, normalized to `[-1, 1]` via `(pixel / 127.5) - 1.0`
- Output: float32, no quantization
- CDR uses a hybrid approach: classical disc localization + trained cup segmentation model
- Vessel density replaces earlier OpenCV heuristics with a trained segmentation model

> **Not included:** Lesion area / haemorrhage count. This was evaluated (IDRiD dataset) and deliberately not built — the dataset was too small (~54 usable images) to justify a model over classical heuristics, so it was removed cleanly from the app and PDF report rather than shipped as an unreliable estimate.

---

## Tech Stack

- **UI:** PySide6 (Qt for Python)
- **Training:** TensorFlow / Keras (Windows dev machine)
- **Inference:** TensorFlow Lite — `tflite-runtime` on the Pi, full TensorFlow on Windows (automatic fallback)
- **Camera capture:** `picamera2` (Pi OS only)
- **Reports:** Generated PDF via `report_generator.py`

---

## Project Structure

```
retinal_disease_detector/
├── main.py                          # App entry point
├── config.py                        # App-wide configuration
├── core/                            # Interfaces, data models, pipeline orchestration
├── services/
│   ├── camera_service.py            # Pi camera capture (burst + sharpest-frame pick)
│   ├── inference_service.py         # Disease classifier (TFLite)
│   ├── cup_segmentation_engine.py   # Cup segmentation (CDR)
│   ├── cdr_service.py               # CDR computation
│   └── vessel_segmentation_engine.py # Vessel segmentation
├── ui/
│   ├── main_window.py               # Fixed 320x480 portrait window
│   ├── home_screen.py
│   ├── camera_screen.py
│   ├── preview_screen.py
│   ├── results_screen.py
│   └── inference_worker.py
├── models/                          # .tflite model files (tracked in git)
├── utils/
│   ├── image_utils.py
│   ├── report_generator.py          # PDF report generation
│   └── disc_localization.py         # Classical disc detection
├── dataset_prep/                    # Dataset preprocessing (training-side)
├── training/                        # Model training scripts (training-side)
└── tests/                           # Pytest suite
```

---

## Getting Started

### Development (Windows)

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Deployment (Raspberry Pi OS, 64-bit)

Virtual environments aren't portable across platforms — create a fresh one on the Pi:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

`requirements.txt` uses PEP 508 environment markers so Pi-only packages (like `picamera2`) are skipped on Windows and vice versa.

### Running Tests

```bash
pytest
```

---

## Known Limitations

- **Disease classifier accuracy** is currently 55.5% (val), with weak performance on Glaucoma (precision 0.26) and AMD (F1 0.43). An accuracy-focused pass is planned — the current priority was getting the full pipeline working end-to-end first.
- **CDR values** can run higher than typical clinical range (e.g. 0.81 vs. expected ~0.3–0.6) on some images and haven't been validated against clinical ground truth yet.
- **Vessel segmentation model** was trained for 20 epochs and loss hadn't fully plateaued — a longer training run is planned.
- **CDR is unavailable ("N/A")** on ~51% of images where disc detection fails — expected behavior, not a bug.
- **Camera quality sliders** (brightness/contrast/saturation/sharpness) were intentionally removed from the UI; `camera_service.py` has not yet been confirmed updated to match.
- **UI has only been tested on Windows**, not yet validated on the actual Pi touchscreen hardware.
- **Lesion detection is not implemented** (see above) — can be revisited later using the IDRiD segmentation subset if needed.

---

## Disclaimer

This is a capstone/research project and is **not a certified medical device**. It is not intended for clinical diagnosis or to replace professional ophthalmological evaluation.

---

## License

Private repository — all rights reserved.