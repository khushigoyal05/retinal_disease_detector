"""
dataset_prep/cdr/check_disc_localization.py

Manual visual sanity check for locate_disc(). Not an automated test —
we have no ground-truth disc coordinates yet, so there's nothing to
assert against. This script just draws what the detector found so a
human (you) can judge whether it's landing on the actual optic disc.

USAGE:
    python dataset_prep/cdr/check_disc_localization.py
"""

import sys
from pathlib import Path

import cv2

# So this script can be run directly without installing the package
sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.disc_localization import locate_disc, DiscLocation

INPUT_DIR = Path("data/uploads")          # your existing test fundus images
OUTPUT_DIR = Path("dataset_prep/cdr/_check_output")  # gitignored scratch folder


def annotate_and_save(image_path: Path, output_dir: Path) -> None:
    bgr_image = cv2.imread(str(image_path))
    if bgr_image is None:
        print(f"  SKIP (couldn't read): {image_path.name}")
        return

    try:
        disc: DiscLocation = locate_disc(bgr_image)
    except ValueError as e:
        print(f"  FAILED to locate disc in {image_path.name}: {e}")
        return

    annotated = bgr_image.copy()
    cv2.circle(annotated, (disc.center_x, disc.center_y), disc.radius, (0, 255, 0), thickness=4)
    cv2.drawMarker(annotated, (disc.center_x, disc.center_y), (0, 0, 255),
                    markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)

    output_path = output_dir / f"annotated_{image_path.name}"
    cv2.imwrite(str(output_path), annotated)

    print(f"  {image_path.name}: center=({disc.center_x},{disc.center_y}) "
          f"radius={disc.radius} confidence={disc.confidence}")


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Input dir not found: {INPUT_DIR.resolve()}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not image_paths:
        print(f"No images found in {INPUT_DIR.resolve()}")
        return

    print(f"Checking {len(image_paths)} image(s) from {INPUT_DIR}...\n")
    for image_path in image_paths:
        annotate_and_save(image_path, OUTPUT_DIR)

    print(f"\nDone. Open the images in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()