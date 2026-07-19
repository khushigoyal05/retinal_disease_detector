"""Checks that every G1020 image has a matching mask."""
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw" / "archive" / "G1020"

def find_files(folder_name, extensions):
    folder = RAW_DIR / folder_name
    if not folder.exists():
        print(f"NOT FOUND: {folder}")
        return {}
    files = {}
    for ext in extensions:
        for f in folder.glob(f"*{ext}"):
            files[f.stem] = f
    return files

def main():
    print(f"Scanning: {RAW_DIR}\n")

    images = find_files("Images", [".jpg", ".png", ".jpeg"])
    masks = find_files("Masks", [".png", ".jpg"])

    print(f"Images found: {len(images)}")
    print(f"Masks found: {len(masks)}")

    missing_masks = set(images) - set(masks)
    missing_images = set(masks) - set(images)

    print(f"\nImages WITHOUT a mask: {len(missing_masks)}")
    if missing_masks:
        print(list(missing_masks)[:5], "...")

    print(f"Masks WITHOUT an image: {len(missing_images)}")
    if missing_images:
        print(list(missing_images)[:5], "...")

if __name__ == "__main__":
    main()