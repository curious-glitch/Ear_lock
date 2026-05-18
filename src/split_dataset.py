from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

# -------- Project Paths --------
ROOT = Path(r"Z:\Code\EarEdge3")

data_raw = ROOT / "data_raw"
train_dir = ROOT / "data_split" / "train"
val_dir = ROOT / "data_split" / "val"

# Create folders if they don't exist
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# -------- Dataset Split --------
for person in data_raw.iterdir():

    if not person.is_dir():
        continue

    images = list(person.glob("*"))

    if len(images) == 0:
        continue

    train_imgs, val_imgs = train_test_split(
        images,
        test_size=0.2,
        random_state=42
    )

    # Create class folders
    (train_dir / person.name).mkdir(parents=True, exist_ok=True)
    (val_dir / person.name).mkdir(parents=True, exist_ok=True)

    # Copy training images
    for img in train_imgs:
        shutil.copy(img, train_dir / person.name / img.name)

    # Copy validation images
    for img in val_imgs:
        shutil.copy(img, val_dir / person.name / img.name)

print("Dataset split complete")