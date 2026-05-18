import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path
from arcface_model import EarRecognitionModel

# ---------------- Paths ----------------
ROOT = Path(r"Z:\Code\EarEdge3")
DATA_DIR = ROOT / "enroll_data"
MODEL_PATH = ROOT / "models" / "arcface_model.pth"
DB_PATH = ROOT / "models" / "embeddings.npy"

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Checks ----------------
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Enrollment folder not found: {DATA_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- Load Model ----------------
print("\nLoading model...")

model = EarRecognitionModel(num_classes=264)

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)

# Remove ArcFace classification layer for inference / enrollment
state_dict.pop("arcface.weight", None)

model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

print("Model loaded successfully\n")

# ---------------- Enrollment ----------------
database = {}

print("Starting enrollment...\n")

with torch.no_grad():
    for person_folder in sorted(DATA_DIR.iterdir()):

        if not person_folder.is_dir():
            continue

        print(f"Processing: {person_folder.name}")

        embeddings_list = []

        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            image_files.extend(person_folder.glob(ext))

        image_files = sorted(image_files)

        if len(image_files) == 0:
            print(f"No image files found for {person_folder.name}\n")
            continue

        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)

                emb = model(img)
                emb = torch.nn.functional.normalize(emb, dim=1)

                embeddings_list.append(emb.cpu().numpy())

            except Exception as e:
                print(f"Skipping {img_path.name} ({e})")
                continue

        if len(embeddings_list) > 0:
            embeddings_array = np.vstack(embeddings_list).astype(np.float32)

            database[person_folder.name] = embeddings_array

            print(f"{person_folder.name} enrolled")
            print(f"Images: {len(embeddings_list)}")
            print(f"Shape: {embeddings_array.shape}\n")

        else:
            print(f"No valid images for {person_folder.name}\n")

# ---------------- Save Database ----------------
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
np.save(DB_PATH, database)

print("Enrollment complete!")
print("Saved to:", DB_PATH)
print("People in database:", list(database.keys()))