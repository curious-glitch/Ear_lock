import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path
from arcface_model import EarRecognitionModel

# ---------------- Paths ----------------
ROOT = Path(r"Z:\Code\EarEdge3")
MODEL_PATH = ROOT / "models" / "arcface_model.pth"
DB_PATH = ROOT / "models" / "embeddings.npy"

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Constants ----------------
SIMILARITY_THRESHOLD = 0.62
MARGIN_THRESHOLD = 0.04
TOP_K = 3

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- Load Database ----------------
database = np.load(DB_PATH, allow_pickle=True).item()

print("\nLoaded database:")
for name, db_embs in database.items():
    print(f"{name} -> shape: {db_embs.shape}")

# ---------------- Load Model ----------------
model = EarRecognitionModel(num_classes=264)

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)

# Remove ArcFace classification layer for inference
state_dict.pop("arcface.weight", None)

model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

# ---------------- Recognition Function ----------------
def recognize(image_path,
              similarity_threshold=SIMILARITY_THRESHOLD,
              margin_threshold=MARGIN_THRESHOLD,
              top_k=TOP_K):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(image)
        emb = torch.nn.functional.normalize(emb, dim=1)
        emb = emb.cpu().numpy()

    results = []

    print("\nSimilarity scores:")
    for name, db_embs in database.items():
        scores = np.dot(emb, db_embs.T)[0]   # shape: (N,)

        k = min(top_k, len(scores))
        top_scores = np.sort(scores)[-k:]
        score = float(np.mean(top_scores))

        results.append((name, score))
        print(f"{name} -> {score:.4f}")

    results.sort(key=lambda x: x[1], reverse=True)

    best_match, best_score = results[0]

    if len(results) > 1:
        second_match, second_score = results[1]
    else:
        second_match, second_score = "None", -1.0

    print("\nTop matches:")
    print(f"1. {best_match} -> {best_score:.4f}")
    print(f"2. {second_match} -> {second_score:.4f}")
    print(f"Margin -> {best_score - second_score:.4f}")

    # Decision logic
    if best_score < similarity_threshold:
        return "Unknown", best_score, second_match, second_score, results

    if (best_score - second_score) < margin_threshold:
        return "Unknown", best_score, second_match, second_score, results

    return best_match, best_score, second_match, second_score, results

# ---------------- Test ----------------
if __name__ == "__main__":
    test_image = ROOT / "test.jpg"

    if not test_image.exists():
        print(f"Test image not found: {test_image}")
    else:
        name, score, second_name, second_score, all_results = recognize(test_image)

        print("\nFinal Result:")
        print(f"Predicted: {name}")
        print(f"Best score: {score:.4f}")
        print(f"Second best: {second_name} ({second_score:.4f})")