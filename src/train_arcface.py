import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from arcface_model import EarRecognitionModel

# ---------------- Paths ----------------
ROOT = Path(r"Z:\Code\EarEdge3")
DATA_DIR = ROOT / "data_split"
MODEL_PATH = ROOT / "models" / "arcface_model.pth"

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Transforms ----------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- Dataset ----------------
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform_train)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)

num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)

# ---------------- Model ----------------
model = EarRecognitionModel(num_classes).to(device)

# Freeze early layers
for name, param in model.backbone.named_parameters():
    if "layer1" in name or "conv1" in name:
        param.requires_grad = False

# ---------------- Training Setup ----------------
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

EPOCHS = 50
best_acc = 0

# ---------------- Training Loop ----------------
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

    for batch_idx, (images, labels) in enumerate(train_loader):

        # 🔥 Batch progress print
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)}")

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, labels)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # ---------------- Validation ----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)

            embeddings = model(images)  # NO labels here

            embeddings = torch.nn.functional.normalize(embeddings)
            centers = torch.nn.functional.normalize(model.arcface.weight)

            similarity = torch.matmul(embeddings, centers.T)
            _, preds = torch.max(similarity, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        MODEL_PATH.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ Model saved")

print("\nTraining complete")
print("Best Accuracy:", best_acc)