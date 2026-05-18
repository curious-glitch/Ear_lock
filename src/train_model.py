import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

# ---------------- Paths ----------------
ROOT = Path(r"Z:\Code\EarEdge3")
DATA_DIR = ROOT / "data_split"
MODEL_PATH = ROOT / "models" / "ear_model.pth"

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Transforms ----------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(0, translate=(0.05,0.05)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------- Dataset ----------------
train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=transform_train
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=transform_val
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

print("Classes:", train_dataset.classes)
print("Number of classes:", len(train_dataset.classes))

# ---------------- Model ----------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze early layers
for name, param in model.named_parameters():
    if "layer1" in name or "conv1" in name:
        param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))

model = model.to(device)
# ---------------- Training Setup ----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)
EPOCHS = 120
best_acc = 0

# ---------------- Training Loop ----------------
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    epoch_loss = running_loss / total

    # ---------------- Validation ----------------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:

        best_acc = val_acc

        MODEL_PATH.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)

        print("Model saved")

print("Training complete")
print("Best Accuracy:", best_acc)