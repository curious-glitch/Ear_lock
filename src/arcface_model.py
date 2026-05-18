import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------- ArcFace Layer ----------------
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.s = s
        self.m = m

    def forward(self, x, labels):
        # normalize features and weights
        x = F.normalize(x)
        W = F.normalize(self.weight)

        cosine = F.linear(x, W)
        theta = torch.acos(torch.clamp(cosine, -1.0, 1.0))
        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        output = cosine * (1 - one_hot) + target_logits * one_hot
        output *= self.s

        return output


# ---------------- Full Model ----------------
class EarRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(EarRecognitionModel, self).__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # remove original FC
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # embedding layer
        self.embedding = nn.Linear(in_features, 512)

        # ArcFace head
        self.arcface = ArcFace(512, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = self.embedding(features)

        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return logits
        else:
            return embeddings