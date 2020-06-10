from torchvision import transforms
import torch
import numpy as np

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def gram(x):
    (b, c, h, w) = x.size()
    feature = x.view(b, c, h*w)
    feature_t = feature.transpose(1, 2)
    gram = feature.bmm(feature_t) / (c * h * w)
    return gram


def train_transform(img_size):
    transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.RandomCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    return transform


def style_transform(img_size):
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    return transform


def denormalize(t):
    for c in range(3):
        t[:, c].mul_(std[c]).add_(mean[c])
    return t

