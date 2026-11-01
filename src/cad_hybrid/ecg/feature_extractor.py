
import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cpu")
PRETRAINED = True


def create_feature_extractor():
    """
    Create the same ImageNet-pretrained ResNet18
    feature extractor used in the original ECG project.
    """

    weights = (
        models.ResNet18_Weights.DEFAULT
        if PRETRAINED
        else None
    )

    model = models.resnet18(weights=weights)

    feature_extractor = nn.Sequential(
        *list(model.children())[:-1]
    )

    feature_extractor.eval()

    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    feature_extractor = feature_extractor.to(DEVICE)

    return feature_extractor


if __name__ == "__main__":
    model = create_feature_extractor()

    sample_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        features = model(sample_input)

    print("ResNet18 feature extractor created successfully.")
    print("Input shape:", sample_input.shape)
    print("Feature shape:", features.shape)
