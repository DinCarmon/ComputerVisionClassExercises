"""Define your architecture here."""
import torch
from torch import nn
from torchvision import models


def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    chosen_model_name = "mobilenetv2"

    if chosen_model_name == "mobilenetv2":
        # MobileNetV2 Pretrained
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.last_channel, 2)  # Modify output layer

    elif chosen_model_name == "resnet18":
        # ResNet-18 Pretrained
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)  # Modify output layer

    elif chosen_model_name == "squeezenet":
        # SqueezeNet Pretrained
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1))  # Modify output layer
        model.num_classes = 2

    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model
