"""Hold all models you wish to train."""
import torch
import torch.nn.functional as F

from torch import nn

from xcpetion import build_xception_backbone, Xception, disable_ssl_verification


class SimpleNet(nn.Module):
    """Simple Convolutional and Fully Connect network."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(7, 7))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(16, 24, kernel_size=(7, 7))
        self.fc1 = nn.Linear(24 * 26 * 26, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, image):
        """Compute a forward pass."""
        first_conv_features = self.pool(F.relu(self.conv1(image)))
        second_conv_features = self.pool(F.relu(self.conv2(
            first_conv_features)))
        third_conv_features = self.pool(F.relu(self.conv3(
            second_conv_features)))
        # flatten all dimensions except batch
        flattened_features = torch.flatten(third_conv_features, 1)
        fully_connected_first_out = F.relu(self.fc1(flattened_features))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output

class XceptionNewHead(nn.Module):
    def __init__(self):
        super(XceptionNewHead, self).__init__()
        self.fc1 = nn.Linear(2048, 1000)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, 256)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

def get_xception_based_model() -> nn.Module:
    """Return an Xception-Based network.

    (1) Build an Xception pre-trained backbone and hold it as `custom_network`.
    (2) Override `custom_network`'s fc attribute with the binary
    classification head stated in the exercise.
    """

    # uncomment only if the weights are not already in pytorch cache and the ssl certificate causes problems
    # disable_ssl_verification()

    xception_model_trained : Xception = build_xception_backbone(pretrained=True)

    xception_model_trained.fc = XceptionNewHead()
    return xception_model_trained
