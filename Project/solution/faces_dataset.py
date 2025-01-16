"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        # real image case
        if 0 <= index < len(self.real_image_names):
            label = 0
            image_name = self.real_image_names[index]
            image_path = os.path.join(os.path.join(self.root_path, 'real'), image_name)
        # fake/synthetic image case
        elif index < self.__len__():
            label = 1
            index -= len(self.real_image_names)
            image_name = self.fake_image_names[index]
            image_path = os.path.join(os.path.join(self.root_path, 'fake'), image_name)
        else:
            raise ValueError(f'Index: {index} is out of bounds. '
                             f'The data set size is {self.__len__()}.')

        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)
        else:
            converter = transforms.PILToTensor()
            image = converter(image)

        return image, label

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.real_image_names) + len(self.fake_image_names)
