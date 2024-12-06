# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/nanoCLIP
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class AlbumDataset(Dataset):
    def __init__(self, root_dir='./gallery/photos', transform=None):
        """
        This class is a simple dataset for loading ALL images from a directory and its subdirectories.
        Formats supported: .jpg, .jpeg, .png, .bmp, .tiff
        Args:
            root_dir (str or Path): Path to the root directory containing images (e.g. gallery/).
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise ValueError(f"Provided path {root_dir} does not exist.")
        
        # Gather all image paths
        self.imgs = [p for p in self.root_dir.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
        
        # sort the images by name
        self.imgs = sorted(self.imgs)
        
        if not self.imgs:
            raise ValueError(f"No images found under {root_dir}.")
        
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel RGB
        
        if self.transform:
            image = self.transform(image)
        
        return image, str(image_path)  # Optionally return the path with the image
