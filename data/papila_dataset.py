import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def apply_clahe(pil_img, clip=1.6, grid=8):
    img_cv = np.array(pil_img)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_img)

class PapilaDataset(Dataset):
    """
    CSV must contain column: Image_Path
    Optionally you can include other columns; they are ignored here (unconditional diffusion).
    """
    def __init__(self, csv_path, transform=None, use_clahe=True):
        assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
        self.df = pd.read_csv(csv_path)
        assert "Image_Path" in self.df.columns, "CSV must contain 'Image_Path' column"
        self.transform = transform
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.iloc[idx]["Image_Path"]
        img = Image.open(p).convert("RGB")
        if self.use_clahe:
            img = apply_clahe(img)
        if self.transform:
            img = self.transform(img)
        return img

default_transforms = T.Compose([
    T.Resize((128, 128)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])


