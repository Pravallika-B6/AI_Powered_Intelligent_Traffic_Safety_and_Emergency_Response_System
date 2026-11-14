"""
PyTorch Dataset and DataLoader module for Accident Detection dataset (data1).
Includes domain-specific data augmentations preserving physical crash semantics.
"""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

class AccidentDataset(Dataset):
    """
    Dataset loader for binary accident classification.
    Supports returning both transformed tensors and raw file paths.
    """
    def __init__(self, split_dir, transform=None, return_path=False):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.return_path = return_path
        self.samples = []
        
        # Load samples
        for label_idx, class_name in enumerate(config.CLASS_NAMES):
            class_folder = self.split_dir / class_name
            # Fallback for underscore format if applicable
            if not class_folder.exists():
                alt_folder = self.split_dir / class_name.replace(" ", "_")
                if alt_folder.exists():
                    class_folder = alt_folder

            if class_folder.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_file in class_folder.glob(ext):
                        self.samples.append((img_file, label_idx))
                        
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            
        if self.transform:
            tensor_img = self.transform(image)
        else:
            tensor_img = transforms.ToTensor()(image)
            
        if self.return_path:
            return tensor_img, label, str(img_path)
        return tensor_img, label

def get_transforms():
    """
    Domain-specific augmentations for CCTV/dashcam traffic frames.
    Note: NO vertical flips to prevent inverting vehicle crash orientations.
    """
    train_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def get_dataloaders(batch_size=config.BATCH_SIZE, return_path=False):
    """Factory creating Train, Val, and Test DataLoaders."""
    train_tf, eval_tf = get_transforms()
    
    train_ds = AccidentDataset(config.TRAIN_DIR, transform=train_tf, return_path=return_path)
    val_ds = AccidentDataset(config.VAL_DIR, transform=eval_tf, return_path=return_path)
    test_ds = AccidentDataset(config.TEST_DIR, transform=eval_tf, return_path=return_path)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader
