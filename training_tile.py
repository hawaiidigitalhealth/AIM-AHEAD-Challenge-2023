import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
import pytorch_lightning as pl
from openslide import OpenSlide

from torchvision.models import resnet50, ResNet50_Weights

from tqdm import tqdm
import h5py
from sklearn.model_selection import train_test_split
import ngsci

instance = os.getenv("NG_INSTANCE_ID")

# torch.cuda.init()
# assert torch.cuda.is_initialized()
# print(torch.cuda.get_device_properties(0))

brca_dir = Path().home() / 'datasets' / 'brca-psj-path' / "contest-phase-2"
image_dir = brca_dir / "basic-downsampling" / "v2-subsample-a"
table_dir = brca_dir / "csv-train"
ndpi_dir = brca_dir / 'ndpi'
clam_train_dir = brca_dir / 'clam-preprocessing-train'

masks_dir = clam_train_dir / 'masks'
patches_dir = clam_train_dir / 'patches'
stitches_dir = clam_train_dir / 'stitches'
features_h5_dir = clam_train_dir / 'resnet50-features'/ 'h5_files'
features_pt_dir = clam_train_dir / 'resnet50-features'/ 'pt_files'

logger_dir = Path().home() / "logs"

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
def substage_aware_labels(stage):
        if stage == "0":
            return torch.tensor([1, 0, 0, 0, 0])
        elif stage == "IA":
            return torch.tensor([0.2, 0.8, 0, 0, 0])
        elif stage == "IB":
            return torch.tensor([0, 0.8, 0.2, 0, 0])
        elif stage == "IIA":
            return torch.tensor([0, 0.2, 0.8, 0, 0])
        elif stage == "IIB":
            return torch.tensor([0, 0, 0.8, 0.2, 0])
        elif stage == "IIIA":
            return torch.tensor([0, 0, 0.2, 0.8, 0])
        elif stage == "IIIB":
            return torch.tensor([0, 0, 0, 1, 0])
        elif stage == "IIIC":
            return torch.tensor([0, 0, 0, 0.8, 0.2])
        elif stage == "IV":
            return torch.tensor([0, 0, 0, 0, 1])
        else:
            return np.nan
        
class DataAugmentation(nn.Module):
    def __init__(self):
        super().__init__()

        self.transforms = nn.Sequential(
            K.RandomErasing(), 
            K.RandomHorizontalFlip(),
            K.RandomMedianBlur()
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        return x_out


class BreastBiopsy(Dataset):
    def __init__(self, mapping_file, 
                 img_size, 
                 transform=None, 
                 target_transform=None):
        
        self.dataframe = pd.read_csv(mapping_file)
        self.img_size = (img_size, img_size)
        
        self.target_transform = target_transform
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        slide_id = self.dataframe.loc[idx, "slide_id"]
        coords = self.dataframe.loc[idx, "coords"][0]
        
        with OpenSlide(ndpi_dir / f'{slide_id}.ndpi') as slide:
            tile_img = slide.read_region(
                location=coords, 
                level=0, 
                size=(256,256)
            )
            
        #label = substage_aware_labels(self.dataframe.loc[idx, 'stage'])
        label = int(self.dataframe.loc[idx, 'stage'])
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label.to(torch.float32)
    
def get_dataloaders(batch_size, img_size, image_type, train_augs, test_augs=None):
    
    training_data = BreastBiopsy(mapping_file='train.csv', image_type=image_type,
                                 img_size=img_size, transform=train_augs)
    
    validation_data = BreastBiopsy(mapping_file='validation.csv', image_type=image_type,
                                 img_size=img_size, transform=test_augs)
    
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=45)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, num_workers=45)
    
    
    return train_loader, validation_loader

class Trainer(pl.LightningModule):
    def __init__(self, model, loss_fn, lr):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.transform = DataAugmentation()
        
    def on_after_batch_transfer(self, batch, dataloader_idx):
        X, y = batch
        if self.trainer.training:
            X = self.transform(X)
        return X, y
    
    def configure_callbacks(self):
        early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", 
                                                                       mode="min", 
                                                                       patience=15)
        early_stop_flg = pl.callbacks.early_stopping.EarlyStopping(monitor="collapse_flg", 
                                                                           mode="max", 
                                                                           patience=100,
                                                                          divergence_threshold=2)
        return [early_stop, early_stop_flg]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        X, y = batch

        # Compute prediction error
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        collapse_flg = torch.unique(pred).size(dim=0)
        self.log("collapse_flg", collapse_flg, sync_dist=True)
        self.log("training_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return {'loss' : loss}
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # training_step defines the train loop.
        X, y = batch

        # Compute prediction error
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        if dataloader_idx == 0:
            self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.99, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        
        return ({
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },},)
    
if __name__ == '__main__':
    install("kornia")
    import kornia.augmentation as K
    IM_SIZE = 1512
    BATCH_SIZE = 8

    train_augs = T.Compose([T.ToTensor(),
                            T.RandomRotation(degrees=(0, 360)),
                            T.RandomCrop(size=(IM_SIZE, IM_SIZE), pad_if_needed=True), 
                            T.Resize(256, antialias=True)])

    test_augs = T.Compose([T.ToTensor(),
                           T.CenterCrop(size=(IM_SIZE, IM_SIZE)), 
                           T.Resize(256, antialias=True)])

    train_loader, validation_loader = get_dataloaders(BATCH_SIZE, IM_SIZE, 'mask', train_augs, test_augs)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    weights_path = "resnet50-11ad3fa6.pth"
    backbone = resnet50(pretrained=False)
    backbone.load_state_dict(torch.load(weights_path))
    backbone.fc = nn.Linear(backbone.fc.in_features, 5)
    
    for i, (name, param) in enumerate(backbone.named_parameters()):
        if i < 115:
            param.requires_grad = False

    model = Trainer(model = backbone,
                   loss_fn=loss_fn, 
                   lr=0.001)
    
    trainer = pl.Trainer(accelerator="gpu", 
                         devices="auto",
                         max_epochs=50, 
                         default_root_dir="lightning_checkpoints/")

    trainer.fit(model, train_loader, [validation_loader])
    ngsci.stop_instance()