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

from torchvision.models import resnet50
from torch.utils.data import Sampler, WeightedRandomSampler

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
ndpi_dir = Path().home() / 'datasets' / 'brca-psj-path' / 'ndpi'
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
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.RandomMedianBlur()
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        return x_out


class BreastBiopsy(Dataset):
    def __init__(self, mapping_file, 
                 transform=None, 
                 target_transform=None):
        
        self.dataframe = pd.read_csv(mapping_file)
        
        self.target_transform = target_transform
        self.transform = transform
        self.demographic_column = "race"

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        slide_id = self.dataframe.loc[idx, "slide_id"]
        
        with h5py.File(patches_dir / f'{slide_id}.h5', 'r') as f:
            coords = f['coords'][:]
            
        random_index = np.random.choice(coords.shape[0])
        random_row = coords[random_index]
    
        with OpenSlide(ndpi_dir / f'{slide_id}.ndpi') as slide:
            tile_img = slide.read_region(
                location=random_row, 
                level=0, 
                size=(256,256)
            )
            
        #label = substage_aware_labels(self.dataframe.loc[idx, 'stage'])
        label = self.dataframe.loc[idx, 'stage_int']
        demo = self.dataframe.loc[idx, self.demographic_column]
        
        if self.transform:
            tile_img = self.transform(tile_img.convert('RGB'))
        if self.target_transform:
            label = self.target_transform(label)

        return tile_img, torch.tensor(label), demo
    
    
class DemographicBalancedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.unique_demographics, self.counts = np.unique(dataset.dataframe[dataset.demographic_column], return_counts=True)
        demographic_indices = {demographic: index for index, demographic in enumerate(self.unique_demographics)}
        # Calculate class weights
        class_weights = 1.0 / torch.Tensor(self.counts)
        demographics = dataset.dataframe[dataset.demographic_column].tolist()
        indices = [demographic_indices[demographic] for demographic in demographics]
        weights = class_weights[torch.tensor(indices).long()]

        # Create weighted random sampler
        self.sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return len(self.dataset)
    
    
def get_dataloaders(batch_size, train_augs, test_augs=None):
    
    training_data = BreastBiopsy(mapping_file='train.csv', transform=train_augs)
    training_sampler = DemographicBalancedSampler(training_data)
    
    validation_data = BreastBiopsy(mapping_file='validation.csv', transform=test_augs)
    validation_sampler = DemographicBalancedSampler(validation_data)
    
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, sampler=validation_sampler, num_workers=8)
    
    
    return train_loader, validation_loader

class Trainer(pl.LightningModule):
    def __init__(self, model, loss_fn, lr):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.transform = DataAugmentation()
        
    def on_after_batch_transfer(self, batch_transfer, dataloader_idx):
        X, y, d = batch_transfer
        if self.trainer.training:
            X = self.transform(X)
        return X, y
    
    def configure_callbacks(self):
        early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", 
                                                                       mode="min", 
                                                                       patience=25)
        early_stop_flg = pl.callbacks.early_stopping.EarlyStopping(monitor="collapse_flg", 
                                                                           mode="max", 
                                                                           patience=100,
                                                                          divergence_threshold=2)
        return [early_stop, early_stop_flg]

    def training_step(self, batch_train, batch_idx):
        # training_step defines the train loop.
        X, y = batch_train

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
    BATCH_SIZE = 16

    train_augs = T.Compose([T.ToTensor(),
                            T.RandomRotation(degrees=(0, 360))])

    test_augs = T.Compose([T.ToTensor()])

    train_loader, validation_loader = get_dataloaders(BATCH_SIZE, train_augs, test_augs)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    weights_path = "resnet50-11ad3fa6.pth"
    backbone = resnet50(pretrained=False)
    backbone.load_state_dict(torch.load(weights_path))
    backbone.fc = nn.Linear(backbone.fc.in_features, 5)
    
    for i, (name, param) in enumerate(backbone.named_parameters()):
        if i < 100:
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