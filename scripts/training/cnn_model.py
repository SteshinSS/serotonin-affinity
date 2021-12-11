import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torch.utils.data import Dataset, DataLoader


class CNN_Model(pl.LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        self.lr = params['lr']
        self.l2_lambda = params['l2_lambda']
        self.n_channels = params['n_channels']
        self.weight_of_ones = params['weight_of_ones']
        self.dropout = params['dropout']
        self.net = nn.Sequential(
            nn.Conv3d(self.n_channels, 8, 3),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            nn.Conv3d(16, 16, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(40),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

    
    def forward(self, X):
        return self.net(X)
    
    def get_weights(self, y_true):
        weights = torch.ones_like(y_true)
        weights[y_true == 1] *= self.weight_of_ones
        return weights

    def training_step(self, batch, batch_n):
        X, y_true = batch
        y_pred = self(X)
        weights = self.get_weights(y_true)
        loss = F.binary_cross_entropy_with_logits(y_pred.squeeze(), y_true, weight=weights)
        return loss
    
    def validation_step(self, batch, batch_n):
        X, y_true = batch
        y_pred = self(X)
        weights = self.get_weights(y_true)
        loss = F.binary_cross_entropy_with_logits(y_pred.squeeze(), y_true, weight=weights)
        self.log("val_loss", loss, prog_bar=True)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.l2_lambda
        )
        return optimizer


class VoxelDataset(Dataset):
    def __init__(self, voxels: torch.Tensor, labels: torch.Tensor):
        super().__init__()
        self.voxels = voxels
        self.labels = labels
    
    def __len__(self):
        return self.voxels.shape[0]
    
    def __getitem__(self, idx):
        # TODO: Add rotation augmentation
        return self.voxels[idx], self.labels[idx]
    
    def to(self, device):
        self.voxels = self.voxels.to(device)
        self.labels = self.labels.to(device)


def construct_dataloader(dataset, batch_size: int, shuffle: bool):
    voxels = torch.Tensor(dataset['X'])
    voxels = rearrange(voxels, 'idx x y z channel -> idx channel x y z')
    labels = torch.Tensor(dataset['y'])
    dataset = VoxelDataset(voxels, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=1
    )
    return dataloader
