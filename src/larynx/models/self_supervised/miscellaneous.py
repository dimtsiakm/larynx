import os
import json
import time
import torch
import matplotlib.pyplot as plt
from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset
from monai.config import print_config
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandRotate90d,
    Resized,
    ScaleIntensityd,
    SqueezeDimd,
)
from glob import glob
import shutil

import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, CacheDataset, PatchDataset, create_test_image_3d

from monai.visualize import matshow3d

from larynx.data.make_dataset_niftii import get_images
from larynx.models.transforms import get_self_supervised_transforms
from larynx.utils.config import Config

set_determinism(seed=123)

def get_dataloaders(batch_size=16, num_samples=8, num_of_patches_per_slice=4):
    print(f'Total batch size: {batch_size*num_of_patches_per_slice}')
    train, val = get_images(percentge=0.7)
    # train = train[:10]
    # val = val[:1]

    print(len(train), len(val), len(train)+len(val))
    config = Config()

    # volume-level transforms for both image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config.min_window_level,
                a_max=config.max_window_level,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            Resized(keys=["image"], spatial_size=[480, 480, -1]), # the size of the final image 96*5 = 480
            EnsureTyped(keys=["image"]),
        ]
    )
    # 3D dataset with preprocessing transforms
    volume_ds_train = CacheDataset(data=train, transform=train_transforms, num_workers=8)
    volume_ds_val = CacheDataset(data=val, transform=train_transforms, num_workers=8)

    
    patch_func = RandSpatialCropSamplesd(
        keys=["image"],
        roi_size=[-1, -1, 1],  # dynamic spatial_size for the first two dimensions
        num_samples=num_samples,
        random_size=False,
    )
    patch_transform = Compose(
        [
            SqueezeDimd(keys=["image"], dim=-1),  # squeeze the last dim
            
            ############################################################

            SpatialPadd(keys=["image"], spatial_size=(96, 96)),
            RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96), random_size=False, num_samples=num_of_patches_per_slice),
            CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                    ),
                    RandCoarseDropoutd(
                        keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                    ),
                ]
            ),
            RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=16),
            # Please note that that if image, image_2 are called via the same transform call because of the determinism
            # they will get augmented the exact same way which is not the required case here, hence two calls are made
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                    ),
                    RandCoarseDropoutd(
                        keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                    ),
                ]
            ),
            RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=16),

        ]
    )

    patch_ds_train = PatchDataset(
        volume_ds_train,
        transform=patch_transform,
        patch_func=patch_func,
        samples_per_image=num_samples,
    )

    patch_ds_val = PatchDataset(
        volume_ds_val,
        transform=patch_transform,
        patch_func=patch_func,
        samples_per_image=num_samples,
    )

    train_loader = DataLoader(
        patch_ds_train,
        batch_size=batch_size,
        shuffle=True,  # this shuffles slices from different volumes
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        patch_ds_val,
        batch_size=batch_size,
        shuffle=True,  # this shuffles slices from different volumes
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, (len(patch_ds_train))


def get_model():
    device = torch.device("cuda:0")
    model = ViTAutoEnc(
        in_channels=1,
        img_size=(96, 96),
        patch_size=(16, 16),
        pos_embed="conv",
        hidden_size=768,
        mlp_dim=3072,
        spatial_dims=2
    )
    model = model.to(device)
    return model, device