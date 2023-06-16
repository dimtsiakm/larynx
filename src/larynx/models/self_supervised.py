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
from monai.data import DataLoader, PatchDataset, create_test_image_3d

from monai.visualize import matshow3d

from larynx.data.make_dataset_niftii import get_images
from larynx.models.transforms import get_self_supervised_transforms
from larynx.utils.config import Config

set_determinism(seed=123)


train, val = get_images(percentge=0.8)
train = train[:5]

# volume-level transforms for both image and segmentation
train_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Spacingd(keys=["image"], pixdim=(2.0, 2.0, 1.0), mode=("bilinear")),
        Resized(keys=["image"], spatial_size=[480, 480, -1]), # the size of the final image 96*5 = 480
        EnsureTyped(keys=["image"]),
        
        #     CropForegroundd(keys=["image"], source_key="image")
    ]
)
# 3D dataset with preprocessing transforms
volume_ds = Dataset(data=train, transform=train_transforms)
# use batch_size=1 to check the volumes because the input volumes have different shapes
# check_loader = DataLoader(volume_ds, batch_size=1)
# check_data = first(check_loader)
# print("first volume's shape: ", check_data["image"].shape)
# exit()

num_samples = 2
patch_func = RandSpatialCropSamplesd(
    keys=["image"],
    roi_size=[-1, -1, 1],  # dynamic spatial_size for the first two dimensions
    num_samples=num_samples,
    random_size=False,
)
patch_transform = Compose(
    [
        SqueezeDimd(keys=["image"], dim=-1),  # squeeze the last dim
        # Resized(keys=["image"], spatial_size=[480, 480]),
        # Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
        # Resized(keys=["image"], spatial_size=[96, 96]),
        # to use crop/pad instead of resize:
        # ResizeWithPadOrCropd(keys=["img", "seg"], spatial_size=[48, 48], mode="replicate"),
    ]
)

patch_ds = PatchDataset(
    volume_ds,
    transform=patch_transform,
    patch_func=patch_func,
    samples_per_image=num_samples,
)

batch_size = 2
train_loader = DataLoader(
    patch_ds,
    batch_size=batch_size,  # must be divided by images per slice, let's say: if 12 slices per image, then it should be:
    # 2, 3, 4, 6 and 12. 
    shuffle=False,  # this shuffles slices from different volumes
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
check_data = first(train_loader)
print(f"check_data shape: {check_data['image'].shape}")


# counter = 0
# for deita in train_loader:
#     print(f"deita size is: {deita['image'].shape}")
#     for i in range(num_samples):
#         imgplot = plt.imshow(deita["image"][i, 0, :, :])
#         pth = "reports/figures/temp" + f'/figure_1_{counter}.png'
#         imgplot.get_figure().savefig(pth)
#         print(f'figure saved at: {pth}')
#         counter += 1
# exit()

train_transforms_2d = Compose(
    [
        # EnsureChannelFirstd(keys=["image"]),
        SpatialPadd(keys=["image"], spatial_size=(96, 96)),
        RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96), random_size=False, num_samples=2),
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image"], prob=1.0, holes=10, spatial_size=5, dropout_holes=True, max_spatial_size=32
                ),
                RandCoarseDropoutd(
                    keys=["image"], prob=1.0, holes=8, spatial_size=20, dropout_holes=False, max_spatial_size=64
                ),
            ]
        ),
        RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
        # Please note that that if image, image_2 are called via the same transform call because of the determinism
        # they will get augmented the exact same way which is not the required case here, hence two calls are made
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image_2"], prob=1.0, holes=10, spatial_size=5, dropout_holes=True, max_spatial_size=32
                ),
                RandCoarseDropoutd(
                    keys=["image_2"], prob=1.0, holes=8, spatial_size=20, dropout_holes=False, max_spatial_size=64
                ),
            ]
        ),
        RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
    ]
)

deita = train_transforms_2d(check_data)
for i in range(batch_size):
    imgplot = plt.imshow(deita["image"][i, 0, :, :])
    pth = "reports/figures/temp" + f'/figure_1_{i}.png'
    imgplot.get_figure().savefig(pth)
    print(f'figure saved at: {pth}')

    imgplot = plt.imshow(deita["image_2"][i, 0, :, :])
    pth = "reports/figures/temp" + f'/figure_2_{i}.png'
    imgplot.get_figure().savefig(pth)
    print(f'figure saved at: {pth}')

    imgplot = plt.imshow(deita["gt_image"][i, 0, :, :])
    pth = "reports/figures/temp" + f'/figure_3_{i}.png'
    imgplot.get_figure().savefig(pth)
    print(f'figure saved at: {pth}')



exit()

# imgplot = plt.imshow(image_2[:, :, 20])
# pth = "reports/figures/temp" + f'/figure_2_1.png'
# imgplot.get_figure().savefig(pth)
# print(f'figure saved at: {pth}')

# imgplot = plt.imshow(gt_image[:, :, 20])
# pth = "reports/figures/temp" + f'/figure_3_1.png'
# imgplot.get_figure().savefig(pth)
# print(f'figure saved at: {pth}')
