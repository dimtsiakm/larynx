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

train_transforms_3d = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            
            SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
            RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=2),
            CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["image"], prob=1.0, holes=25, spatial_size=5, dropout_holes=True, max_spatial_size=32
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
                        keys=["image_2"], prob=1.0, holes=25, spatial_size=5, dropout_holes=True, max_spatial_size=32
                    ),
                    RandCoarseDropoutd(
                        keys=["image_2"], prob=1.0, holes=8, spatial_size=20, dropout_holes=False, max_spatial_size=64
                    ),
                ]
            ),
            RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
        ]
    )




check_ds = Dataset(data=train, transform=train_transforms_3d)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image = check_data["image"][0][0]
image_2 = check_data["image_2"][0][0]
gt_image = check_data["gt_image"][0][0]

print(f"image shape: {image.shape}")
print("first volume's shape: ", check_data["image"].shape)

imgplot = plt.imshow(image[:, :, 20])
pth = "reports/figures/temp" + f'/figure_1_1.png'
imgplot.get_figure().savefig(pth)
print(f'figure saved at: {pth}')

imgplot = plt.imshow(image_2[:, :, 20])
pth = "reports/figures/temp" + f'/figure_2_1.png'
imgplot.get_figure().savefig(pth)
print(f'figure saved at: {pth}')

imgplot = plt.imshow(gt_image[:, :, 20])
pth = "reports/figures/temp" + f'/figure_3_1.png'
imgplot.get_figure().savefig(pth)
print(f'figure saved at: {pth}')








# print("----------------------------------------")
# print("new tranformation...")
# # num_samples = 4

# patch_func, patch_transform, train_transforms = get_self_supervised_transforms()

# check_ds = Dataset(data=train, transform=train_transforms)
# # patch_ds = PatchDataset(
# #     check_ds,
# #     transform=patch_transform,
# #     patch_func=patch_func,
# #     samples_per_image=4,
# # )
# train_loader = DataLoader(
#     check_ds,
#     batch_size=3,
#     shuffle=True,  # this shuffles slices from different volumes
#     num_workers=2,
#     pin_memory=torch.cuda.is_available(),
# )
# check_data = first(train_loader)
# print("first patch's shape: ", check_data["image"].shape)

# for idx, d in enumerate(train_loader):
#     imgplot = plt.imshow(d["image"][0][0])
#     config = Config()
#     pth = "reports/figures/temp" + f'/figure_1_{idx}.png'
#     imgplot.get_figure().savefig(pth)
#     print(f'figure saved at: {pth}')
#     if idx > 6:
#         break


# # # Define Network ViT backbone & Loss & Optimizer
# # device = torch.device("cuda:0")
# # model = ViTAutoEnc(
# #     in_channels=1,
# #     img_size=(96, 96, 96),
# #     patch_size=(16, 16, 16),
# #     pos_embed="conv",
# #     hidden_size=768,
# #     mlp_dim=3072,
# # )

# # model = model.to(device)