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
from larynx.models.self_supervised.miscellaneous import get_dataloaders, get_model
from larynx.utils.config import Config

set_determinism(seed=123)

def save_image(train_loader):
    counter = 0
    for batch in train_loader:
        print(f"batch size is: {batch['image'].shape}")
        for i in range(train_loader.batch_size):
            imgplot = plt.imshow(batch["image"][i, 0, :, :], cmap='gray')
            pth = "reports/figures/temp" + f'/figure_image_{counter}.png'
            imgplot.get_figure().savefig(pth)
            print(f'figure saved at: {pth}')
            counter += 1

def save_augmentations(train_loader):
    counter = 0
    for batch in train_loader:
        print(f"gt_image size is: {batch['gt_image'].shape}")
        print(f"image size is: {batch['image'].shape}")
        print(f"image_2 size is: {batch['image_2'].shape}")
        for i in range(train_loader.batch_size*2):
            gt_image = np.array(batch["gt_image"][i, 0, :, :])
            image = np.array(batch["image"][i, 0, :, :])
            image_2 = np.array(batch["image_2"][i, 0, :, :])

            # print(gt_image.shape, image.shape, image_2.shape)
            plt.subplot(1, 3, 1)
            plt.imshow(gt_image, cmap='gray')

            plt.subplot(1, 3, 2)
            plt.imshow(image, cmap='gray')

            plt.subplot(1, 3, 3)
            plt.imshow(image_2, cmap='gray')

            pth = "reports/figures/temp" + f'/figure_1_{counter}.png'
            plt.savefig(pth)
            counter += 1


train_loader, val_loader, patch_ds_train_len = get_dataloaders()

model, device = get_model()

# Define Hyper-paramters for training loop
max_epochs = 600
val_interval = 2
lr = 1e-4
epoch_loss_values = []
step_loss_values = []
epoch_cl_loss_values = []
epoch_recon_loss_values = []
val_loss_values = []
best_val_loss = 1000.0

recon_loss = L1Loss()
contrastive_loss = ContrastiveLoss(temperature=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    epoch_cl_loss = 0
    epoch_recon_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        start_time = time.time()

        inputs, inputs_2, gt_input = (
            batch_data["image"].to(device),
            batch_data["image_2"].to(device),
            batch_data["gt_image"].to(device),
        )
        optimizer.zero_grad()
        outputs_v1, hidden_v1 = model(inputs)
        outputs_v2, hidden_v2 = model(inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=3)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=3)

        r_loss = recon_loss(outputs_v1, gt_input)
        cl_loss = contrastive_loss(flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        total_loss = r_loss + cl_loss * r_loss

        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        step_loss_values.append(total_loss.item())

        # CL & Recon Loss Storage of Value
        epoch_cl_loss += cl_loss.item()
        epoch_recon_loss += r_loss.item()

        end_time = time.time()
        print(
            f"{step}/{patch_ds_train_len // train_loader.batch_size}, "
            f"train_loss: {total_loss.item():.4f}, "
            f"time taken: {end_time-start_time}s"
        )

    epoch_loss /= step
    epoch_cl_loss /= step
    epoch_recon_loss /= step

    epoch_loss_values.append(epoch_loss)
    epoch_cl_loss_values.append(epoch_cl_loss)
    epoch_recon_loss_values.append(epoch_recon_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if epoch % val_interval == 0:
        print("Entering Validation for epoch: {}".format(epoch + 1))
        total_val_loss = 0
        val_step = 0
        model.eval()
        for val_batch in val_loader:
            val_step += 1
            start_time = time.time()
            inputs, gt_input = (
                val_batch["image"].to(device),
                val_batch["gt_image"].to(device),
            )
            # print("Input shape: {}".format(inputs.shape))
            outputs, outputs_v2 = model(inputs)
            val_loss = recon_loss(outputs, gt_input)
            total_val_loss += val_loss.item()
            end_time = time.time()

        total_val_loss /= val_step
        val_loss_values.append(total_val_loss)
        print(f"epoch {epoch + 1} Validation avg loss: {total_val_loss:.4f}, " f"time taken: {end_time-start_time}s")

        if total_val_loss < best_val_loss:
            print(f"Saving new model based on validation loss {total_val_loss:.4f}")
            best_val_loss = total_val_loss
            checkpoint = {"max_epochs": max_epochs, "epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, os.path.join("models/ViT", "best_model.pt"))

        plt.figure(1, figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.plot(epoch_loss_values)
        plt.grid()
        plt.title("Training Loss")

        plt.subplot(2, 2, 2)
        plt.plot(val_loss_values)
        plt.grid()
        plt.title("Validation Loss")

        plt.subplot(2, 2, 3)
        plt.plot(epoch_cl_loss_values)
        plt.grid()
        plt.title("Training Contrastive Loss")

        plt.subplot(2, 2, 4)
        plt.plot(epoch_recon_loss_values)
        plt.grid()
        plt.title("Training L1 Loss")

        plt.savefig(os.path.join("reports/figures/ViT", f"loss_plots_{max_epochs}.png"))
        plt.close(1)

print("Done")
