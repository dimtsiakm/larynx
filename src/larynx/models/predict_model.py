import monai
import torch
import cv2
import pickle
import numpy as np

from monai.data import create_test_image_2d, list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    SaveImage,
    ScaleIntensityd,
)
from monai.visualize import blend_images
import matplotlib.pyplot as plt

from larynx.data.make_dataset import get_dataloaders
from larynx.utils.config import Config

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def inference_save_img():
    train_dataloader, val_dataloader = get_dataloaders(first_n=5)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    saver = SaveImage(output_dir="./output", output_ext=".png", output_postfix="seg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64),
        strides=(2, 2),
    ).to(device)

    model.eval()
    with torch.no_grad():
        for val_data in val_dataloader:
            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (96, 96)
            sw_batch_size = 1
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            print('got segmentations')
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            
            from matplotlib import pyplot as plt
            for val_output in val_outputs:
                val_output = torch.squeeze(val_output)
                val_images = torch.squeeze(val_images)

                image, label = (val_images, val_output[1])
                image = image.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                print(f"image shape: {image.shape}, label shape: {label.shape}")
                plt.figure("check", (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("image")
                plt.imshow(image, cmap="gray")
                plt.subplot(1, 2, 2)
                plt.title("label")
                plt.imshow(label)
                plt.savefig('figure_1.png')
                exit()

def save_mask(filename, results):
    with open(filename, "wb") as fp:   #Pickling
            pickle.dump(results, fp)

def load_mask(filename):
    with open(filename, "rb") as fp:   # Unpickling
            b = pickle.load(fp)
            return b

def segment_anything(use_model=False):

    config = Config()
    img = cv2.imread(config.data_processed_path+'304/00010001_itk.png', cv2.IMREAD_GRAYSCALE)
    filename = config.join_data_path_with('interim/') + "masks.pickle"

    if use_model:
        sam = sam_model_registry["vit_h"](checkpoint=config.models_path + 'sam/sam_vit_h_4b8939.pth')
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(img)
        save_mask(filename=filename, results=masks)
        print("Exit program. The mask is saved completelly.")
        return

    masks = load_mask(filename=filename)
    num_of_masks = len(masks)
    
    for i, mask in enumerate(masks):
        mask_image = mask['segmentation'].astype(int)
        data_masked = np.ma.masked_where(mask_image == 0, mask_image)
        plt.figure("check", (24, 12))
        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(img, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title("label")
        plt.imshow(mask_image)
        plt.subplot(1, 3, 3)
        plt.title("ret")
        plt.imshow(img, cmap='gray')
        plt.imshow(data_masked, 'jet', interpolation='none', alpha=0.7)
        plt.savefig('figure_'+str(i)+'.png')



    return

if __name__ == '__main__':
    # inference_save_img()
    segment_anything()