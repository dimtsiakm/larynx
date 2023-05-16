import monai
import torch
import cv2

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

def segment_anything():
    config = Config()
    img = cv2.imread(config.data_processed_path+'304/00010001_itk.png')

    sam = sam_model_registry["vit_h"](checkpoint=config.models_path + 'sam/sam_vit_h_4b8939.pth')
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)

    
    return

if __name__ == '__main__':
    # inference_save_img()
    segment_anything()