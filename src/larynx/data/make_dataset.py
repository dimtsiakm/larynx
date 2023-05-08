import glob
import os
import sys
import numpy as np
import logging

import SimpleITK as sitk  # noqa: N813
from monai.transforms import SaveImage
from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import first, set_determinism

from larynx.data.transforms import get_transforms, transform_png
from larynx.utils.config import Config

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def get_images_from_folder(project_path=None, folder_name='304'):
    """ 
        Gets the data paths from data/raw/ + folder_name
    """
    config = Config()
    if project_path is None:
        project_path = config.project_path

    dir = os.path.join(config.data_raw_path, folder_name)
    train_images = sorted(glob.glob(os.path.join(dir, "*")))

    logging.warning('GET IMAGES FROM: ' + dir + ', len: ' + str(len(train_images)))

    return train_images


def save_image(image, folder_name):
    """Save image to data/processed/ + folder_name.
    """
    pth = os.path.join(Config().data_processed_path, folder_name)
    saver = SaveImage(
        output_dir=pth,
        output_ext=".png",
        output_postfix="itk",
        output_dtype=np.uint8,
        resample=False,
        writer="ITKWriter",
        separate_folder=False
    )
    # saver.set_options(data_kwargs={"spatial_ndim": 2})
    saver(image)


def save_images(img_list: list, folder_name: str):
    for img in img_list:
        save_image(img, folder_name=folder_name)


def get_dataloaders(first_n=None):
    """Returns a TRAIN dataloader"""
    train_images = get_images_from_folder(folder_name='304')
    if not train_images:
        logging.warning('Watch out! There are no data in the specific folder')
        exit()
    if first_n is None:
        first_n = -int(0.2*len(train_images))  # 20% split ratio Train/Val
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_images)]
    train_files, val_files =  data_dicts[:first_n], data_dicts[first_n:]

    train_transforms, val_transforms = get_transforms()
    set_determinism(seed=0)

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    return train_loader


def export_png_from_dcm(folder_name, project_path=None, first_n=None):
    images = get_images_from_folder(folder_name=folder_name)
    transform = transform_png()

    counter = len(images)
    if first_n is not None:
        counter = first_n

    for im in images:
        imag = transform(im)
        save_image(imag, folder_name=folder_name)
        counter -= 1
        if counter <= 0:
            return
    return


if __name__ == '__main__':
    
    """get dataloader, get the first batch and then, save the list to the path"""
    # train_loader = get_dataloaders(first_n=2)
    # check_data = first(train_loader)
    # print(check_data["image"].shape)
    # print(type(check_data["image"]))
    # save_images(check_data["image"], folder_name='mico')

    """get files from path/folder name and then, save them under a specific configuration"""
    # export_png_from_dcm(folder_name='304', first_n=2)