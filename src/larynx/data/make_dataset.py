import glob
import os
import sys
import SimpleITK as sitk  # noqa: N813
import numpy as np
import logging

# import itk
from monai.transforms import SaveImage
from monai.visualize import matshow3d
from monai.data import PILReader
from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import first, set_determinism# import itk
from monai.transforms import SaveImage
from monai.visualize import matshow3d
from monai.data import PILReader
from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import first, set_determinism
from monai.transforms import LoadImage

from larynx.data.transforms import get_transforms
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
        logging.critical('checjk')
        exit()
    if first_n is None:
        first_n = -int(0.2*len(train_images))  # 20% split ratio Train/Val
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_images)]
    train_files, val_files =  data_dicts[:first_n], data_dicts[first_n:]

    train_transforms, val_transforms = get_transforms()
    set_determinism(seed=0)

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0, num_workers=4)
    # train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=3, shuffle=True, num_workers=4)

    return train_loader


if __name__ == '__main__':
    train_loader = get_dataloaders(first_n=2)

    check_data = first(train_loader)
    print(check_data["image"].shape)
    print(type(check_data["image"]))
    save_image(check_data["image"], folder_name='mico')