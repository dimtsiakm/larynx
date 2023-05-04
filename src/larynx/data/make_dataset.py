from pathlib import Path
import glob
import os
import numpy as np
import SimpleITK as sitk  # noqa: N813
import numpy as np
import itk
from monai.transforms import SaveImage
from monai.visualize import matshow3d
from monai.data import PILReader

from larynx.data.transforms import get_tranforms
from larynx.utils.config import Config

def get_images_from_folder(project_path=None, folder_name='304'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        
        Gets the data from ../../data/raw
    """
    config = Config()
    if project_path is None:
        project_path = config.project_path

    dir = os.path.join(config.data_raw_path, folder_name)
    train_images = sorted(glob.glob(os.path.join(dir, "*")))
    
    print(len(train_images))

    return train_images


def save_image(img, folder_name):
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
    saver.set_options(data_kwargs={"spatial_ndim": 2})
    img = saver(img)



if __name__ == '__main__':
    config = Config()
    images = get_images_from_folder(folder_name='304')
    transform = get_tranforms()

    test_data = {"image": images[0]}
    result = transform(test_data)

    print(f"image data shape:{result['image'].shape}")

    save_image(img=result["image"], folder_name='304')