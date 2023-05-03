from pathlib import Path
import glob
import os
import shutil
import SimpleITK as sitk  # noqa: N813
import numpy as np
import itk
import tempfile
import monai
from monai.data import PILReader
from monai.transforms import LoadImage, LoadImaged, Resized, Compose, SaveImage
from monai.config import print_config


def get_data(project_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        
        Gets the data from ../../data/raw
    """

    caso_dir = os.path.join(project_path, 'data/', "raw/304/")
    train_images = sorted(glob.glob(os.path.join(caso_dir, "*")))
    
    print(caso_dir)
    print(len(train_images))

    return train_images


def main():
    return

if __name__ == '__main__':
    
    # not used in this stub but often useful for finding various files
    PROJECT_DIR = Path(__file__).resolve().parents[2]

    get_data(PROJECT_DIR)

