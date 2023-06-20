import os
import json
import glob
from monai.utils import set_determinism, first
from larynx.utils.config import Config

def get_images(percentge=0.8):
    DATASET_PATH = "CT-Covid-19-August2020"
    config = Config()

    data_root = os.path.join(config.data_raw_path, DATASET_PATH, "data")
    print(f"DATASET_PATH: {DATASET_PATH}")

    images = sorted(glob.glob(os.path.join(data_root, "*")))

    for idx, _each_d in enumerate(images):
        images[idx] = os.path.join(data_root, images[idx])

    images = [{"image": img} for img in images]

    idx = int(percentge * len(images))
    train_data = images[:idx]
    val_data = images[idx:]

    

    print("Total Number of Training Data Samples: {}".format(len(train_data)))
    print("#" * 10)
    print("Total Number of Validation Data Samples: {}".format(len(val_data)))
    # print(val_data)
    return train_data, val_data

if __name__ == '__main__':
    train, _ = get_images()
    print(train[0])