import json
import re
import os
import glob

from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import first, set_determinism

from larynx.models.transforms import get_transforms

def get_DECT_images_from_filename(file_upload):
    '''regex to extract the name of sample. For istance: 00010352_CASO_1'''
    filename = re.search('[0-9]+_CASO_[0-9]+', file_upload).group(0)
    subnames = filename.split('_')

    caso_folder = 'CASO_' + subnames[2]
    image_name = subnames[0]
    mod_name = int(image_name)%1000 # different names, e.g. 001033, 002033

    #print(f"name: {image_name}, caso: {caso_folder}")

    rootdir = 'data/raw/larynx_dataset/' + caso_folder
    DECT_names =  glob.glob(f'{rootdir}/*/*/*{mod_name}', recursive=True)
    return DECT_names

def load_json():
    file = open('data/interim/project-2-at-2023-08-30-12-18-a879983f.json')
    samples = json.load(file)
    file.close()

    return samples
    

def get_dataloaders():
    samples = load_json()

    images_per_sample = []
    for sample in samples:
        images = get_DECT_images_from_filename(sample['file_upload'])
        images_per_sample.append(images)
    
    # Create the GT images from the json's instructions.


    """
    TODO: In the final version should be like the command above! Check it.
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    """
    data_dicts = [{"image": image[0]} for image in images_per_sample]
    
    """
    TODO: I should choose a plan regarding data; All images contribute equal to the training? Or in each epoch, the dataloader would choose an image randomly?
    """

    print(images_per_sample)
    print(data_dicts)

    # train_transforms, val_transforms = get_transforms()
    # set_determinism(seed=0)

    # train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0, num_workers=4)
    # train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    # val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0, num_workers=4)
    # val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # return train_loader, val_loader


if __name__ == '__main__':
    get_dataloaders()