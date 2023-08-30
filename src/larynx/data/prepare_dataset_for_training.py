import json
import re
import os
import glob


def get_DECT_images_from(file_upload):
    '''regex to extract the name of sample. For istance: 00010352_CASO_1'''
    filename = re.search('[0-9]+_CASO_[0-9]+', file_upload).group(0)
    subnames = filename.split('_')

    caso_folder = 'CASO_' + subnames[2]
    image_name = subnames[0]
    mod_name = int(image_name)%1000 # different names, e.g. 001033, 002033

    #print(f"name: {image_name}, caso: {caso_folder}")

    rootdir = 'data/raw/larynx_dataset/' + caso_folder
    double_energy_CT_names =  glob.glob(f'{rootdir}/*/*/*{mod_name}', recursive=True)
    return double_energy_CT_names

        

def load_json():
    file = open('data/interim/project-2-at-2023-08-30-12-18-a879983f.json')
    samples = json.load(file)
    for sample in samples:
        pths = get_DECT_images_from(sample['file_upload'])
        print(pths)
        
    file.close()

if __name__ == '__main__':
    load_json()