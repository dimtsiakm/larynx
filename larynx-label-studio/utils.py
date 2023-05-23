import random
import re
import os

from larynx.utils.config import Config

def get_n_random_points(self, n):
    list_of_points = []
    x = random.random()*60
    y = random.random()*60
    step = 20  # percentage
    for i in range(n):
        if i < n/2:
            x += random.random()*step
            y += random.random()*step
            list_of_points.append([x, y])
        else:
            x -= random.random()*step
            y -= random.random()*step
            list_of_points.append([x, y])
    return list_of_points

def produce_fake_data(self, n=3):
    results = []
    for i in range(n):
        n = random.randint(3, 8)
        randon_class = self.classes[random.randint(0,2)]
        random_points = self.get_n_random_points(n=n)
        result = {
            "value": {
                "points": random_points,
                "polygonlabels": [randon_class]
            },
            "image_rotation": 0,
            "from_name": "label",
            "to_name": "image",
            "type": "polygonlabels"
        }
        results.append(result)
    return results

def extract_names_from_path(pth: str):
    """extract CASO_# from path, as well as the name of the file"""
    matches = re.search(r'CASO_\d+', pth)
    caso_id = matches[0]

    name = os.path.basename(pth)
    name = name.split('.')[0]
    return caso_id, name

def get_pickle_path(img_pth: str, model_str: str):
    config = Config()
    _, img_name = extract_names_from_path(img_pth)
    mask_name = f'masks_{img_name}.pickle'

    pickle_path = os.path.join(config.data_path, 'interim/pickles', model_str)
    os.makedirs(pickle_path, exist_ok=True)
    if os.path.isdir(pickle_path):
        return os.path.join(pickle_path, mask_name)
    else:
        assert('pickle path is not correct')