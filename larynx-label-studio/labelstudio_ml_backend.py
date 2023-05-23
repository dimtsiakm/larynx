import json
import os
import cv2
import random
import numpy as np

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_local_path, get_image_size

from label_studio_tools.core.utils.io import get_data_dir

from larynx.utils.config import Config
from larynx.models.predict_model import (load_sam_model, 
                                         model_extract_masks, 
                                         save_mask, 
                                         load_mask, 
                                         load_mask_and_visualize)
from utils import produce_fake_data, extract_names_from_path, get_pickle_path
# https://labelstud.io/playground : Label Studio tool playground!

HOSTNAME = 'http://localhost:8080'
API_KEY = 'a7da269601ff79cd1e28cc6161826d28134cd2dc'

class LarynxSegmentator(LabelStudioMLBase):

    def __init__(
        self, batch_size=32, num_epochs=100, logging_steps=1, train_logs=None, **kwargs
    ):
        super(LarynxSegmentator, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.train_logs = train_logs

        assert len(self.parsed_label_config) == 1
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        assert self.info['type'] == 'PolygonLabels'

        assert len(self.info['to_name']) == 1
        assert len(self.info['inputs']) == 1
        assert self.info['inputs'][0]['type'] == 'Image'
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']
        self.classes = ["Mucosa", "Cartilage", "Paraglottic Fat"]


    def predict(self, tasks, **kwargs):
        predictions = []
        n = random.randint(1, 3)
        results = produce_fake_data(n)
        
        predictions.append({'result': results, 'score': 0.55})
        
        return predictions


import label_studio_sdk
def download_tasks(project):
    
    ls = label_studio_sdk.Client(HOSTNAME, API_KEY)
    project = ls.get_project(id=project)
    
    task_ids = project.get_tasks_ids()
    length = min(4, len(task_ids))
    for i, id in enumerate(task_ids):
        print(id)
        if i > length:
            return
    project.create_prediction(task_ids[1], result=[], score=0.9)


def inference(img_pth: str, model_str: str, save_results: bool):
    if not os.path.isfile(img_pth):
        img_pth = get_local_path(img_pth)

    img = cv2.imread(img_pth)
    pickle_path = get_pickle_path(img_pth, model_str)

    if os.path.isfile(pickle_path):
        print('WARNING: This image has already been extracted')
        return load_mask(pickle_path)
    model = load_sam_model(model_str)
    masks = model_extract_masks(model, img)
    if save_results:
        save_mask(pickle_path, masks)
    return masks


def transform_pixels_to_percentage_polygons(polygons: list, width: int, height: int):
    polygons_processed = []
    for pol in polygons:
        polygon_processed = transform_pixels_to_percentage(pol, width, height)
        polygons_processed.append(polygon_processed)
    return polygons_processed


def transform_pixels_to_percentage(polygon: list, width: int, height: int):
    polygon_percentage = []
    for coord in polygon:
        h, w = coord
        h_perc, w_perc = h/height, w/width
        h_perc, w_perc = h_perc*100, w_perc*100
        polygon_percentage.append([h_perc,w_perc])
    return polygon_percentage


def export_polygons_from_masks(masks):
    print(f'Total number of masks is {len(masks)}')
    polygons = []
    for mask in masks:
        polygon = export_polygons_from_mask(mask)
        polygons.append(polygon)
    return polygons


def export_polygons_from_mask(mask):
    segmentation = mask['segmentation'].astype(np.uint8)
    segmentation = segmentation*255
    idxs = cv2.findContours(segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    idxs = idxs[0][0]
    coords = []
    for idx in idxs:
        coords.append(list(idx[0]))
    return coords

def prepare_data_for_upload(polygons: list):
    results = []
    for polygon in polygons:
        result = {
            "value": {
                "points": polygon,
                "polygonlabels": ["Mucosa"]
            },
            "image_rotation": 0,
            "from_name": "label",
            "to_name": "image",
            "type": "polygonlabels"
        }
        results.append(result)
    return results


def make_predictions_dataset(first_n:int=None):
    EARLY_STOP = False
    if first_n is not None:
        EARLY_STOP = True

    model_version = 'SAM-23.05.23'
    ls = label_studio_sdk.Client(HOSTNAME, API_KEY)
    project = ls.get_project(id=1)
    tasks_id = project.get_tasks_ids()
    
    for i, task_id in enumerate(tasks_id):
        if EARLY_STOP:
            if i >= first_n:
                return
        print(f'Task {task_id} is processing...')
        task = project.get_task(task_id)
        img_pth = task['data']['image']
        img_pth = get_local_path(img_pth)
        width, height = get_image_size(img_pth)
        masks = inference(img_pth, model_version, save_results=True)
        polygons = export_polygons_from_masks(masks)
        polygons = transform_pixels_to_percentage_polygons(polygons, width, height)
        results = prepare_data_for_upload(polygons)
        project.create_prediction(task_id, result=results, score=0.5, model_version=model_version)
        print('prediction uploaded successfully')
   
if __name__ == '__main__':
    make_predictions_dataset(first_n=None)