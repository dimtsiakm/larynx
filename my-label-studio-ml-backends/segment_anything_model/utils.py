import json
import os
import cv2
import random
import numpy as np

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_local_path, get_image_size

from label_studio_tools.core.utils.io import get_data_dir

from larynx.utils.config import Config
from larynx.models.utils import retrive_binary_filter_imaging
from larynx.models.predict_model import (load_sam_model, 
                                         model_extract_masks, 
                                         save_mask, 
                                         load_mask, 
                                         load_mask_and_visualize)
from utils import produce_fake_data, extract_names_from_path, get_pickle_path


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

def there_are_predictions_from(model: str, task):
    predictions = task['predictions']
    for prediction in predictions:
        if prediction['model_version'] == model:
            return True
    return False

def masks_filtering(masks: list, square_length: int):
    results = []
    width, height = masks[0]['segmentation'].shape
    mask_filter = retrive_binary_filter_imaging(width, height, square_length)
    for mask in masks:
        mask_array = mask['segmentation']
        logical_and_array = np.logical_and(mask_array, mask_filter)
        if (mask_array == logical_and_array).all() and mask_array.sum() > 250:
            results.append(mask)
    return results

def make_predictions_dataset(first_n:int=None, upload_masks:bool=False, filter_out:bool=True):
    EARLY_STOP = False
    if first_n is not None:
        EARLY_STOP = True
    
    if not upload_masks:
        print('The procedure will not upload the results..')

    model_version = 'MedSAM-23.05.23'
    if filter_out:
        model_version_str = f'{model_version}-filtered-out'
    else:
        model_version_str = model_version

    ls = label_studio_sdk.Client(HOSTNAME, API_KEY)
    project = ls.get_project(id=1)
    tasks_id = project.get_tasks_ids()

    for i, task_id in enumerate(tasks_id):
        if EARLY_STOP:
            if i >= first_n:
                return
        print(f'Task {task_id} is processing...')
        task = project.get_task(task_id)
        
        if there_are_predictions_from(model_version_str, task):
            print('This task is skipped because predictions from this model has already been uploaded.')
            continue
        img_pth = task['data']['image']
        img_pth = get_local_path(img_pth)
        width, height = get_image_size(img_pth)
        masks = inference(img_pth, model_version, save_results=True)
        if filter_out:
            masks = masks_filtering(masks, square_length=350)
        polygons = export_polygons_from_masks(masks)
        polygons = transform_pixels_to_percentage_polygons(polygons, width, height)
        results = prepare_data_for_upload(polygons)
        if upload_masks:
            project.create_prediction(task_id, result=results, score=0.5, model_version=model_version_str)
            print('prediction uploaded successfully')
        

if __name__ == '__main__':
    make_predictions_dataset(first_n=None, upload_masks=True, filter_out=True)