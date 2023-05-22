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


def inference(img_pth: str, model_str: str):
    img_pth = get_local_path(img_pth)

    img = cv2.imread(img_pth)
    pickle_path = get_pickle_path(img_pth, model_str)

    if os.path.isfile(pickle_path):
        print('WARNING: This image has already been extracted')
    model = load_sam_model(model_str)
    masks = model_extract_masks(model, img)
    return masks
    save_mask(pickle_path, masks)
    # load_mask_and_visualize(pickle_path, model_str, img)

def export_polygons_from_masks(masks):
    print(f'Total number of masks is {len(masks)}')
    for mask in masks:
        segmentation = mask['segmentation'].astype(np.uint8)
        segmentation = segmentation*255
        idx = cv2.findContours(segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(len(idx))
        break
    return


    
if __name__ == '__main__':
    # provide_prediction_to_tasks()
    img_pth = "/data/upload/1/3c78c00f-00010003_CASO_1.png"
    pickle_pth = get_pickle_path(img_pth, 'sam')
    masks = load_mask(pickle_pth)
    export_polygons_from_masks(masks)
