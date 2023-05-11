from label_studio_ml.model import LabelStudioMLBase
import json
import os
import time

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (get_choice, get_env, get_local_path,
                                   get_single_tag_keys, is_skipped)

from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, UNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import random

# https://labelstud.io/playground : Label Studio tool playground!


class LarynxSegmentator(LabelStudioMLBase):

    def __init__(
        self, batch_size=32, num_epochs=100, logging_steps=1, train_logs=None, **kwargs
    ):
        super(LarynxSegmentator, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.train_logs = train_logs

        # then collect all keys from config which will be used to extract data from task and to form prediction
        # Parsed label config contains only one output of <Choices> type
        assert len(self.parsed_label_config) == 1
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        # print(self.from_name)
        # print('-'*30)
        # print(self.info)
        assert self.info['type'] == 'PolygonLabels'

        # the model has only one textual input
        assert len(self.info['to_name']) == 1
        assert len(self.info['inputs']) == 1
        assert self.info['inputs'][0]['type'] == 'Image'
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']
        self.classes = ["Mucosa", "Cartilage", "Paraglottic Fat"]


    def get_n_points(self, n):
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
            random_points = self.get_n_points(n=n)
            result = {
                "value": {
                    "points": random_points,
                    "polygonlabels": [randon_class]
                },
                "original_width": 508,
                "original_height": 508,
                "image_rotation": 0,
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels"
            }
            results.append(result)
        return results


    def predict(self, tasks, **kwargs):
        predictions = []
        n = random.randint(1, 3)
        results = self.produce_fake_data(n=n)
        # print(str(results))
        
        predictions.append({'result': results, 'score': 0.55})
        
        return predictions
    