import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from label_studio_ml.model import LabelStudioMLBase
from label_studio_converter import brush
from label_studio_ml.utils import get_image_local_path
import numpy as np
import cv2
import os
from PIL import Image
import string
import random

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic


def prepare_data_for_upload(polygon: list):
    results = []
    result = {
        "value": {
            "points": polygon,
            "polygonlabels": [""]
        },
        "image_rotation": 0,
        "from_name": "label",
        "to_name": "image",
        "type": "polygonlabels"
    }
    results.append(result)
    return results

def transform_pixels_to_percentage(polygon: list, width: int, height: int):
    polygon_percentage = []
    for coord in polygon:
        h, w = coord
        # includes the zero pixels
        h_perc, w_perc = h/(height-1), w/(width-1)
        h_perc, w_perc = h_perc*100, w_perc*100
        polygon_percentage.append([h_perc,w_perc])
    return polygon_percentage

def export_polygons_from_mask(mask):
    segmentation = mask.astype(np.uint8)
    segmentation = segmentation*255
    idxs = cv2.findContours(segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    idxs = idxs[0][0]
    coords = []
    for idx in idxs:
        coords.append(list(idx[0]))
    
    return coords

VITH_CHECKPOINT = os.environ.get("VITH_CHECKPOINT", "sam_vit_h_4b8939.pth")
ONNX_CHECKPOINT = os.environ.get("ONNX_CHECKPOINT", "sam_onnx_quantized_example.onnx")

def load_my_model():
        """
        Loads the Segment Anything model on initializing Label studio, so if you call it outside MyModel it doesn't load every time you try to make a prediction
        Returns the predictor object. For more, look at Facebook's SAM docs
        """
        device = "cuda"     # if you're not using CUDA, use "cpu" instead .... good luck not burning your computer lol
        
        sam = sam_model_registry["vit_h"](VITH_CHECKPOINT)        # Note: YOU MUST HAVE THE MODEL SAVED IN THE SAME DIRECTORY AS YOUR BACKEND
        sam.to(device=device)

        predictor = SamPredictor(sam)
        return predictor

PREDICTOR = load_my_model()
PREV_IMG_PATH = ""
PREV_IMG = 0
IMAGE_EMBEDDING = 0
previous_id = 0

# empty mask and indicator for no mask
onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

ORT = onnxruntime.InferenceSession(ONNX_CHECKPOINT)

class MyModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
    
    def predict(self, tasks, **kwargs):
        """ Returns the predicted mask for a smart keypoint that has been placed."""
        
        # Use this to check times for your predictions
        # print(f"Current data and time1: {str(datetime.datetime.now())}") # Current data and time1: 2023-04-16 18:56:09.361688 (ALMOST INSTANTANEOUS FROM THE RUN)

        results = []
        predictions = []
        predictor = PREDICTOR

        image_url = tasks[0]['data']['image']
        # print(f"the kwargs are {kwargs}")
        # print(f"the tasks are {tasks}")


        # getting the height and width of the image that you are annotating real-time 
        height = kwargs['context']['result'][0]['original_height']
        width = kwargs['context']['result'][0]['original_width']

        # getting x and y coordinates of the keypoint
        x = kwargs['context']['result'][0]['value']['x'] * width / 100
        y = kwargs['context']['result'][0]['value']['y'] * height / 100

        # label that you selected with the keypoint. If this is running into error, use the second line of code instead
        label = kwargs['context']['result'][0]['value']['labels'][0]
        keypointlabel = kwargs['context']['result'][0]['value']['keypointlabels'][0]

        task = tasks[0]
        img_path = task["data"]["image"]

        # loading the image you are annotating
        image_path = get_image_local_path(img_path)


        # this is to speed up inference after the first time you selected an image
        global PREV_IMG_PATH
        global PREV_IMG
        global IMAGE_EMBEDDING
        global onnx_has_mask_input
        global onnx_mask_input
        global previous_id

        if image_path != PREV_IMG_PATH:
            PREV_IMG_PATH = image_path
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            PREV_IMG = image
            # retrieving predictions from SAM. For more info, look at Facebook's SAM docs
            predictor.set_image(image)

            image_embedding = predictor.get_image_embedding()
            IMAGE_EMBEDDING = image_embedding.cpu().numpy()
        elif image_path == PREV_IMG_PATH:
            image = PREV_IMG
            image_embedding = IMAGE_EMBEDDING

        input_point = np.array([[int(x), int(y)]])
        input_label = np.array([1])

        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

        # Package to run in onnx
        ort_inputs = {
            "image_embeddings": IMAGE_EMBEDDING,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
        }

        # Predict and threshold mask
        masks, _, low_res_logits = ORT.run(None, ort_inputs)
        masks = masks > predictor.model.mask_threshold

        mask = masks[0, 0, :, :].astype(np.uint8) # each mask has shape [H, W]
        polygon = export_polygons_from_mask(mask)
        polygon = transform_pixels_to_percentage(polygon, width, height)
        results = prepare_data_for_upload(polygon)
        predictions.append({"result": results,
                            "model_version": "vit_h"
        })

        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        return {'random': random.randint(1, 10)}
