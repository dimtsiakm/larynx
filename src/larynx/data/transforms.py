from monai.transforms import LoadImage, LoadImaged, Resized, Compose, SaveImage
import numpy as np


def get_tranforms():
    transform = Compose(
            [
                LoadImaged(keys="image", image_only=True, ensure_channel_first=True, dtype=np.uint8),
                Resized(keys="image", spatial_size=[512, 512, -1]),
            ]
        )
    
    return transform

