from monai.transforms import (LoadImaged, 
                              Resized, 
                              Compose,
                              EnsureChannelFirstd, 
                              CropForegroundd, 
                              Orientationd, 
                              RandCropByPosNegLabeld,
                              ScaleIntensityRanged,
                              Spacingd,
                              SqueezeDimd
                              )

from monai.transforms import (LoadImage, 
                              Resize, 
                              Compose,
                              ScaleIntensityRange,
                              )

from larynx.utils.config import Config

def get_train_transforms():
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
            Orientationd(keys=["image", "label"], axcodes="RPI"),
            # Spacingd(keys=["image", "label"], pixdim=(1.05, 1.05, 1), mode=("nearest")),
            Resized(keys=["image", "label"], spatial_size=(480, 480, -1)),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 1),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            SqueezeDimd(keys=["image"], dim=-1, update_meta=True),
        ]
    )
    return train_transforms

def get_val_transforms():
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
            Orientationd(keys=["image", "label"], axcodes="RPI"),
            # Spacingd(keys=["image", "label"], pixdim=(1.05, 1.05, 1), mode=("nearest")),
            Resized(keys=["image", "label"], spatial_size=(480, 480, -1)),
            SqueezeDimd(keys=["image"], dim=-1, update_meta=True)
        ]
    )
    return val_transforms

def get_inference_transforms():
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image", margin=1),
            Orientationd(keys=["image"], axcodes="RPI"),
            # Spacingd(keys=["image"], pixdim=(1.05, 1.05, 1), mode=("nearest")),
            Resized(keys=["image"], spatial_size=(480, 480, -1)),
            SqueezeDimd(keys=["image"], dim=-1, update_meta=True)
        ]
    )
    return test_transforms

def get_transforms():
    return get_train_transforms(), get_val_transforms()

def transforms_for_png():
    config = Config()
    min_ct, max_ct = config.get_ct_window_level()
    transform = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensityRange(
                a_min=min_ct,
                a_max=max_ct,
                b_min=0.0,
                b_max=255.0,
                clip=True,
            ),
            Resize(spatial_size=(512, 512, 1)),
        ]
    )
    return transform