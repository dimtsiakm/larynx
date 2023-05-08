from monai.transforms import (LoadImaged, 
                              Resized, 
                              Compose,
                              EnsureChannelFirstd, 
                              CropForegroundd, 
                              Orientationd, 
                              RandCropByPosNegLabeld,
                              ScaleIntensityRanged,
                              Spacingd,
                              )

from monai.transforms import (LoadImage, 
                              Resize, 
                              Compose,
                              ScaleIntensityRange,
                              )


def get_transforms():
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
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RPI"),
            Spacingd(keys=["image", "label"], pixdim=(1.05, 1.05, 1), mode=("nearest")),
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
        ]
    )
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
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RPI"),
            Spacingd(keys=["image", "label"], pixdim=(1.05, 1.05, 1), mode=("nearest")),
        ]
    )
    return train_transforms, None

def transform_png():

    transform = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensityRange(
                a_min=-200,
                a_max=400,
                b_min=0.0,
                b_max=255.0,
                clip=True,
            ),
            Resize(spatial_size=(512, 512, 1)),
        ]
    )
    return transform