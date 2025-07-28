# import torch
# from PIL import Image
from torchvision import transforms

# Crops the spectrogram image to a specified box with excludes ticks and labels (left, upper, right, lower)
class CropSpectrogramPlot:
    def __init__(self, crop_box=(100, 70, 720, 535)):
        self.crop_box = crop_box

    def __call__(self, img):
        return img.crop(self.crop_box)

# Compose all transforms needed for the PNG dataset
def get_png_transform(input_size=(128, 128)):
    return transforms.Compose([
        CropSpectrogramPlot(),
        transforms.Resize(input_size),
        transforms.ToTensor(),  # Converts to shape [C, H, W], range [0,1]
    ])