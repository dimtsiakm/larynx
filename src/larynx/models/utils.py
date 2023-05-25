import numpy as np
import cv2

from larynx.utils.config import Config

def retrive_binary_filter_imaging(width, height, square_length):
    img = np.zeros((width, height), dtype=np.uint8)
    w_c, h_c = int(width/2), int(height/2)
    square_length = int(square_length/2)
    img[w_c-square_length:w_c+square_length, h_c-square_length:h_c+square_length]=1

    img = img.astype(bool)
    return img

if __name__ == '__main__':
    retrive_binary_filter_imaging()