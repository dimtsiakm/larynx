import numpy as np
import cv2

from larynx.utils.config import Config

def retrieve_contours():
    """create an image 480x480, and create a square at the center with length 
    equals to 100, then save the image"""
    img = np.zeros((480, 480), dtype=np.uint8)
    w, h = img.shape
    w_c, h_c = int(w/2), int(h/2)
    img[w_c-50:w_c+50, h_c-50:h_c+50]=1

    config = Config()
    print(img)

    # img = img*255
    # cv2.imwrite(config.temp_figures_path + 'temp.png', img)

    
    idx = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(idx[0][0]))
    print(idx[0][0])


if __name__ == '__main__':
    retrieve_contours()