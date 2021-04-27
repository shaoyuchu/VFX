import numpy as np
import cv2
import os

def image_paths_under_dir(input_dir):
    image_paths = os.listdir(input_dir)
    image_paths = list(filter(lambda f: (f.endswith('.JPG') or f.endswith('.jpg') or f.endswith('.png')), image_paths))
    image_paths.sort()
    return image_paths

def mk_parent_dir(path):
    if path is None:
        return
    directory = path.rsplit('/', 1)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_img(path, image):
    if path is None:
        return
    cv2.imwrite(path, image)
    print(f'{path} saved')
