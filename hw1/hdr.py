import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def parse_shutter_speed(filename):
    # eg. '1_8.JPG' -> 0.125
    segments = filename.replace('.', '_').replace('/', '_').split('_')
    return int(segments[-3]) / int(segments[-2])

def read_images(image_paths):
    all_B, all_G, all_R = [], [], []
    for path in image_paths:
        image = cv2.imread(path)
        print(f'Reading {path}: {image.shape}')
        all_B.append(image[:, :, 0])
        all_G.append(image[:, :, 1])
        all_R.append(image[:, :, 2])
        
    all_B = np.array(all_B)
    all_G = np.array(all_G)
    all_R = np.array(all_R)
    return all_B, all_G, all_R




# read images
# sample Eij
# construct linear system
# solve linear system, get inverse of response curve
# construct HDR radiance map
# save as .hdr

if __name__ == '__main__':

    # read images
    input_image_dir = './input_images/bridge/'
    input_image_paths = os.listdir(input_image_dir)
    input_image_paths = list(map(lambda f: input_image_dir+f, input_image_paths))
    input_image_paths = list(filter(lambda f: f.endswith('.JPG'), input_image_paths))
    all_b, all_g, all_r = read_images(input_image_paths)
    
    # parse shutter speed
    shutter = np.array(list(map(parse_shutter_speed, input_image_paths)))
    print(shutter)
    
