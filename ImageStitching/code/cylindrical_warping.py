import numpy as np
import cv2
import argparse
from util import *

def warp(image, focal_length, output_path=None):
    warped_image = np.zeros_like(image)
    height, width = warped_image.shape[0], warped_image.shape[1]
    center_y, center_x = height // 2, width // 2
    mk_parent_dir(output_path)

    # inverse warping
    for warped_y in range(height):
        for warped_x in range(width):
            x = np.tan((warped_x - center_x) / focal_length) * focal_length + center_x
            y = (warped_y - center_y) * np.sqrt((x - center_x)**2 + focal_length**2) / focal_length + center_y
            x = round(x)
            y = round(y)
            if x >= 0 and x < width and y >= 0 and y < height:
                warped_image[warped_y, warped_x, :] = image[y, x, :]
            else:
                warped_image[warped_y, warped_x, :] = np.zeros(3)

    # save image
    save_img(output_path, warped_image)
    return warped_image


if __name__ == '__main__':
    
    # parse command line arguments
    # Usage: python3 cylindrical_warping.py <input> <output> [focal]
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of the input images')
    parser.add_argument('output', help='directory of the output images')
    parser.add_argument('-f', '--focal', type=float, help='focal length')
    args = parser.parse_args()
    input_dir = args.input     # eg. ../data/input/parrington
    output_dir = args.output   # eg. ../data/warped/parrington
    focal_length = args.focal if args.focal is not None else 704.916

    # read images, warp
    image_paths = image_paths_under_dir(input_dir)
    for i in range(len(image_paths)):
        image = cv2.imread(f'{input_dir}/{image_paths[i]}')
        warped = warp(image, focal_length, output_path=f'{output_dir}/{image_paths[i]}')
