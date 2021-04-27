import numpy as np
import cv2
import argparse
from util import *

class HarrisCornerDetector:

    def __init__(self, image, output_path=None):
        # convert to grayscale
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image = image.astype(np.float64)
        self.output_path = output_path
    
    def denoise(self, window_size, sigma):
        self.image = cv2.GaussianBlur(self.image, (window_size, window_size), sigmaX=sigma)
    
    def compute_derivatives(self):
        self.Ix = cv2.Sobel(self.image, cv2.CV_16S, dx=1, dy=0, ksize=3)
        self.Iy = cv2.Sobel(self.image, cv2.CV_16S, dx=0, dy=1, ksize=3)
        self.Ix = cv2.convertScaleAbs(self.Ix)
        self.Iy = cv2.convertScaleAbs(self.Iy)

if __name__ == '__main__':

    # parse command line arguments
    # Usage: python3 cylindrical_warping.py <input> <output>
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of the input images')
    parser.add_argument('output', help='directory of the output images')
    args = parser.parse_args()
    input_dir = args.input     # eg. ../data/warped/parrington
    output_dir = args.output   # eg. ../data/output/parrington

    # read images
    image_paths = image_paths_under_dir(input_dir)
    for file_name in image_paths:
        image = cv2.imread(f'{input_dir}/{file_name}')

        # harris corner detection
        harris = HarrisCornerDetector(image, output_path=f'{output_dir}/{file_name}')
        harris.denoise(window_size=5, sigma=1)
        harris.compute_derivatives()


        break