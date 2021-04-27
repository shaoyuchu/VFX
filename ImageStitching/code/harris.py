import numpy as np
import cv2
import argparse
from util import *

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
    print(image_paths)