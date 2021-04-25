import numpy as np
import cv2
import argparse


if __name__ == '__main__':
    
    # parse command line arguments
    # Usage: python3 cylindrical_warping <input> <output> [focal]
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="directory of the input images")
    parser.add_argument("output", help="directory of the output images")
    parser.add_argument("-f", "--focal", type=float, help="focal length")
    args = parser.parse_args()
    input_path = args.input     # eg. ../data/input/parrington
    output_path = args.output   # eg. ../data/output/parrington
    focal = args.focal if args.focal is not None else 704.916

