import numpy as np
import cv2
import argparse
from util import *

class HarrisCornerDetector:

    def __init__(self, image, output_path=None):
        # convert to grayscale
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.unblurred = image.astype(np.float64)
        self.output_path = output_path
    
    def denoise(self, window_size, sigma):
        self.image = cv2.GaussianBlur(self.unblurred, (window_size, window_size), sigmaX=sigma)
    
    def derivatives(self):
        self.Ix = cv2.Sobel(self.image, cv2.CV_64F, dx=1, dy=0, ksize=3)
        self.Iy = cv2.Sobel(self.image, cv2.CV_64F, dx=0, dy=1, ksize=3)
        self.Ix = cv2.convertScaleAbs(self.Ix).astype(np.float64)
        self.Iy = cv2.convertScaleAbs(self.Iy).astype(np.float64)
        self.Ixx = self.Ix ** 2
        self.Ixy = self.Ix * self.Iy
        self.Iyy = self.Iy ** 2
    
    def gaussian_conv(self, window_size, sigma):
        # get Gaussian kernel
        kernel = cv2.getGaussianKernel(ksize=window_size, sigma=sigma)
        kernel = kernel @ kernel.T

        # compute the sum of the products of derivatives
        self.Sxx = cv2.filter2D(self.Ixx.astype(np.float64), cv2.CV_64F, kernel)
        self.Sxy = cv2.filter2D(self.Ixy.astype(np.float64), cv2.CV_64F, kernel)
        self.Syy = cv2.filter2D(self.Iyy.astype(np.float64), cv2.CV_64F, kernel)
        self.Sxx = cv2.convertScaleAbs(self.Sxx).astype(np.float64)
        self.Sxy = cv2.convertScaleAbs(self.Sxy).astype(np.float64)
        self.Syy = cv2.convertScaleAbs(self.Syy).astype(np.float64)
    
    def corner_response(self, k, feature_map_path=None):
        # k should be between 0.04 and 0.06
        det = self.Sxx * self.Syy - self.Sxy**2
        trace = self.Sxx + self.Syy
        response = det - k * trace**2
        thresh = np.percentile(response, 99)
        corner = (response > thresh)

        # show the image with feature points marked
        if feature_map_path is not None:
            mark_on_img(self.unblurred, corner, path=feature_map_path)

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
        harris.derivatives()
        harris.gaussian_conv(window_size=5, sigma=1)
        harris.corner_response(k=0.05, feature_map_path=f'{output_dir}/feature_{file_name}')

        break