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
        self.height, self.width = self.unblurred.shape
        mk_parent_dir(output_path)
    
    def get_feature_map(self, guassian_window_size, gaussian_sigma, harris_k, nms_window_size, desc_window_size, feature_map_path=None):
        self.denoise(window_size=guassian_window_size, sigma=gaussian_sigma)
        self.derivatives()
        self.gaussian_conv(window_size=guassian_window_size, sigma=gaussian_sigma)
        return self.feature_response(k=harris_k, nms_window_size=nms_window_size, desc_window_size=desc_window_size, feature_map_path=feature_map_path)

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
    
    def feature_response(self, k, nms_window_size, desc_window_size, feature_map_path=None):
        # k should be between 0.04 and 0.06
        det = self.Sxx * self.Syy - self.Sxy**2
        trace = self.Sxx + self.Syy
        response = det - k * trace**2
        thresh = np.percentile(response, 99)
        feature = (response > thresh)
        feature, description, feature_point = self.non_maximal_suppression(response, feature, nms_window_size, desc_window_size)

        # show the image with feature points marked
        if feature_map_path is not None:
            mark_on_img(self.unblurred, feature, path=feature_map_path)

        return feature, description, feature_point
    
    def non_maximal_suppression(self, response, feature_map, window_size, desc_window_size):
        # check if window size is odd
        assert(window_size % 2 == 1)
        assert(desc_window_size % 2 == 1)

        # check every feature point
        margin = window_size // 2
        desc_margin = desc_window_size // 2
        feature_desc = np.empty([self.height, self.width, desc_window_size**2])
        feature_point = []

        for r in range(self.height):
            for c in range(self.width):
                # continue if not feature point
                if not feature_map[r, c]:
                    continue
                # construct the window
                up = max(r - margin, 0)
                down = min(r + margin, self.height-1)
                left = max(c - margin, 0)
                right = min(c + margin, self.width-1)
                # remove feature if not having the largest response within the given window
                window_max = np.max(response[up:down+1, left:right+1])
                if response[r, c] < window_max:
                    feature_map[r, c] = False
                # create feature description
                else:
                    feature_point.append([r,c])
                    if (r - desc_margin) < 0 or (r + desc_margin) > self.height-1:
                        continue
                    if (c - desc_margin) < 0 or (c + desc_margin) > self.width-1:
                        continue
                    feature_desc[r,c] = self.unblurred[r-desc_margin:r+desc_margin+1, c-desc_margin:c+desc_margin+1].flatten()
        return feature_map, feature_desc, feature_point


guassian_window_size = 5
gaussian_sigma = 3
harris_k = 0.05
non_maximal_window_size = 15
descriptor_window_size = 11
if __name__ == '__main__':

    # parse command line arguments
    # Usage: python3 harris.py <input> <output>
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of the input images')
    parser.add_argument('output', help='directory of the output images')
    args = parser.parse_args()
    input_dir = args.input     # eg. ../data/warped/parrington
    output_dir = args.output   # eg. ../data/harris/parrington

    # apply harris corner detection
    image_paths = image_paths_under_dir(input_dir)
    for file_name in image_paths:
        image = cv2.imread(f'{input_dir}/{file_name}')
        harris = HarrisCornerDetector(image, output_path=f'{output_dir}/{file_name}')
        feature_map, feature_desc, feature_point = harris.get_feature_map(guassian_window_size, gaussian_sigma, harris_k, non_maximal_window_size, descriptor_window_size, f'{output_dir}/feature_{file_name}')
