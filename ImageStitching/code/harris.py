# ---
# HOG: https://learnopencv.com/histogram-of-oriented-gradients/
# ---
import numpy as np
import cv2
import argparse
from scipy import ndimage
from util import *
import math

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
        self.gradient()
        return self.feature_response(k=harris_k, nms_window_size=nms_window_size, desc_window_size=desc_window_size, feature_map_path=feature_map_path)

    def gradient(self):
        gx = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=1)
        gy = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=1)
        self.mag, self.angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

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
        feature, feature_point, descriptor_hist = self.non_maximal_suppression(response, feature, nms_window_size, desc_window_size)

        # show the image with feature points marked
        if feature_map_path is not None:
            mark_on_img(self.unblurred, feature, path=feature_map_path)

        return feature, feature_point, descriptor_hist
    
    def non_maximal_suppression(self, response, feature_map, window_size, desc_window_size):
        # check if window size is odd
        assert(window_size % 2 == 1)
        assert(desc_window_size % 2 == 1)

        # check every feature point
        margin = window_size // 2
        desc_margin = desc_window_size // 2
        # feature_desc = np.empty([self.height, self.width, desc_window_size**2])
        feature_point = []
        descriptor_hist = []

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
                    if (r - desc_margin) < 0 or (r + desc_margin) > self.height-1:
                        continue
                    if (c - desc_margin) < 0 or (c + desc_margin) > self.width-1:
                        continue
                    feature_point.append([r,c])
                    # feature_desc[r,c] = self.unblurred[r-desc_margin:r+desc_margin+1, c-desc_margin:c+desc_margin+1].flatten()
                    
                    L_x = self.image[r+1,c] - self.image[r-1,c]
                    L_y = self.image[r,c+1] - self.image[r,c-1]
                    dominant_theta = math.atan(L_x / L_y)
                    # dominant_m = math.sqrt(L_x**2 + L_y**2)
                    # ndimage.rotate(self.image, -dominant_theta, reshape=False)

                    mag_temp = self.mag[r-desc_margin:r+desc_margin+1, c-desc_margin:c+desc_margin+1]
                    angle_temp = self.angle[r-desc_margin:r+desc_margin+1, c-desc_margin:c+desc_margin+1]
                    descriptor_hist.append(self.HOG(mag_temp, angle_temp, dominant_theta))
        return feature_map, feature_point, descriptor_hist

    # Calculate Histogram of Gradients
    def HOG(self, mag_temp, angle_temp, dominant_theta):
        assert(descriptor_window_size % 4 == 1)
        bin_number = 8 # bins corresponding to angles 360/45
        bin_array = np.zeros(bin_number)
        step = math.floor(descriptor_window_size / 4)
        bin_vector = np.zeros(bin_number*4*4)
        temp = 0

        for count_row in range(0, 4*step, step):
            for count_col in range(0, 4*step, step):
                for i in range(count_row, count_row+step+1):
                    for j in range(count_col, count_col+step+1):
                        angle_temp[i,j] = (angle_temp[i,j] - dominant_theta) % 360
                        pos = math.floor(angle_temp[i,j] / 45)
                        bin_array[pos] += (mag_temp[i,j] / 45) * (45 - (angle_temp[i,j] % 45))
                        pos = (pos+1) % bin_number
                        bin_array[pos] += (mag_temp[i,j] / 45) * (angle_temp[i,j] % 45)
                array_norm = np.linalg.norm(bin_array)
                array_norm = np.where(array_norm != 0, array_norm, 1) # resolve divided by zero
                bin_array = bin_array / array_norm # normalize
                bin_vector[temp: temp+bin_number] = bin_array
                temp += bin_number


        # for i in range(descriptor_window_size):
        #     for j in range(descriptor_window_size):
        #         angle_temp[i,j] = (angle_temp[i,j] - dominant_theta) % 360
        #         pos = math.floor(angle_temp[i,j] / 45)
        #         bin_array[pos] += (mag_temp[i,j] / 45) * (45 - (angle_temp[i,j] % 45))
        #         pos = (pos+1) % bin_number
        #         bin_array[pos] += (mag_temp[i,j] / 45) * (angle_temp[i,j] % 45)

        # bin_array = bin_array / np.linalg.norm(bin_array) # normalize

        return bin_vector


gaussian_sigma = 3
harris_k = 0.05
guassian_window_size = 31
non_maximal_window_size = 89
descriptor_window_size = 17

# guassian_window_size = 5
# non_maximal_window_size = 15
# descriptor_window_size = 17 # mod 4 == 1
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
        feature_map, feature_point, descriptor_hist = harris.get_feature_map(guassian_window_size, gaussian_sigma, harris_k, non_maximal_window_size, descriptor_window_size, f'{output_dir}/feature_{file_name}')
