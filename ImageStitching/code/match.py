import numpy as np
import cv2
import argparse
import random
import math
from scipy.spatial import distance
from util import *
from harris import *

class Matching:
    def __init__(self, images, features, HOG, output_dir=None, stitch_dir=None):
        self.images = images
        self.features = features # Feature point locations. Ex: (3, 4)
        self.descriptors = HOG # Feature point description. Ex: [0.3 0.33 0.21 0.18 0.29 0.49 0.42 0.3 0.22]
        self.image_amount = len(self.images)
        self.output_dir = output_dir
        self.stitch_dir = stitch_dir
        mk_dir(output_dir)
        mk_dir(stitch_dir)

    def process(self):
    	# creat a blank canvas for final image
        self.result_figure = np.zeros((self.images[0].shape[0], self.images[0].shape[1], 3), np.uint8)
        self.result_figure = np.copy(self.images[0])

        # go through each image
        for index, image in enumerate(self.images):
            if index == (self.image_amount-1): # skip last image
                break
            self.feature_matching(self.features[index], self.features[index+1], self.descriptors[index], self.descriptors[index+1])
            draw_match_line(self.images[index], self.images[index+1], self.match_prev_list, self.match_next_list, path=f'{self.output_dir}/{index}')
            

    def feature_matching(self, feature_points, next_feature_points, descriptors, next_descriptors):
        # find matching feature descriptors for a pair of images
        self.match_prev_list = np.zeros((len(feature_points),2))
        self.match_next_list = np.zeros((len(feature_points),2))
        self.matched_count = 0

        # Go through feature points in the right image
        for i in range(len(feature_points)):
            r, c = feature_points[i][0], feature_points[i][1]
            min_dist = second_min_dist = float("inf")
            match_prev_r = match_prev_c = -1
            match_next_r = match_next_c = -1

            # Go through feature points in the left image
            for j in range(len(next_feature_points)):
                next_r, next_c = next_feature_points[j][0], next_feature_points[j][1]
                # dist = np.linalg.norm(descriptors[i] - next_descriptors[j])
                dist = distance.euclidean(descriptors[i], next_descriptors[j])

                if dist < second_min_dist:
                    if dist < min_dist:
                        min_dist = dist
                        match_prev_r, match_prev_c = r, c
                        match_next_r, match_next_c = next_r, next_c
                    else:
                        second_min_dist = dist

            # # distance ratio is smaller than "matching_threshold", then it is accepted as a match.
            if second_min_dist > 0:    
                if (min_dist/second_min_dist) <= matching_threshold:
                    self.match_prev_list[i] = [int(match_prev_r), int(match_prev_c)]
                    self.match_next_list[i] = [int(match_next_r), int(match_next_c)]
                    self.matched_count += 1
                    # temp_match_prev_list[i] = [int(match_prev_r), int(match_prev_c)]
                    # temp_match_next_list[i] = [int(match_next_r), int(match_next_c)]
                    # temp_matched_count += 1


matching_threshold = 0.76
if __name__ == '__main__':
    # parse command line arguments
    # Usage: python3 matching.py <input> <output> <match> <stitch>
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of the input images')
    parser.add_argument('output', help='directory of the output images')
    parser.add_argument('match', help='directory of the matching images')
    parser.add_argument('stitch', help='directory of the stitched images')
    args = parser.parse_args()
    input_dir = args.input     # eg. ../data/warped/parrington
    output_dir = args.output   # eg. ../data/harris/parrington
    match_dir = args.match    # eg. ../data/matched/parrington
    stitch_dir = args.stitch  # eg. ../data/stitched/parrington

        # apply harris corner detection
    image_paths = image_paths_under_dir(input_dir)
    image_list, feature_list, HOG_list = [], [], []
    for index, file_name in enumerate(image_paths):
        if index < 3:
            image = cv2.imread(f'{input_dir}/{file_name}')
            image_list.append(image)
            harris = HarrisCornerDetector(image, output_path=f'{output_dir}/{file_name}')
            feature_map, feature_point, descriptor_hist = harris.get_feature_map(guassian_window_size, gaussian_sigma, harris_k, non_maximal_window_size, descriptor_window_size, f'{output_dir}/feature_{file_name}')
            feature_list.append(feature_point)
            HOG_list.append(descriptor_hist)

    # match feature and match image
    match = Matching(image_list, feature_list, HOG_list, output_dir=f'{match_dir}', stitch_dir=f'{stitch_dir}')
    match.process()