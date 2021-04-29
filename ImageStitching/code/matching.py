import numpy as np
import cv2
import argparse
from util import *
from harris import *

class Matching:
    def __init__(self, images, features, descriptors, output_dir=None):
        self.images = images
        self.features = features
        self.descriptors = descriptors
        self.image_amount = len(descriptors)
        self.output_dir = output_dir

    def warp(self):
        # go through each image
        for index, image in enumerate(self.images):
            if index == (self.image_amount-1): # skip last image
                break
            self.feature_matching(self.features[index], self.features[index+1], self.descriptors[index], self.descriptors[index+1])
            draw_match_line(self.images[index], self.images[index+1], self.match_list, self.features[index], path=f'{self.output_dir}/{index}')
        return

    def feature_matching(self, feature_points, next_feature_points, descriptor, next_descriptor):
        # find matching feature descriptors for a pair of images
        self.match_list = np.zeros((len(feature_points),2))

        for i in range(len(feature_points)):
            r, c = feature_points[i][0], feature_points[i][1]
            min_dist = second_min_dist = float("inf")
            match_r = match_c = -1

            for j in range(len(next_feature_points)):
                next_r, next_c = next_feature_points[j][0], next_feature_points[j][1]
                dist = np.linalg.norm(descriptor[r, c] - next_descriptor[next_r, next_c])
                if dist < second_min_dist:
                    if dist < min_dist:
                        min_dist = dist
                        match_r, match_c = next_r, next_c
                    else:
                        second_min_dist = dist

            # best match distance is 0.8 times smaller than second best match distance
            if second_min_dist > 0:    
                if (min_dist/second_min_dist) <= 0.8:
                    self.match_list[i] = [match_r, match_c]
            # self.match_list[i] = [match_r, match_c]
        return



if __name__ == '__main__':
    # parse command line arguments
    # Usage: python3 matching.py <input> <output>
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of the input images')
    parser.add_argument('output', help='directory of the output images')
    parser.add_argument('match', help='directory of the matching images')
    args = parser.parse_args()
    input_dir = args.input     # eg. ../data/warped/parrington
    output_dir = args.output   # eg. ../data/harris/parrington_test
    match_dir = args.match    # eg. ../data/matched/parrington

    # apply harris corner detection
    image_paths = image_paths_under_dir(input_dir)
    image_list, feature_list, desc_list = [], [], []
    for index, file_name in enumerate(image_paths):
        # if index < 2:
        image = cv2.imread(f'{input_dir}/{file_name}')
        image_list.append(image)
        harris = HarrisCornerDetector(image, output_path=f'{output_dir}/{file_name}')
        feature_map, feature_desc, feature_point = harris.get_feature_map(guassian_window_size, gaussian_sigma, harris_k, non_maximal_window_size, descriptor_window_size, f'{output_dir}/feature_{file_name}')
        feature_list.append(feature_point)
        desc_list.append(feature_desc)

    # match feature and match image
    match = Matching(image_list, feature_list, desc_list, output_dir=f'{match_dir}')
    match.warp()



