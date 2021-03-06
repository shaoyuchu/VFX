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
        H_list = []

        # go through each image
        for index, image in enumerate(self.images):
            if index == (self.image_amount-1): # skip last image
                break
            self.feature_matching(self.features[index], self.features[index+1], self.descriptors[index], self.descriptors[index+1])
            draw_match_line(self.images[index], self.images[index+1], self.match_prev_list, self.match_next_list, path=f'{self.output_dir}/{index}')
            H, inliers_pair = self.Ransac()
            H_list.append(H)
            self.blend(H_list, self.images[index], self.images[index+1], index)
        
        uint_img = self.result_figure.astype(np.uint8)
        # show_img("result",uint_img)
        save_img(f'{self.stitch_dir}/result.jpg', uint_img)

    def feature_matching(self, feature_points, next_feature_points, descriptors, next_descriptors):
        # find matching feature descriptors for a pair of images
        temp_match_prev_list = np.zeros((len(feature_points),2))
        temp_match_next_list = np.zeros((len(feature_points),2))
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
                    temp_match_prev_list[self.matched_count] = [int(match_prev_r), int(match_prev_c)]
                    temp_match_next_list[self.matched_count] = [int(match_next_r), int(match_next_c)]
                    self.matched_count += 1
        
        self.match_prev_list = temp_match_prev_list[:self.matched_count]
        self.match_next_list = temp_match_next_list[:self.matched_count]
        self.all_match_pairs = np.concatenate([self.match_prev_list, self.match_next_list], axis=1)

    def Ransac(self):
        H, mask = cv2.findHomography(self.match_prev_list, self.match_next_list)

        # get sample points with no duplicate
        min_error, max_inlier = float("inf"), float("-inf")
        best_H = H
        for k in range(repeat_k):   
            sample, sample_uv, sample_xy = [], [], []
            while True:
                temp = random.randint(0,len(self.match_prev_list)-1)
                sample.append(temp)

                if len(set(sample)) == sample_amount:
                    break

            sample = set(sample)
            for s in sample:
                sample_uv.append([int(self.match_prev_list[s][0]), int(self.match_prev_list[s][1])])
                sample_xy.append([int(self.match_next_list[s][0]), int(self.match_next_list[s][1])])

            H = self.homography(sample_xy, sample_uv)
            # error, inliers_amount, inliers_pair = self.get_error_old(H)
            error, inliers_pair = self.get_error(H, self.match_next_list, self.match_prev_list)
            inliers_amount = len(inliers_pair)

            if inliers_amount > max_inlier:
                min_error = error[0]
                max_inlier = inliers_amount
                best_H = H.copy()
                best_inliers_pair = inliers_pair
        # print(best_inliers_pair)
        return best_H, best_inliers_pair

    # calculate match point distance with H as homography matrix
    def get_error_old(self, H):
        estimate_point = np.zeros((len(self.match_next_list), 2))
        errors, inliers_amount = 0, 0
        inliers_pair = []

        for i in range(len(self.match_next_list)):
            H = H[0:2,0:2]
            estimate_point[i] = np.dot(H, self.match_next_list[i])
            err = math.sqrt((self.match_prev_list[i][0] - estimate_point[i][0])**2 + (self.match_prev_list[i][1] - estimate_point[i][1])**2 )
            if err < inlier_threshold:
                inliers_amount += 1
                inliers_pair.append([self.match_prev_list[i][0], self.match_prev_list[i][1], int(self.match_next_list[i][0]), int(self.match_next_list[i][1])])
            errors += err
        return errors, inliers_amount, inliers_pair


    def homography(self, pair_xy, pair_uv):
	    rows = []
	    for i in range(len(pair_xy)):
	        p1 = np.append(pair_xy[i][0:2], 1)
	        p2 = np.append(pair_uv[i][0:2], 1)
	        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
	        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
	        rows.append(row1)
	        rows.append(row2)
	    rows = np.array(rows)
	    U, s, V = np.linalg.svd(rows)
	    H = V[-1].reshape(3, 3)
	    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
	    return H

    def get_error(self, H, all_p1, all_p2): # p1 is next, p2 is prev
        num_points = len(all_p1)
        all_p1 = np.concatenate((all_p1, np.ones((num_points, 1))), axis=1)
        estimate_p2 = np.zeros((num_points, 2))

        for i in range(num_points):
            temp = np.dot(H, all_p1[i])
            temp = np.where(temp != 0, temp, 0.001) # resolve divided by zero
            estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
        # Compute error
        errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2
        idx = np.where(errors < inlier_threshold)[0]
        inliers = self.all_match_pairs[idx]

        return errors, inliers


    def blend(self, H_list, img, next_img, index):
        min_r, min_c, max_r, max_c = 0, 0, 0, 0
        h_result, w_result = self.result_figure.shape[:2]

        # Save the new positions (pixels transfer by matrix H) to an array
        move_position_array = np.zeros((next_img.shape[0], next_img.shape[1], 2))
        for r in range(len(img)):
            for c in range(len(img[0])):
                for idx in range(index, -1, -1):
                    pos = np.dot(H_list[index], [r,c,1])
                    pos = np.where(pos!= 0, pos, 0.001) # resolve divided by zero
                norm_pos = (pos/pos[2])[0:2]
                new_pos_r = int(round(norm_pos[0]))
                new_pos_c = int(round(norm_pos[1]))
                move_position_array[r,c] = [new_pos_r, new_pos_c]
        # find the min and max position
        min_r = np.amin(move_position_array[:,:,0])
        max_r = np.amax(move_position_array[:,:,0])
        min_c = np.amin(move_position_array[:,:,1])
        max_c = np.amax(move_position_array[:,:,1])

        # print(min_r, max_r)
        # print(min_c, max_c)

        final_move_row = move_position_array[:,:,0].copy()
        final_move_col = move_position_array[:,:,1].copy()

        # Expand the final image column
        if min_c < 0:
            self.result_figure = np.hstack((np.zeros((h_result, int(abs(min_c)), 3)), self.result_figure))
            final_move_col = move_position_array[:,:,1] + int(abs(min_c))
        if max_c >= img.shape[1]:
            self.result_figure = np.hstack((self.result_figure, np.zeros((h_result, int(abs(max_c)-img.shape[1]), 3))))
        # Expand the final image row
        if min_r < 0:
            self.result_figure = np.vstack((np.zeros((int(abs(min_r)+1), w_result+int(abs(min_c)), 3)), self.result_figure))
            final_move_row = move_position_array[:,:,0] + int(abs(min_r))
        if max_r >= img.shape[0]:
            self.result_figure = np.vstack((self.result_figure, np.zeros((int(abs(max_r)-img.shape[0]+1), w_result+int(abs(min_c)), 3))))

        blending_width = int((next_img.shape[1] - abs(min_c))/5)

        for r in range(len(next_img)):
            b = 0
            for c in range(len(next_img[0])- 2*blending_width):        
                temp_r = int(final_move_row[r,c])
                temp_c = int(final_move_col[r,c])
                if c < len(next_img[0]) - 3*blending_width:
                    self.result_figure[temp_r,temp_c] = next_img[r, c]
                # linear blending
                else:
                    self.result_figure[temp_r,temp_c] = next_img[r, c]*(1 - b/blending_width) + self.result_figure[temp_r,temp_c]*(b/blending_width)
                    b += 1


repeat_k = 80
sample_amount = 4
matching_threshold = 0.7
inlier_threshold = 20
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
        # if index <= 3 and index >= 2:
        image = cv2.imread(f'{input_dir}/{file_name}')
        image_list.append(image)
        harris = HarrisCornerDetector(image, output_path=f'{output_dir}/{file_name}')
        feature_map, feature_point, descriptor_hist = harris.get_feature_map(guassian_window_size, gaussian_sigma, harris_k, non_maximal_window_size, descriptor_window_size, f'{output_dir}/feature_{file_name}')
        feature_list.append(feature_point)
        HOG_list.append(descriptor_hist)

    # match feature and match image
    match = Matching(image_list, feature_list, HOG_list, output_dir=f'{match_dir}', stitch_dir=f'{stitch_dir}')
    match.process()