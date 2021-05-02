# -----
# For DLT: https://github.com/Fishified/Tracker3D/blob/master/DLT.py
# For RANSAC: https://tigercosmos.xyz/post/2020/05/cv/image-stitching/
# -----
import numpy as np
import cv2
import argparse
import random
import math
from util import *
from harris import *

class Matching:
    def __init__(self, images, features, descriptors, output_dir=None, stitch_dir=None):
        self.images = images
        self.features = features
        self.descriptors = descriptors
        self.image_amount = len(descriptors)
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
            draw_match_line(self.images[index], self.images[index+1], self.match_list, self.features[index], path=f'{self.output_dir}/{index}')
            H_matrix, inliers_pair = self.Ransac(self.match_list, self.features[index])
            # self.blend_single(self.images[index], self.images[index+1], inliers_pair, path=f'{self.stitch_dir}/{index}')
            self.blend(self.images[index], self.images[index+1], inliers_pair)
        uint_img = self.result_figure.astype(np.uint8)
        save_img(f'{self.stitch_dir}/result.jpg', uint_img)

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

            # best match distance is much smaller than second best match distance
            if second_min_dist > 0:    
                if (min_dist/second_min_dist) <= matching_threshold:
                    self.match_list[i] = [int(match_r), int(match_c)]


    def Ransac(self, next_img_match, img_feature):
        min_error, max_inlier = float("inf"), float("-inf")
        for k in range(repeat_k):
            # get sample points with no duplicate
            sample, sample_uv, sample_xy = [], [], []
            while True:
                temp = random.randint(0,len(next_img_match)-1)
                if next_img_match[temp][0] != 0 and next_img_match[temp][1] != 0:
                    sample.append(temp)

                if len(set(sample)) == sample_amount:
                    break

            sample = set(sample)
            for s in sample:
                sample_uv.append([int(next_img_match[s][0]), int(next_img_match[s][1])])
                sample_xy.append([int(img_feature[s][0]), int(img_feature[s][1])])

            # calculate H using normalized DLT
            H = self.DLT(sample_xy, sample_uv)
            if np.linalg.matrix_rank(H) < 3:
                continue
            error, inlier, inliers_pair = self.get_error(H, next_img_match, img_feature)
            if error < min_error:
                min_error = error
                max_inlier = inlier
                best_H = H.copy()
                best_inliers_pair = inliers_pair

        # print(min_error, max_inlier)
        return best_H, best_inliers_pair
            
    def DLT(self, xy, uv):
        xy = np.asarray(xy)
        uv = np.asarray(uv)
        # number of points
        num = xy.shape[0]

        Txy, xyn = self.Normalization(xy)
        Tuv, uvn = self.Normalization(uv)

        A = []
        for i in range(num):
            x,y = xyn[i,0], xyn[i,1]
            u,v = uvn[i,0], uvn[i,1]
            A.append( [x, y, 1, 0, 0, 0, -u*x, -u*y, -u] )
            A.append( [0, 0, 0, x, y, 1, -v*x, -v*y, -v] )

        # convert A to array
        A = np.asarray(A) 
        # find the 8 parameters:
        U, S, Vh = np.linalg.svd(A)
        # the parameters are in the last line of Vh and normalize them:
        L = Vh[-1,:] / Vh[-1,-1]
        # camera projection matrix
        H = L.reshape(3,3)
        # denormalization
        H = np.dot( np.dot( np.linalg.pinv(Tuv), H ), Txy );
        H = H / H[-1,-1]
        # L = H.flatten(order='C')

        return H

    def Normalization(self, x):
        x = np.asarray(x)
        m, s = np.mean(x,0), np.std(x)
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
            
        Tr = np.linalg.inv(Tr)
        x = np.dot( Tr, np.concatenate( (x.T, np.ones((1,x.shape[0]))) ) )
        x = x[0:2,:].T
        return Tr, x

    # calculate match point distance with H as homography matrix
    def get_error(self, H, next_img_match, img_feature):
        xy_match_points = np.concatenate((img_feature, np.ones(((len(next_img_match)), 1))), axis=1)
        estimate_point = np.zeros((len(next_img_match), 2))
        errors, inliers = 0, 0
        inliers_pair = []

        for i in range(len(next_img_match)):
            if next_img_match[i][0] == 0 and next_img_match[i][1] == 0:
                continue
            prime = np.dot(H, xy_match_points[i])
            estimate_point[i] = prime[0:2]
            err = math.sqrt((next_img_match[i][0] - estimate_point[i][0])**2 + (next_img_match[i][1] - estimate_point[i][1])**2 )
            if err < threshold:
                inliers += 1
                inliers_pair.append([img_feature[i][0], img_feature[i][1], int(next_img_match[i][0]), int(next_img_match[i][1])])
            errors += err
        return errors, inliers, inliers_pair

    def blend_single(self, feature_image, matching_image, inliers_pair, path):
        shift_r, shift_c, temp_x, temp_y  = 0, 0, 0, 0
        for inliers in inliers_pair:
            temp_x += inliers[2] - inliers[0]
            temp_y += inliers[3] - inliers[1]
        shift_r = int(temp_x / len(inliers_pair))
        shift_c = int(temp_y / len(inliers_pair))

        # stitch image to the right
        if shift_c < 0:
            temp = matching_image
            matching_image = feature_image
            feature_image = temp

        height, width = feature_image.shape[:2]
        height_match, width_match = matching_image.shape[:2]
        blending_col = width-abs(shift_c)
        # creat a blank canvas
        blank_image = np.zeros((height+abs(shift_r), width+abs(shift_c), 3), np.uint8)
        
        # stitch image is higher
        if shift_r >= 0:
            linear_wid = int(blending_col/5)
            sr = abs(shift_r)
            sc = abs(shift_c)
            blank_image[0:height_match, 0:sc+2*linear_wid] = matching_image[0:height_match, 0:sc+2*linear_wid]
            blank_image[sr:height+sr, sc+3*linear_wid:width+sc] = feature_image[0:height, 3*linear_wid:width]
            for r in range(sr,height):
                for c in range(0,linear_wid):
                    temp = (feature_image[r-sr,2*linear_wid+c] * (c / linear_wid)) + (matching_image[r,sc+2*linear_wid+c] * (1 - c / linear_wid))
                    blank_image[r, sc+2*linear_wid+c] = temp
        # stitch image is lower
        else:
            linear_wid = int(blending_col/5)
            sr = abs(shift_r)
            sc = abs(shift_c)
            blank_image[sr:height_match+sr, 0:sc+2*linear_wid] = matching_image[0:height_match, 0:sc+2*linear_wid]
            blank_image[0:height, sc+3*linear_wid:width+sc] = feature_image[0:height, 3*linear_wid:width]
            for r in range(sr,height):
                for c in range(0,linear_wid):
                    temp = (feature_image[r-sr,2*linear_wid+c] * (c / linear_wid)) + (matching_image[r,sc+2*linear_wid+c] * (1 - c / linear_wid))
                    blank_image[r, sc+2*linear_wid+c] = temp

        # show_img("result", blank_image)
        uint_img = blank_image.astype(np.uint8)
        cv2.imwrite(f'{path}.jpg', uint_img)

    def blend(self, feature_image, matching_image, inliers_pair):
        shift_r, shift_c, temp_x, temp_y  = 0, 0, 0, 0
        for inliers in inliers_pair:
            temp_x += inliers[2] - inliers[0]
            temp_y += inliers[3] - inliers[1]
        shift_r = int(temp_x / len(inliers_pair))
        shift_c = int(temp_y / len(inliers_pair))

        h_feature, w_feature = feature_image.shape[:2]
        h_match, w_match = matching_image.shape[:2]
        h_result, w_result = self.result_figure.shape[:2]
        blending_col = w_feature-abs(shift_c)

        self.result_figure = np.hstack((np.zeros((h_result, abs(shift_c), 3)), self.result_figure))
        self.result_figure = np.vstack((np.zeros((abs(shift_r), w_result+abs(shift_c), 3)), self.result_figure))
        h_result, w_result = self.result_figure.shape[:2]
        
        # stitch image is higher
        if shift_r >= 0:
            linear_wid = int(blending_col/5)
            sr = abs(shift_r)
            sc = abs(shift_c)
            self.result_figure[0:h_match, 0:sc+2*linear_wid] = matching_image[0:h_match, 0:sc+2*linear_wid]
            for r in range(sr,h_match):
                for c in range(0,linear_wid):
                    temp = (feature_image[r-sr,2*linear_wid+c] * (c / linear_wid)) + (matching_image[r,sc+2*linear_wid+c] * (1 - c / linear_wid))
                    self.result_figure[r, sc+2*linear_wid+c] = temp
        # stitch image is lower
        else:
            linear_wid = int(blending_col/5)
            sr = abs(shift_r)
            sc = abs(shift_c)
            self.result_figure[sr:h_match+sr, 0:sc+2*linear_wid] = matching_image[0:h_match, 0:sc+2*linear_wid]
            for r in range(sr,h_match):
                for c in range(0,linear_wid):
                    temp = (feature_image[r-sr,2*linear_wid+c] * (c / linear_wid)) + (matching_image[r,sc+2*linear_wid+c] * (1 - c / linear_wid))
                    self.result_figure[r, sc+2*linear_wid+c] = temp


repeat_k = 100
sample_amount = 4
matching_threshold = 0.75
threshold = 20
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
    output_dir = args.output   # eg. ../data/harris/parrington_test
    match_dir = args.match    # eg. ../data/matched/parrington
    stitch_dir = args.stitch  # eg. ../data/stitched/parrington

    # apply harris corner detection
    image_paths = image_paths_under_dir(input_dir)
    image_list, feature_list, desc_list = [], [], []
    for index, file_name in enumerate(image_paths):
        # if index < 3:
        image = cv2.imread(f'{input_dir}/{file_name}')
        image_list.append(image)
        harris = HarrisCornerDetector(image, output_path=f'{output_dir}/{file_name}')
        feature_map, feature_desc, feature_point = harris.get_feature_map(guassian_window_size, gaussian_sigma, harris_k, non_maximal_window_size, descriptor_window_size, f'{output_dir}/feature_{file_name}')
        feature_list.append(feature_point)
        desc_list.append(feature_desc)

    # match feature and match image
    match = Matching(image_list, feature_list, desc_list, output_dir=f'{match_dir}', stitch_dir=f'{stitch_dir}')
    match.process()



