import numpy as np
import cv2
import argparse
import random
import math
from util import *
from harris import *

class Matching:
    def __init__(self, images, features, descriptors, output_dir=None):
        self.images = images
        self.features = features
        self.descriptors = descriptors
        self.image_amount = len(descriptors)
        self.output_dir = output_dir

    def process(self):
        # go through each image
        for index, image in enumerate(self.images):
            if index == (self.image_amount-1): # skip last image
                break
            self.feature_matching(self.features[index], self.features[index+1], self.descriptors[index], self.descriptors[index+1])
            draw_match_line(self.images[index], self.images[index+1], self.match_list, self.features[index], path=f'{self.output_dir}/{index}')
            self.Ransac(self.match_list, self.features[index])

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
                if (min_dist/second_min_dist) <= 0.75:
                    self.match_list[i] = [int(match_r), int(match_c)]

    def Ransac(self, next_img_match, img_feature):
        min_error, max_inlier = float("inf"), float("-inf")
        for k in range(repeat_k):
            if k%10 == 0:
                print("Epoch " + str(k))

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
            error, inlier = self.get_error(H, next_img_match, img_feature)
            if error < min_error:
                min_error = error
                max_inlier = inlier
                best_H = H.copy()

        print(min_error, max_inlier)
            



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
        for i in range(len(next_img_match)):
            if next_img_match[i][0] == 0 and next_img_match[i][1] == 0:
                continue
            prime = np.dot(H, xy_match_points[i])
            estimate_point[i] = prime[0:2]
            err = math.sqrt((next_img_match[i][0] - estimate_point[i][0])**2 + (next_img_match[i][1] - estimate_point[i][1])**2 )
            if err < threshold:
                inliers += 1
            errors += err
        return errors, inliers


repeat_k = 100
sample_amount = 4
threshold = 30
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
        if index < 2:
            image = cv2.imread(f'{input_dir}/{file_name}')
            image_list.append(image)
            harris = HarrisCornerDetector(image, output_path=f'{output_dir}/{file_name}')
            feature_map, feature_desc, feature_point = harris.get_feature_map(guassian_window_size, gaussian_sigma, harris_k, non_maximal_window_size, descriptor_window_size, f'{output_dir}/feature_{file_name}')
            feature_list.append(feature_point)
            desc_list.append(feature_desc)

    # match feature and match image
    match = Matching(image_list, feature_list, desc_list, output_dir=f'{match_dir}')
    match.process()



