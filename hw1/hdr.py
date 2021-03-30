import os
import numpy as np
import cv2
from matplotlib import image
import matplotlib.pyplot as plt

def parse_shutter_speed(filename):
    # eg. '1_8.JPG' -> 0.125
    segments = filename.replace('.', '_').replace('/', '_').split('_')
    return int(segments[-3]) / int(segments[-2])

def read_images(image_paths):
    all_B, all_G, all_R = [], [], []
    for path in image_paths:
        image = cv2.imread(path)
        all_B.append(image[:, :, 0])
        all_G.append(image[:, :, 1])
        all_R.append(image[:, :, 2])
        print(f'Reading {path}: {image.shape}')

    all_B = np.array(all_B)
    all_G = np.array(all_G)
    all_R = np.array(all_R)
    return all_B, all_G, all_R


class HDR:

    def __init__(self, shutter_speed, values):
        # self.shutter_speed[i]: the shutter speed of the i-th image
        # self.values[i, j, k]: the intensity of pixel (j, k) in the i-the image
        self.shutter_speed = shutter_speed
        self.values = values
        self.n_image = values.shape[0]
        self.height = values.shape[1]
        self.width = values.shape[2]
    
    def sample_pixels(self, n_sample):
        # compute the variance of values in all images
        variances = np.array([])
        for i in range(self.n_image):
            all_value = self.values[i, :, :]
            variances = np.append(variances, np.var(all_value))
        
        # select the image with the largest variance
        self.sample_img_idx = np.argmax(variances)

        # select pixels with both high and low values
        percentiles = np.linspace(0, 100, num=n_sample)
        sample_values = np.percentile(self.values[self.sample_img_idx], percentiles, interpolation='nearest')
        self.sample_pt = []
        for value in sample_values:
            occurances = np.where(self.values[self.sample_img_idx] == value)
            r, c = self.closest_to_center(occurances[0], occurances[1])
            self.sample_pt.append((r, c))
        print(self.sample_pt)
        
        for r, c in self.sample_pt:
            plt.scatter(c, r, color='red', s=2)
        plt.imshow(self.values[self.sample_img_idx])
        plt.show()

        # # visualize sampled values
        # plt.hist(self.values[self.sample_img_idx].reshape(-1), bins=np.arange(0, 260, 5))
        # for i in range(len(sample_values)):
        #     plt.axvline(sample_values[i], color='k', linestyle='dashed', linewidth=1)
        # plt.show()
    
    def closest_to_center(self, rs, cs):
        center_h = self.height / 2
        center_w = self.width / 2
        best_dist = np.inf
        best_idx = -1
        for index, (r, c) in enumerate(zip(rs, cs)):
            dist = np.abs(r - center_h) + np.abs(c - center_w)
            if dist < best_dist:
                best_dist = dist
                best_idx = index
        return rs[best_idx], cs[best_idx]


            



# read images
# sample Eij
# construct linear system
# solve linear system, get inverse of response curve
# construct HDR radiance map
# save as .hdr

n_sample_pt = 50

if __name__ == '__main__':

    # read images
    input_image_dir = './input_images/bridge/'
    input_image_paths = os.listdir(input_image_dir)
    input_image_paths = list(map(lambda f: input_image_dir+f, input_image_paths))
    input_image_paths = list(filter(lambda f: f.endswith('.JPG'), input_image_paths))
    all_b, all_g, all_r = read_images(input_image_paths)
    
    # parse shutter speed
    shutter = np.array(list(map(parse_shutter_speed, input_image_paths)))
    print('shutter speed: ', shutter)

    # HDR of channel G
    hdr = HDR(shutter, all_g)
    hdr.sample_pixels(n_sample_pt)



    
