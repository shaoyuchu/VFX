import os
import numpy as np
import cv2
from matplotlib import image
import matplotlib.pyplot as plt

def parse_shutter_speed(filename):
    # eg. './bridge/1_8.JPG' -> 0.125
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
    
    def __close_to_center(self, rs, cs, radius):
        # compute the Manhattan distances to the center
        # select one random point with dist < min(height, width)*radius or the one closest to the center
        center_r = self.height / 2
        center_c = self.width / 2
        thresh = np.min([center_r, center_c]) * radius

        # shuffle
        permutation = np.random.permutation(rs.shape[0])
        rs = rs[permutation]
        cs = cs[permutation]
        
        # search
        best_dist = np.inf
        best_idx = -1
        for index, (r, c) in enumerate(zip(rs, cs)):
            dist = np.abs(r - center_r) + np.abs(c - center_c)
            if dist < thresh:
                return r, c
            elif dist < best_dist:
                best_dist = dist
                best_idx = index
        return rs[best_idx], cs[best_idx]
    
    def sample_pixels(self, n_sample, sample_radius=0.8, plot_hist=False, visualize_sample_pt=False):
        # compute the variance of values in all images
        variances = np.array([])
        for i in range(self.n_image):
            all_value = self.values[i, :, :]
            variances = np.append(variances, np.var(all_value))
        
        # select the image with the largest variance
        sample_img_idx = np.argmax(variances)

        # compute the values with evenly spread percentiles
        percentiles = np.linspace(0, 100, num=n_sample)
        sample_values = np.percentile(self.values[sample_img_idx], percentiles, interpolation='nearest')
        if plot_hist:
            plt.clf()
            plt.hist(self.values[sample_img_idx].reshape(-1), bins=np.arange(0, 260, 5))
            for i in range(len(sample_values)):
                plt.axvline(sample_values[i], color='k', linestyle='dashed', linewidth=0.5)
            plt.show()

        # select pixels with both high and low values and avoid those on the edges
        self.sample_pts = []
        for value in sample_values:
            occurances = np.where(self.values[sample_img_idx] == value)
            r, c = self.__close_to_center(occurances[0], occurances[1], sample_radius)
            self.sample_pts.append((r, c))
        if visualize_sample_pt:
            plt.clf()
            for r, c in self.sample_pts:
                plt.scatter(c, r, color='red', s=1)
            plt.imshow(self.values[sample_img_idx])
            plt.show()




# read images
# sample Eij
# construct linear system
# solve linear system, get inverse of response curve
# construct HDR radiance map
# save as .hdr

n_sample_pt = 50
sample_radius = 0.8

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
    hdr.sample_pixels(n_sample_pt, sample_radius=sample_radius, visualize_sample_pt=True)



    
