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
    print(f'{len(image_paths)} images in total\n')

    all_B = np.array(all_B)
    all_G = np.array(all_G)
    all_R = np.array(all_R)
    return all_B, all_G, all_R


class HDR:

    def __init__(self, shutter_speed, values):
        # self.ln_shutter_speed[i]: ln(shutter speed) of the i-th image
        # self.values[i, j, k]: the intensity of pixel (j, k) in the i-the image
        self.ln_shutter_speed = np.log(shutter_speed)
        self.values = values
        self.n_image = values.shape[0]
        self.height = values.shape[1]
        self.width = values.shape[2]
    
    def __close_to_center(self, rs, cs, radius):
        # compute the distances to the center
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
            # dist = np.abs(r - center_r) + np.abs(c - center_c)
            dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
            if dist < thresh:
                return r, c
            elif dist < best_dist:
                best_dist = dist
                best_idx = index
        return rs[best_idx], cs[best_idx]

    def __construct_Z(self):
        # self.z[i, j] = value of the j-th sampled pixel in the i-th image
        self.z = np.zeros((self.n_image, self.n_sample), dtype=np.uint8)
        for img_idx in range(self.n_image):
            for sample_idx in range(self.n_sample):
                self.z[img_idx, sample_idx] = self.values[img_idx, self.sample_pts[sample_idx][0], self.sample_pts[sample_idx][1]]
    
    def __weight(self, value, z_min=0, z_max=255):
        if value < z_min or value > z_max:
            raise RuntimeError(f'Invalid value {value}.')
        return min(value - z_min, z_max - value)

    def sample_pixels(self, n_sample, sample_radius=0.8, plot_hist=False, visualize_sample_pt=False):
        # compute the variance of values in all images
        variances = np.array([])
        for i in range(self.n_image):
            all_value = self.values[i, :, :]
            variances = np.append(variances, np.var(all_value))
        
        # select the image with the largest variance
        self.sample_img_idx = np.argmax(variances)
        print(f'Sample pixels based on image with index {self.sample_img_idx}')

        # compute the values with evenly spread percentiles
        percentiles = np.linspace(0, 100, num=n_sample)
        sample_values = np.percentile(self.values[self.sample_img_idx], percentiles, interpolation='nearest')
        print(f'Sampled pixel values:\n{sample_values}\n')
        if plot_hist: 
            plt.clf()
            plt.hist(self.values[self.sample_img_idx].reshape(-1), bins=np.arange(0, 260, 5))
            for i in range(len(sample_values)):
                plt.axvline(sample_values[i], color='k', linestyle='dashed', linewidth=0.5)
            plt.show()

        # select pixels with both high and low values and avoid those on the edges
        self.n_sample = n_sample
        self.sample_pts = []
        for value in sample_values:
            occurances = np.where(self.values[self.sample_img_idx] == value)
            r, c = self.__close_to_center(occurances[0], occurances[1], sample_radius)
            self.sample_pts.append((r, c))
        if visualize_sample_pt:
            plt.clf()
            for r, c in self.sample_pts:
                plt.scatter(c, r, color='red', s=1)
            plt.imshow(self.values[self.sample_img_idx])
            plt.show()
        
        # construct self.z
        self.__construct_Z()
        print(f'Z:\n{self.z}\n')
    
    def response_curve(self, smoothing_lambda, visualize_g=False):
        mat_A = np.zeros((self.n_image*self.n_sample + 255, 256 + self.n_sample), dtype=np.float64)
        mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)
        print('mat_A.shape:', mat_A.shape)
        print('mat_b.shape:', mat_b.shape)

        cur_r = 0
        # data-fitting constraints
        for i in range(self.n_sample):
            for j in range(self.n_image):
                # i-th sampled pixel in the j-th image
                z = self.z[j, i]
                w = self.__weight(z)
                mat_A[cur_r, z] = w
                mat_A[cur_r, 256 + i] = -1 * w
                mat_b[cur_r] = w * self.ln_shutter_speed[j]
                cur_r += 1

        # curve centering constraint
        mat_A[cur_r, 127] = 1
        cur_r += 1

        # smoothing constraints
        for z in range(1, 255):
            w = self.__weight(z)
            mat_A[cur_r, z-1] = w * smoothing_lambda
            mat_A[cur_r, z] = -2 * w * smoothing_lambda
            mat_A[cur_r, z+1] = w * smoothing_lambda
            cur_r += 1

        # solve the linear system
        print('Computing the response curve...')
        pseudo_inv_A = np.linalg.pinv(mat_A)
        x = np.dot(pseudo_inv_A, mat_b)
        inv_response_curve = x[:256, 0]

        # visualization
        if visualize_g:
            plt.clf()
            plt.scatter(inv_response_curve, np.arange(0, 256))
            plt.title('Response Curve')
            plt.xlabel('log exposure (E_i * (delta t)_j)')
            plt.ylabel('pixel value (Z_ij)')
            plt.show()

        return inv_response_curve
        




# V read images
# V sample Eij
# construct linear system
# solve linear system, get inverse of response curve
# construct HDR radiance map
# save as .hdr

n_sample_pt = 50
sample_radius = 0.8
smoothing_lambda = 50

if __name__ == '__main__':

    # read images
    input_image_dir = './input_images/corner/'
    input_image_paths = os.listdir(input_image_dir)
    input_image_paths = list(map(lambda f: input_image_dir+f, input_image_paths))
    input_image_paths = list(filter(lambda f: (f.endswith('.JPG') or f.endswith('.png')), input_image_paths))
    all_b, all_g, all_r = read_images(input_image_paths)
    
    # parse shutter speed
    shutter = np.array(list(map(parse_shutter_speed, input_image_paths)))
    print('Shutter speed:\n', shutter, '\n')

    # HDR for channel G
    hdr = HDR(shutter, all_g)
    hdr.sample_pixels(n_sample_pt, sample_radius=sample_radius, plot_hist=False, visualize_sample_pt=False)
    hdr.response_curve(smoothing_lambda, visualize_g=True)

    
