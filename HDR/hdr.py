import os
import numpy as np
import cv2
from matplotlib import image
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS

def parse_exposure_time(image_paths):
    # eg. './bridge/1_8.JPG' -> 0.125
    all_exposure_time = []
    for path in image_paths:
        segments = path.replace('.', '_').replace('/', '_').split('_')
        all_exposure_time.append(int(segments[-3]) / int(segments[-2]))
    all_exposure_time = np.array(all_exposure_time)
    return all_exposure_time

def get_exposure_time_from_metadata(image_paths):
    exposure_id = [tag_id for tag_id, tag_name in TAGS.items() if tag_name == 'ExposureTime'][0]
    all_exposure_time = []
    for path in image_paths:
        image = Image.open(path)
        exifdata = image.getexif()
        all_exposure_time.append(float(exifdata[exposure_id]))
    all_exposure_time = np.array(all_exposure_time)
    return all_exposure_time

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

# ---------------------------------------- HDR class ----------------------------------------
class HDR:

    def __init__(self, exposure_time, values):
        # self.ln_exposure_time[i]: ln(exposure time) of the i-th image
        # self.values[i, j, k]: the intensity of pixel (j, k) in the i-the image
        self.ln_exposure_time = np.log(exposure_time)
        self.values = values
        self.n_image = values.shape[0]
        self.height = values.shape[1]
        self.width = values.shape[2]
        print(f'Create HDR image with {self.values.shape[0]} images of shape ({self.height}, {self.width})')
        print('ln exposure time:')
        print(self.ln_exposure_time)
    
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

    def sample_pixels(self, n_sample, sample_radius=0.8, show_hist=False, show_sample_pt=False):
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
        if show_hist: 
            plt.clf()
            plt.hist(self.values[sample_img_idx].reshape(-1), bins=np.arange(0, 260, 5))
            for i in range(len(sample_values)):
                plt.axvline(sample_values[i], color='k', linestyle='dashed', linewidth=0.5)
            plt.show()

        # select pixels with both high and low values and avoid those on the edges
        self.n_sample = n_sample
        self.sample_pts = []
        for value in sample_values:
            occurances = np.where(self.values[sample_img_idx] == value)
            r, c = self.__close_to_center(occurances[0], occurances[1], sample_radius)
            self.sample_pts.append((r, c))
        if show_sample_pt:
            plt.clf()
            for r, c in self.sample_pts:
                plt.scatter(c, r, color='red', s=1)
            plt.imshow(self.values[sample_img_idx])
            plt.show()
        
        # construct self.z
        self.__construct_Z()
    
    def compute_inv_response_curve(self, smoothing_lambda, visualize_g=False):
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
                mat_b[cur_r] = w * self.ln_exposure_time[j]
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
        pseudo_inv_A = np.linalg.pinv(mat_A)
        x = np.dot(pseudo_inv_A, mat_b)
        self.inv_response_curve = x[:256, 0]

        # visualization
        if visualize_g:
            plt.clf()
            plt.scatter(self.inv_response_curve, np.arange(0, 256))
            plt.title('Response Curve')
            plt.xlabel('log exposure (E_i * (delta t)_j)')
            plt.ylabel('pixel value (Z_ij)')
            plt.show()

        return self.inv_response_curve
    
    def radiance_map(self, show_radiance=False):
        print('Computing radiance map...\n')

        # compute weighted average of irradiance
        w = np.vectorize(self.__weight)(self.values)
        ln_t = np.array(list(map(lambda t: np.full_like(self.values[0], t, dtype=np.float64), self.ln_exposure_time)))
        g_z = np.vectorize(lambda v: self.inv_response_curve[v])(self.values)
        ln_E = g_z - ln_t
        weighted_sum = np.mean(w * ln_E, axis=0)
        total_weight = np.mean(w, axis=0)
        self.ln_irradiance = np.zeros((self.height, self.width), dtype=np.float64)      
        for r in range(self.height):
            for c in range(self.width):
                if total_weight[r, c] == 0:
                    self.ln_irradiance[r, c] = np.mean(ln_E[:, r, c])
                else:
                    self.ln_irradiance[r, c] = weighted_sum[r, c] / total_weight[r, c]
        self.irradiance = np.exp(self.ln_irradiance)

        # plot radiance map
        if show_radiance:
            plt.clf()
            plt.imshow(self.ln_irradiance, cmap='rainbow')
            plt.colorbar()
            plt.show()

# ---------------------------------------- end of HDR class ----------------------------------------

input_image_dir = './input_images/'
output_image_path = './output_images/'
image_type = 'My_Images_aligned'
n_sample_pt = 51
sample_radius = 0.8
smoothing_lambda = 100
plot_all_response_curves = True
plot_all_radiance_map = True
exposure_from_metadata = False

if __name__ == '__main__':

    # read images
    input_image_paths = os.listdir(f'{input_image_dir}{image_type}/')
    input_image_paths = list(map(lambda f: f'{input_image_dir}{image_type}/{f}', input_image_paths))
    input_image_paths = list(filter(lambda f: (f.endswith('.JPG') or f.endswith('.png')), input_image_paths))
    all_b, all_g, all_r = read_images(input_image_paths)
    
    # parse exposure time
    if exposure_from_metadata:
        exposure = get_exposure_time_from_metadata(input_image_paths)
    else:
        exposure = parse_exposure_time(input_image_paths)
    print(exposure)

    # HDR compute response curves and radiance map
    all_inv_response_curves = []
    all_ln_irradiance = []
    all_irradiance = []
    color_channels = ['blue', 'green', 'red']
    for channel, color in [(all_b, color_channels[0]), (all_g, color_channels[1]), (all_r, color_channels[2])]:
        print(f'Computing channel {color}...')
        hdr = HDR(exposure, channel)
        hdr.sample_pixels(n_sample_pt, sample_radius=sample_radius, show_hist=False, show_sample_pt=False)
        hdr.compute_inv_response_curve(smoothing_lambda, visualize_g=False)
        hdr.radiance_map(show_radiance=False)
        all_inv_response_curves.append(hdr.inv_response_curve)
        all_ln_irradiance.append(hdr.ln_irradiance)
        all_irradiance.append(hdr.irradiance)
    
    # create directory if not exist
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)
    sub_directory = f'{output_image_path}{image_type}/'
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)

    # save .hdr image
    result_file_name = f'{output_image_path}{image_type}/hdr_result.hdr'
    height, width = all_irradiance[0].shape
    rgb = np.stack(tuple(all_irradiance), axis=-1)
    cv2.imwrite(result_file_name, rgb)
    print(f'{result_file_name} saved')
    
    # plot response curves
    if plot_all_response_curves:
        plt.clf()
        plt.title('Response Curves')
        plt.xlabel('log exposure (E_i * (delta t)_j)')
        plt.ylabel('pixel value (Z_ij)')
        for i in range(3):
            plt.plot(all_inv_response_curves[i], np.arange(256), label=color_channels[i], c=color_channels[i])
        plt.legend()
        file_name = f'{output_image_path}{image_type}/response_curves.png'
        plt.savefig(file_name)
        print(f'{file_name} saved')
    
    # plot radiance maps
    all_ln_irradiance = np.array(all_ln_irradiance)
    v_min = np.min(all_ln_irradiance)
    v_max = np.max(all_ln_irradiance)
    if plot_all_radiance_map:
        for i in range(3):
            plt.clf()
            plt.imshow(all_ln_irradiance[i], cmap='rainbow', vmin=v_min, vmax=v_max)
            plt.colorbar()
            plt.axis('off')
            plt.title(color_channels[i])
            file_name = f'{output_image_path}{image_type}/radiance_map_{color_channels[i]}.png'
            plt.savefig(file_name)
            print(f'{file_name} saved')
