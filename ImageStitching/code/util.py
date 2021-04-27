import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def image_paths_under_dir(input_dir):
    image_paths = os.listdir(input_dir)
    image_paths = list(filter(lambda f: (f.endswith('.JPG') or f.endswith('.jpg') or f.endswith('.png')), image_paths))
    image_paths.sort()
    return image_paths

def mk_parent_dir(path):
    if path is None:
        return
    directory = path.rsplit('/', 1)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_img(path, image):
    if path is None:
        return
    cv2.imwrite(path, image)
    print(f'{path} saved')

def show_img(window_name, image):
    uint_img = image.astype(np.uint8)
    cv2.imshow(window_name, uint_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mark_on_img(image, map, path=None):
    # mark
    plt.clf()
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if map[r, c]:
                plt.scatter(c, r, color='r', s=0.5)
    plt.imshow(image, cmap='gray')

    # save or show
    if path is not None:
        plt.savefig(path, dpi=300)
        print(f'{path} saved')
    else:
        plt.show()
