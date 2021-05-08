# Project #2: Image Stitching

- 資管四 B06705028 朱紹瑜
- 資工所碩一 R09922063 鄭筠庭

### Introduction

#### Usage

Run the following command to reproduce the result.

```python
python3 cylindrical_warping.py ../data/input/library ../data/warped/library
python3 match.py ../data/warped/library ../data/harris/library../data/matched/parrington ../data/stitched/library
```

`cylindrical_warping.py` warps the images to the cylinder coordinate.

`match.py` finds the feature points on a image and saves it to the "harris/library" directory. Then, it finds the matching feature point pairs and saves it to the "matched/library" directory. Finally, the code stitches the images together with the matching feature data, and saves it to the "stitched/library" directory.  

The final panorama image is called result.jpg, and is saved in the "stitched/library" directory.

### Warp to Cylinder Coordinate

Let $f$ denote the focal length, and $x_c$, $y_c$ denote the x, y coordinate of the image center. We warp the image by reprojecting them onto a cylinder with the following formula:
$$
x^\prime = f \cdot \tan^{-1}(\frac{x-x_c}{f}) + x_c \\
y^\prime = f \cdot \frac{y - y_c}{\sqrt{(x-x_c)^2 + f^2}} + y_c
$$
To avoid "holes" in the resulting images, we implement inverse warping. The formula becomes
$$
x = f \cdot \tan(\frac{x^\prime - x_c}{f}) + x_c \\
y = \frac{y^\prime - y_c}{f} \cdot \sqrt{(x - x_c)^2+f^2} + y_c.
$$
For example, the warped image is shown below.

|               Original               |                Warped                |
| :----------------------------------: | :----------------------------------: |
| ![](https://i.imgur.com/pFcVdck.jpg) | ![](https://i.imgur.com/MBmhiSG.jpg) |

### Feature Detection

We implement Harris corner detection to capture feature points.

**Step 1**: Blur the image with a Gaussian kernel.

**Step 2**: Compute $x$ and $y$ derivatives, $I_x$ and $I_y$.

**Step 3**: Compute the products of derivatives at each pixel, $I_{xx}$, $I_{yy}$, and $I_{xy}$.

**Step 4**: Apply Gaussian convolution over the products of derivatives with the following formula:
$$
S_{xx} = G_\sigma * I_{xx} \quad \quad S_{yy} = G_\sigma * I_{yy} \quad \quad S_{xy} = G_\sigma * I_{xy}.
$$
**Step 5**: Compute the response of the detector of each pixel, where
$$
R = \det M - k(\text{trace} M)^2
$$
with
$$
M(x, y) = 
\begin{bmatrix}
	S_{xx}(x, y) & S_{xy}(x, y) \\
	S_{xy}(x, y) & S_{yy}(x, y)
\end{bmatrix}.
$$
Here, we select $k$ to be $0.05$.

**Step 6**:  Get candidate features by thresholding on the response of the detector. Here we set the threshold as the 99-percentile of the response.

**Step 7**: To avoid clustered feature points, apply non-maximal suppression. Candidate features are selected to be feature points only if the corresponding response is higher than that of all pixels in the window.

For example, the detected feature points are labeled red in the image below.

|                 Features                  |
| :---------------------------------------: |
| ![image](https://i.imgur.com/kgxQHSm.jpg) |

### Feature Descriptor

We chose SIFT descriptor to store our feature data since it is rotation invariant. For each feature point, find the dominant orientation of a window around the keypoint. Calculate the neighbor histogram of gradients within `descriptor_window_size`. Aligns the window to that orientation by subtracting this dominant orientation from all other orientations in the window. In this way the keypoint is "orientation invariant".

![image](https://i.imgur.com/FsN8rAX.png)

**Step 1**: Filter the image with a kernel to find the horizontal and vertical gradients:
$$
g_x = image[r+1,c] - image[r-1,c] \\
g_x = image[r,c+1] - image[r,c-1] \\
$$
Calculate the gradient value and angle of each pixel:
$$
g = \sqrt{g_x^2+g_y^2}\\
\theta = \arctan \frac{g_y} {g_x}
$$
**Step 2**: Subtract the angle of each pixel in the window with feature point angle (dominate angle).

**Step 3**: Divide the window into a 4x4 array. Use a 8 orientation bins to save the gradient histogram in each section. A gradients’s contribution is divided among the nearby histograms based on distance. If 
it’s halfway between two histogram locations, it gives a percentage of contribution to both orientation.

![image](https://i.imgur.com/6IHVcr4.png)

For more detail description, here is a reference slide: http://vision.stanford.edu/teaching/cs131_fall1718/files/07_DoG_SIFT.pdf



### Feature Matching

Go through every image and compare the feature descriptor of current image with the next image using `np.linalg.norm`.  We aimed to find the matched feature points with the minimal difference.

To reduce the amount of matching pairs, minimal match point difference needs to be 0.75 times smaller than second minimal match point difference.

### Image Matching

### Blending

### Result