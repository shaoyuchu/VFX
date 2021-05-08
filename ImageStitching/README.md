# Project #2: Image Stitching

- 資管四 B06705028 朱紹瑜
- 資工所碩一 R09922063 鄭筠庭

### Introduction

#### Usage

Run the following command to reproduce the result.

```python
python3 cylindrical_warping.py ../data/input/parrington ../data/warped/parrington
python3 matching.py ../data/warped/parrington ../data/harris/parrington ../data/matched/parrington ../data/stitched/parrington
```

`cylindrical_warping.py` warps the images to the cylinder coordinate.

`matching.py` finds the feature points on a image and saves it to the "harris/parrington" directory. Then, it finds the matching feature point pairs and saves it to the "matched/parrington" directory. Finally, the code stitches the images together with the matching feature data, and saves it to the "stitched/parrington" directory.  

The final panorama image is called result.jpg, and is saved in the stitched/parrington directory.

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

| Original | Warped |
| :------: | :----: |
|          |        |

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

| Features |
| :------: |
|          |

### Feature Descriptor

We chose SIFT descriptor to store our feature data since it is rotation invariant. 

![image](https://i.imgur.com/FsN8rAX.png)

### Feature Matching

Go through every image and compare the feature descriptor of current image with the next image using `np.linalg.norm`.  We aimed to find the matched feature points with the minimal difference.

To reduce the amount of matching pairs, minimal match point difference needs to be 0.75 times smaller than second minimal match point difference.

### Image Matching

### Blending

### Result