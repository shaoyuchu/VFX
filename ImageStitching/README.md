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

### Warp to cylinder coordinate

Let $f$ denote the focal length, and $x_c$, $y_c$ denote the x, y coordinate of the image center. We warp the image by reproject them onto a cylinder with the following formula:
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

### Feature detection

### Feature matching

### Image matching

### Blending

### Result