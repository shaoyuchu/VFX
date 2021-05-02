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

`cylindrical_warping.py` warps the images to a cylinder.

`matching.py` finds the feature points on a image and saves it to the "harris/parrington" directory. Then, it finds the matching feature point pairs and saves it to the "matched/parrington" directory. Finally, the code stitches the images together with the matching feature data, and saves it to the "stitched/parrington" directory.  

The final panorama image is called result.jpg, and is saved in the stitched/parrington directory.

### Warp to cylinder

### Feature detection

### Feature matching

### Image matching

### Blending

### Result