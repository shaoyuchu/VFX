# project #1: High Dynamic Range Imaging
### Group member
- 資管四 B06705028 朱紹瑜
- 資工所碩一 R09922063 鄭筠庭

## Image Alignment:  Median Threshold Bitmap (MTB)
Image alignment methods: Ward's MTB algorithm  
Reference paper：http://www.anyhere.com/gward/papers/jgtpap2.pdf  

**Step 1: Get grayscale images**  
Use `cv2.imread` to read in images with different exposure time. The name of each image is its corresponding shutter time. Turn the images into gray scale by Y=(54R+183G+19B)/256).  

**Step 2: Build an image pyramid**  
Run the recursive `GetExpShift()` function. This recursive function will utilze multi-scale technique and generate an image pyramid with log2(max_offset) levels past the base resolution. For each smaller level in the pyramid, we filter the previous grayscale image down by a factor of two in each dimension.

**Step 3: Find the minimal error between 9 neighbors**  
Here we use the first image as baseline.  
Call `ComputeBitmaps()` to get *threshold_bitmap* and *exclusion_bitmap* of the baseline image and target image.


## HDR

## Tone mapping