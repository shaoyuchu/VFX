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
Call `ComputeBitmaps()` to get *threshold_bitmap* and *exclusion_bitmap* of the baseline image and target image. The threshold for generating black-and-white bitmap is the median value of all pixels. If the threshold value is lower than 20, the result bitmap will have lots of noises, so we set the threshold lower boundary to be 20. The exclusion bitmap sets the pixel to balck if its value is between (median - 4) to (median + 4), and sets to white otherwise.  

Shift the target image to its 9 neighbors' location with `cv2.warpAffine`, and compare the error value for each location.  
The fundamental idea is to calculate`(BaselineImage) XOR (TargetImage) AND (exclusion_bitmap)` as error value. XOR for taking difference, AND with exclusion maps to reduce noise.   

**Step 4: Shift the image to complete alignment**
Find the shift direction that creates the least amount of error, store the shifting value and pass it down the image pyramid. Repeat Step 3 until we get the shifting value for original sized image. Use `cv2.warpAffine` to align the soure images and save them to the output folder.


## HDR

## Tone mapping
