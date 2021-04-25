import cv2
import numpy as np
import glob
import os
from PIL import Image
from fractions import Fraction

def readImages(folder):
	imageFormat = ['png', 'JPG']
	files = []
	# Read in images
	[files.extend(glob.glob(folder + '*.' + e)) for e in imageFormat]
	imageList = [cv2.imread(file) for file in sorted(files)]

	# Get image exposure time from metadata
	exposureTimeList = []
	images = glob.glob(folder+"*.JPG")
	for image in sorted(images):
		with open(image, 'rb') as file:
			tempImage = Image.open(file)
			exifdata = tempImage.getexif()
			#  0x829A: "ExposureTime"
			dataValue = exifdata.get(0x829A)
			if isinstance(dataValue, bytes):
				dataValue = dataValue.decode()
			dataValue = Fraction(dataValue).limit_denominator()
			print(dataValue)

			if dataValue>=1:
				dataValueString = str(dataValue)+'_1.JPG'
			else:
				dataValueArray = str(dataValue).split('/')
				dataValueString = str(dataValueArray[0]) + '_' + str(dataValueArray[1]) + '.JPG'
			exposureTimeList.append(dataValueString)
	print(exposureTimeList)

	# # Get image names. Will use these names when saving result images
	# imageNames = []
	# for name in sorted(files):
	# 	name = (name.split('/')[2])
	# 	imageNames.append(name)

	grayscaleList = turnGrayscale(imageList)
	return imageList, grayscaleList, exposureTimeList




# Show image on screen
def showImage(img):
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return


def turnGrayscale(imageList):
	grayscaleList = []
	count = 0
	for img in imageList:
		grayscaleList.append(np.dot(img[...,:3], [19/256, 183/256, 54/256]).astype(np.uint8))
		count += 1
	return grayscaleList

# -------------- Median Threshold Bitmap ------------------
def ComputeBitmaps(img):
	threshold_bitmap = np.zeros(img.shape, np.uint8)
	threshold = np.median(img)
	if threshold < 20.0:
			threshold = 20.0
	for row in range(len(img)):
		for column in range(len(img[0])):
			if img[row][column] > threshold:
				threshold_bitmap[row][column] = 255

	exclusion_bitmap = cv2.inRange(img, np.median(img) - 4, np.median(img) + 4)
	np.invert(exclusion_bitmap)

	return threshold_bitmap, exclusion_bitmap


def imageShrink(img):
	return cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)


def GetExpShift(img1, img2, shift_bits):
	cur_shift = np.zeros(2)
	if shift_bits > 0:
		sml_img1 = imageShrink(img1)
		sml_img2 = imageShrink(img2)
		cur_shift = GetExpShift(sml_img1, sml_img2, shift_bits - 1)
		cur_shift[0] *= 2
		cur_shift[1] *= 2
	else:
		cur_shift[0] = cur_shift[1] = 0
	tb1, eb1 = ComputeBitmaps(img1)
	tb2, eb2 = ComputeBitmaps(img2)
	min_err = len(img1) * len(img1[0])
	shift_ret = np.zeros(2)
	for i in range(-1,2):
		for j in range(-1,2):
			xs = cur_shift[0] + i
			ys = cur_shift[1] + j
			M = np.float32([[1, 0, xs], [0, 1, ys]])
			shifted_tb2 = np.zeros(tb2.shape)
			shifted_eb2 = np.zeros(eb2.shape)
			tb_rows, tb_cols = tb2.shape[:2]
			shifted_tb2 = cv2.warpAffine(tb2, M, (tb_cols, tb_rows))
			shifted_eb2 = cv2.warpAffine(eb2, M, (tb_cols, tb_rows))
			diff_b = np.logical_xor(tb1, shifted_tb2)
			diff_b = np.logical_and(diff_b, eb1)
			diff_b = np.logical_and(diff_b, shifted_eb2)
			err = np.sum(diff_b == 255)
			if (err < min_err):
				shift_ret[0] = xs
				shift_ret[1] = ys
				min_err = err
	return shift_ret


def MedianThreshold(sourceImages, imageList, imageNames, outputFileDirectory):
	alignedImageList = []
	img_rows, img_cols = imageList[0].shape[:2]
	for img, sourceimg, imgName in zip(imageList, sourceImages, imageNames):
		shift_step = GetExpShift(imageList[0], img, 3)
		M = np.float32([[1, 0, shift_step[0]], [0, 1, shift_step[1]]])
		tempImage = cv2.warpAffine(sourceimg, M, (img_cols, img_rows))
		alignedImageList.append(tempImage)

		cv2.imwrite(outputFileDirectory + '/' + imgName, tempImage)
		print('Image ' + imgName + ' complete')
	return alignedImageList
# -------------- End of Median Threshold Bitmap ------------------

if __name__ == '__main__':
	inputDirectory = './input_images/My_Images'
	sourceImages, grayImages, imageNames = readImages(inputDirectory + '/')

	# Check if output directory exist. If not exist, create one
	outputFileDirectory = './' + inputDirectory + '_Aligned'
	if os.path.isdir(outputFileDirectory) != True:
		os.mkdir(outputFileDirectory)

	MedianThreshold(sourceImages, grayImages, imageNames, outputFileDirectory)