import numpy as np
import cv2

# def toneMapping(HDRImage):
	# intensity = 1/61*(R*20+G*40+B)


if __name__ == '__main__':
	inputFile = './output_images/SocialScienceLibrary_aligned'
	HDRImage = cv2.imread(inputFile + '/hdr_result.hdr', cv2.IMREAD_ANYDEPTH)
	# print(HDRImage.shape)
	tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
	ldrReinhard = tonemapReinhard.process(HDRImage)
	cv2.imwrite("ldrReinhard.jpg", ldrReinhard * 255)