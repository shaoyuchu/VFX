import numpy as np
import cv2

def showImage(img):
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return


def intensityAdjustment(image, template):
	g, b, r = cv2.split(image)
	tg, tb, tr = cv2.split(template)
	b *= np.average(tb) / np.nanmean(b)
	g *= np.average(tg) / np.nanmean(g)
	r *= np.average(tr) / np.nanmean(r)
	image = cv2.merge((g,b,r))
	return image


def toneMapping(HDRImage):
	B, G, R = cv2.split(HDRImage)
	intensity = 1/61 * (20 * R + 40 * G + B)
	b = np.divide(B, intensity)
	g = np.divide(G, intensity)
	r = np.divide(R, intensity)

	log_intensity_layer = np.log10(intensity)
	log_base = cv2.bilateralFilter(log_intensity_layer, 9, 5, 5)
	log_detail = log_intensity_layer - log_base

	targetContrast = np.log10(5)
	compressionfactor = targetContrast/(np.max(log_base) - np.min(log_base))
	log_absolute_scale= np.max(log_base) * compressionfactor
	log_output_intensity = log_base * compressionfactor + log_detail - log_absolute_scale
	B_output = b * np.power(10, (log_output_intensity))
	G_output = g * np.power(10, (log_output_intensity))
	R_output = r * np.power(10, (log_output_intensity))


	# Normalize
	B_output = cv2.normalize(B_output, np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
	G_output = cv2.normalize(G_output, np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
	R_output = cv2.normalize(R_output, np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
	# Merge RGB channels to single image
	tone_mapping_result = cv2.merge([B_output,G_output,R_output])
	return tone_mapping_result


if __name__ == '__main__':
	inputFile = './output_images/Memorial_SourceImages'
	HDRImage = cv2.imread(inputFile + '/official_hdr_result.hdr', flags = cv2.IMREAD_ANYDEPTH)
	tone_mapping_result = toneMapping(HDRImage)
	cv2.imwrite("tone_mapping_result.jpg", tone_mapping_result)

	# OpenCV tone mapping
	# tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
	# ldrReinhard = tonemapReinhard.process(HDRImage)
	# cv2.imwrite("ldrReinhard.jpg", ldrReinhard * 255)