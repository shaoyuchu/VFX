import numpy as np
import cv2

def showImage(img):
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return

def intensityAdjustment(input_image, template):
	g, b, r = cv2.split(input_image)
	tg, tb, tr = cv2.split(template)
	b *= np.average(tb) / np.nanmean(b)
	g *= np.average(tg) / np.nanmean(g)
	r *= np.average(tr) / np.nanmean(r)
	input_image = cv2.merge((g,b,r))
	return input_image

# -------------- Bilateral Filter ------------------
def gaussianFunction(x, standard_deviation):
	pi = np.pi
	sd = standard_deviation
	return (1 / (sd * (2 * pi) ** (1 / 2))) * np.exp(-(x ** 2 / (2 * sd ** 2)))


def bilateralFilter(input_image, radius):
	height, width = input_image.shape[:2]
	output_image = np.zeros(input_image.shape)
	for h in range(height):
		print("bilateralFilter: "+str(h))
		for w in range(width):
			output_image[h,w] = pixelBilateralFilter(input_image, h, w, radius)
	return output_image

def pixelBilateralFilter(input_image, height, width, r):
	spatial_domain = 100
	range_domain = 100
	denoised_intensity_fraction, denoised_intensity_denominator = 0, 0
	# Search the pixels within the radius -- r
	for k in range(height-r, height+r+1):
		for l in range(width-r, width+r+1):
			# Check if the pixel is located outside the border
			if k >= height or k < 0 or l >= width or l < 0:
				continue
			ws = gaussianFunction((height-k)**2 - (width-l)**2, spatial_domain)
			wr = gaussianFunction((input_image[height, width] - input_image[k,l]), range_domain)
			w = ws * wr
			denoised_intensity_fraction += w*input_image[k,l]
			denoised_intensity_denominator += w

	denoised_intensity = np.divide(denoised_intensity_fraction, denoised_intensity_denominator)
	return denoised_intensity

# -------------- End of Bilateral Filter ------------------

def toneMapping(HDRImage):
	B, G, R = cv2.split(HDRImage)
	intensity = 1/61 * (20 * R + 40 * G + B)
	b = np.divide(B, intensity)
	g = np.divide(G, intensity)
	r = np.divide(R, intensity)

	log_intensity_layer = np.log10(intensity)
	log_base = cv2.bilateralFilter(log_intensity_layer, 10, 15, 15)
	# log_base = bilateralFilter(log_intensity_layer, 10)
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
	# intensity adjustment
	templateImage = cv2.imread('./input_images/indoor_aligned/1_25.JPG')
	tone_mapping_result = intensityAdjustment(tone_mapping_result, templateImage)
	return tone_mapping_result


if __name__ == '__main__':
	# inputFile = './output_images/My_Images_aligned'
	inputFile = './output_images/indoor_aligned'
	HDRImage = cv2.imread(inputFile + '/hdr_result.hdr', flags = cv2.IMREAD_ANYDEPTH)
	tone_mapping_result = toneMapping(HDRImage)
	cv2.imwrite("./tone_mapping_images/tone_mapping_result.jpg", tone_mapping_result)


	### OpenCV tone mapping
	# tonemapReinhard = cv2.createTonemapReinhard(1, 0.6, 0.6, 0.7)
	# ldrReinhard = tonemapReinhard.process(HDRImage)
	# cv2.imwrite("./tone_mapping_images/ldr-Reinhard.jpg", ldrReinhard * 255)

	# tonemapMantiuk = cv2.createTonemapMantiuk(0.83, 0.83, 0.83)
	# ldrMantiuk = tonemapMantiuk.process(HDRImage)
	# ldrMantiuk = 3 * ldrMantiuk
	# cv2.imwrite("./tone_mapping_images/ldr-Mantiuk.jpg", ldrMantiuk * 255)

	# tonemapDrago = cv2.createTonemapDrago(0.52, 0.52, 0.68)
	# ldrDrago = tonemapDrago.process(HDRImage)
	# ldrDrago = 3 * ldrDrago
	# cv2.imwrite("./tone_mapping_images/ldr-Drago.jpg", ldrDrago * 255)