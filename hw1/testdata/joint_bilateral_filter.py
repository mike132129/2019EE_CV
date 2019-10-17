import numpy as np
import cv2 
import time

class Joint_bilateral_filter(object):

	def __init__(self, sigma_s, sigma_r, border_type='reflect'):

		self.border_type = border_type
		self.sigma_r = sigma_r
		self.sigma_s = sigma_s
	
	def spatial_kernel(self): 

		r = 3*self.sigma_s
		a = np.arange(-r, r+1)
		a = np.multiply(a, a)
		b = np.zeros((len(a), len(a)))

		for j in range(len(a)):
			for i in range(len(a)):
				b[i][j] = a[i] + a[j]

		b = np.float64(np.true_divide(b, (-2*(self.sigma_s**2))))

		return np.exp(b)

	def range_kernel(self, guidance, x, y, reflect_g):

		r = 3*self.sigma_s
		a = np.arange(-r, r+1)
		
		b = np.zeros((len(a), len(a)))

		if len(guidance.shape) is 2:


			b = reflect_g[x:(x+2*r+1), y:(y+2*r+1)]
					
			b = np.float64(np.true_divide(np.subtract(b, np.float64(guidance[x][y])), 255))
			b = np.square(b)
			

		else:

			guidance_diff = reflect_g[x:(x+2*r+1), y:(y+2*r+1)]

			guidance_diff = np.float64(np.true_divide(np.subtract(guidance_diff, np.float64(guidance[x][y])), 255))
			b = np.square(guidance_diff)
			b = np.sum(b, axis = 2)

		b = np.float64(np.true_divide(b, (-2*(self.sigma_r**2))))

		return np.exp(b)

	def ori_img(self, input, x, y, reflect_i):

		r = 3*self.sigma_s
		a = np.arange(-r, r+1)

		
		
		if len(input.shape) is 2:
			input_diff = np.zeors((len(a), len(a), 1))
		
		else:
			input_diff = np.zeros((len(a), len(a), input.shape[2]))

		for j in range(len(a)):
			for i in range(len(a)):
				
				input_diff[i][j] = np.float64(reflect_i[x-a[i]+r][y-a[j]+r])

		return input_diff

	def joint_bilateral_filter(self, input, guidance):

		r = 3*self.sigma_s
		a = np.arange(-r, r+1)

		reflect_g = cv2.copyMakeBorder(guidance, r, r, r, r, cv2.BORDER_REFLECT)
		reflect_i = cv2.copyMakeBorder(input, r, r, r, r, cv2.BORDER_REFLECT)

		h_s = self.spatial_kernel()
		h_s_1 = np.float64(h_s[:, :, np.newaxis])

		output = np.zeros(input.shape)

		for y in range(0, input.shape[1]):
			for x in range(0, input.shape[0]):

				h_r = np.float64(self.range_kernel(guidance, x, y, reflect_g))
				h_r_1 = np.float64(h_r[:, :, np.newaxis])

				h = np.float64(np.multiply(h_s_1, h_r_1))
				kernel = np.float64(np.sum(h))

				f = reflect_i[x:(x+2*r+1), y:(y+2*r+1)]
	
				c = np.float64(np.multiply(h, f))

				d = np.float64(np.sum(c, axis = 0))
				output[x][y] = np.float64(np.sum(d, axis = 0)/kernel)


		return output



















