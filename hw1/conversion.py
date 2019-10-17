import numpy as np
import sys
import cv2
from joint_bilateral_filter import Joint_bilateral_filter
#np.set_printoptions(threshold=sys.maxsize)

def find_local_minimum(error_mtx):
	local_min = []
	width = error_mtx.shape[0]
	error_mtx = np.insert(error_mtx, 0, 600000000, axis = 0)
	error_mtx = np.insert(error_mtx, 0, 600000000, axis = 1)
	error_mtx = np.insert(error_mtx, 0, 600000000, axis = 2)
	error_mtx = np.insert(error_mtx, width + 1, 600000000, axis = 0)
	error_mtx = np.insert(error_mtx, width + 1, 600000000, axis = 1)
	error_mtx = np.insert(error_mtx, width + 1, 600000000, axis = 2)


	for i in range(1, error_mtx.shape[0] - 1):
		for j in range(1, error_mtx.shape[0] - 1):
			for k in range(1, error_mtx.shape[0] - 1):
				if error_mtx[i][j][k] < error_mtx[i-1][j+1][k] and error_mtx[i][j][k] < error_mtx[i+1][j-1][k] and error_mtx[i][j][k] < error_mtx[i][j-1][k+1] and error_mtx[i][j][k] < error_mtx[i][j+1][k-1] and (error_mtx[i][j][k] < error_mtx[i+1][j][k-1]) and (error_mtx[i][j][k] < error_mtx[i-1][j][k+1]) :
					local_min.append([i-1, j-1, k-1])

	return local_min

a = np.array([(i, 10-i-k, k) for i in range(0, 11) for k in range(0, 11-i)])
a = a / 10
print(a)


# do BF three original photo
img_a = cv2.imread('1a.png')
img_b = cv2.imread('1b.png')
img_c = cv2.imread('1c.png')
img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
img_c_rgb = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)

#do conventional conversion
# weight = [0.299, 0.587, 0.114]
# img_a_gray = np.dot(img_a_rgb, weight)
# img_b_gray = np.dot(img_b_rgb, weight)
# img_c_gray = np.dot(img_c_rgb, weight)

# cv2.imwrite('1a_gray.png', img_a_gray)
# cv2.imwrite('1b_gray.png', img_b_gray)
# cv2.imwrite('1c_gray.png', img_c_gray)


for i in a:
	b = np.dot(img_a_rgb, i).astype(np.uint8)
	cv2.imwrite( 'a' + str(i) + '.png',b)

for i in a:
	b = np.dot(img_b_rgb, i).astype(np.uint8)
	cv2.imwrite( 'b' + str(i) + '.png',b)

for i in a:
	b = np.dot(img_c_rgb, i).astype(np.uint8)
	cv2.imwrite( 'c' + str(i) + '.png',b)


sigma_r = [0.05, 0.1, 0.2]
sigma_s = [1, 2, 3]

# #bilateral filter
# for i in sigma_s:
# 	for j in sigma_r:
# 		print(i, j)
# 		JBF = Joint_bilateral_filter(i, j, border_type='reflect')
# 		abf_out = JBF.joint_bilateral_filter(img_a_rgb, img_a_rgb).astype(np.uint8)
# 		bbf_out = JBF.joint_bilateral_filter(img_b_rgb, img_b_rgb).astype(np.uint8)
# 		cbf_out = JBF.joint_bilateral_filter(img_c_rgb, img_c_rgb).astype(np.uint8)
# 		cv2.imwrite('BF_a'+str(i)+str(j)+'.png', abf_out)
# 		cv2.imwrite('BF_b'+str(i)+str(j)+'.png', bbf_out)
# 		cv2.imwrite('BF_c'+str(i)+str(j)+'.png', cbf_out)

for i in a:
	for j in sigma_s:
		for k in sigma_r:
			print(i, j, k)

			JBF = Joint_bilateral_filter(j, k, border_type='reflect')

			guidance_a = cv2.imread('a'+str(i)+'.png')
			guidance_a = cv2.cvtColor(guidance_a, cv2.COLOR_BGR2GRAY)
			ajbf_out = JBF.joint_bilateral_filter(img_a_rgb, guidance_a).astype(np.uint8)
			cv2.imwrite('JBF_a'+str(i)+str(j)+str(k)+'.png', ajbf_out)

			guidance_b = cv2.imread('b'+str(i)+'.png')
			guidance_b = cv2.cvtColor(guidance_b, cv2.COLOR_BGR2GRAY)
			bjbf_out = JBF.joint_bilateral_filter(img_b_rgb, guidance_b).astype(np.uint8)
			cv2.imwrite('JBF_b'+str(i)+str(j)+str(k)+'.png', bjbf_out)

			guidance_c = cv2.imread('c'+str(i)+'.png')
			guidance_c = cv2.cvtColor(guidance_c, cv2.COLOR_BGR2GRAY)
			cjbf_out = JBF.joint_bilateral_filter(img_c_rgb, guidance_c).astype(np.uint8)
			cv2.imwrite('JBF_c'+str(i)+str(j)+str(k)+'.png', cjbf_out)


voting_matrix = np.zeros((11, 11, 11))


x = np.array([(i, 10-i-k, k) for i in range(0, 11) for k in range(0, 11-i)])


for i in sigma_r:
	for j in sigma_s:
		error_matrix = np.full((11, 11, 11), 600000000)
		for k in x:
			a = cv2.imread('BF_a'+str(j)+str(i)+'.png').astype(np.float64)
			b = cv2.imread('JBF_a'+str(k/10)+str(j)+str(i)+'.png').astype(np.float64)
			
			error_matrix[k[0]][k[1]][k[2]] = np.sum(np.abs(a - b))
			print(error_matrix[k[0]][k[1]][k[2]])
		local_m = find_local_minimum(error_matrix)
		
		print(local_m)
		for l in local_m:
			voting_matrix[l[0]][l[1]][l[2]] += 1


for i in range(3):
	result = np.where(voting_matrix == np.amax(voting_matrix))
	ans = np.max(voting_matrix)
	print(i, result, ans)
	for j in range(len(result[0])):
		voting_matrix[result[0][j]][result[1][j]][result[2][j]] = 0

voting_matrix = np.zeros((11, 11, 11))

for i in sigma_r:
	for j in sigma_s:
		error_matrix = np.full((11, 11, 11), 600000000)
		for k in x:
			a = cv2.imread('BF_b'+str(j)+str(i)+'.png').astype(np.float64)
			b = cv2.imread('JBF_b'+str(k/10)+str(j)+str(i)+'.png').astype(np.float64)

			error_matrix[k[0]][k[1]][k[2]] = np.sum(np.abs(a - b))
			#print(error_matrix[k[0]][k[1]][k[2]])

		local_m = find_local_minimum(error_matrix)
		print(local_m)
		for l in local_m:
			voting_matrix[l[0]][l[1]][l[2]] += 1


for i in range(3):
	result = np.where(voting_matrix == np.amax(voting_matrix))
	ans = np.max(voting_matrix)
	print(i, result, ans)
	for j in range(len(result[0])):
		voting_matrix[result[0][j]][result[1][j]][result[2][j]] = 0

voting_matrix = np.zeros((11, 11, 11))


for i in sigma_r:
	for j in sigma_s:
		error_matrix = np.full((11, 11, 11), 600000000)
		for k in x:

			a = cv2.imread('BF_c'+str(j)+str(i)+'.png').astype(np.double)
			b = cv2.imread('JBF_c'+str(k/10)+str(j)+str(i)+'.png').astype(np.double)

			error_matrix[k[0]][k[1]][k[2]] = (np.sum(np.abs(a - b)))

		local_m = find_local_minimum(error_matrix)
		#print(local_m)
		for l in local_m:
			voting_matrix[l[0]][l[1]][l[2]] += 1


for i in range(3):
	result = np.where(voting_matrix == np.amax(voting_matrix))
	ans = np.max(voting_matrix)
	print(i, result, ans)
	for j in range(len(result[0])):
		voting_matrix[result[0][j]][result[1][j]][result[2][j]] = 0



			

	
