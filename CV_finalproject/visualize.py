import sys
from util import readPFM, writePFM, cal_avgerr
import numpy as np
import cv2
import sys
# np.set_printoptions(threshold=sys.maxsize)

# read disparity pfm file (float32)
# the ground truth disparity maps may contain inf pixels as invalid pixels
disp = readPFM(str(sys.argv[1]))
gt = readPFM(str(sys.argv[2]))

# normalize disparity to 0.0~1.0 for visualization
def img_proc(disp):
	max_disp = np.nanmax(disp[disp != np.inf])
	min_disp = np.nanmin(disp[disp != np.inf])
	disp_normalized = (disp - min_disp) / (max_disp - min_disp)

	# Jet color mapping
	disp_normalized = (disp_normalized * 255.0).astype(np.uint8)
	#cv2.imshow("visualized disparity", disp_normalized)
	#cv2.waitKey(0)
	disp_normalized = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
	#cv2.imshow("visualized disparity", disp_normalized)
	return disp_normalized

disp = img_proc(disp)
gt = img_proc(gt)

numpy_horizontal = np.hstack((disp, gt))

cv2.imshow('Numpy Horizontal', numpy_horizontal)
cv2.waitKey(0)

