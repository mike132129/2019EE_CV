import sys
from util import readPFM, writePFM, cal_avgerr
import numpy as np
import cv2
import sys
# np.set_printoptions(threshold=sys.maxsize)

# read disparity pfm file (float32)
# the ground truth disparity maps may contain inf pixels as invalid pixels
disp = readPFM(str(sys.argv[1]))
#gt = readPFM(str(sys.argv[2]))

# normalize disparity to 0.0~1.0 for visualization
def img_proc(disp):
	max_disp = np.nanmax(disp[disp != np.inf])
	min_disp = np.nanmin(disp[disp != np.inf])
	disp_normalized = (disp - min_disp) / (max_disp - min_disp)

	# Jet color mapping
	disp_normalized = (disp_normalized * 255.0).astype(np.uint8)
	disp_normalized = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
	return disp_normalized

disp = img_proc(disp)

cv2.imshow('show disparity', disp)
cv2.waitKey(0)

