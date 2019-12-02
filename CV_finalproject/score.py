import cv2
from util import cal_avgerr, readPFM
import sys

GT, disp = sys.argv[1], sys.argv[2]


GT = readPFM(str(GT))
disp = readPFM(str(disp))


print(cal_avgerr(GT, disp))