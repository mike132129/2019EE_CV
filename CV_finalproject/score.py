import cv2
from util import cal_avgerr, readPFM
import sys


# for finding the score, please try following
# python3 score.py data/Synthetic/TLD0.pfm result1/TL0.pfm 


GT, disp = sys.argv[1], sys.argv[2]


GT = readPFM(str(GT))
disp = readPFM(str(disp))


print("average score:", cal_avgerr(GT, disp))