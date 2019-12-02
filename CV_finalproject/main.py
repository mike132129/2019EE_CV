import numpy as np
import argparse
import cv2
import time
from util import writePFM

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./result1/TL0.pfm', type=str, help='left disparity map')

def cut(disparity, image, threshold):
    for i in range(0, image.height):
        for j in range(0, image.width):
            if cv2.GetReal2D(disparity, i, j) > threshold:
                cv2.Set2D(disparity, i, j, Get2D(image, i, j))


# You can modify the function interface as you like
def computeDisp(Il, Ir):
    h, w, ch = Il.shape
    disp = np.zeros((h, w), dtype=np.int32)

    Il = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    Ir = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)

    # Il, Ir = cv2.GaussianBlur(Il, (1, 1), ), cv2.GaussianBlur(Ir, (1, 1), )

    # stereo = cv2.cv2.StereoSGBM_create(0, 32, 31)
    # disp = stereo.compute(Il, Ir)

    disp_l = cv2.CreateImage(h, w, cv2.CV_16S)
    disp_r = cv2.CreateImage(Ir.shape[0], Ir.shape[1], cv2.CV_16S)

    state = cv2.CreateStereoGCState(16, 2)

    cv2.FindStereoCorrespondenceGC(Il, Ir, disp_l, disp_r, state)

    disp_l_visual = cv2.CreateMat(Il.height, Ir.width, cv2.CV_8U)
    cv2.ConvertScale(disp_l, disp_l_visual, -20)
    cut(disp_l_visual, Il, 120)

    cv2.namedWindow('DISPARITY MAP', cv2.WINDOW_NORMAL)
    cv2.imshow('DISPARITY MAP', disp_l_visual)
    cv2.waitKey(0)




    return disp_l_visual


def main():
    args = parser.parse_args()

    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    disp = np.float32(computeDisp(img_left, img_right))
    toc = time.time()
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))


if __name__ == '__main__':
    main()
