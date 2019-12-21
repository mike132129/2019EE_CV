import numpy as np
import cv2
import time
import cv2.ximgproc as cv2_x
from matplotlib import pyplot as plt
import pdb

MIN_MATCH_COUNT = 10

def SIFT(Ir, Il):
    h, w, ch = Il.shape
    disp = np.zeros((h, w), dtype=np.int32)

    Il = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    Ir = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(Il, None)
    kp2, des2 = sift.detectAndCompute(Ir, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k = 2)

    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = Il.shape[0], Il.shape[1]
        pts = np.float32([[0 ,0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

    else:
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)

    img_sift = cv2.drawMatches(Il, kp1, Ir, kp2, good, None, **draw_params)

    src_pts_int = np.int16([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst_pts_int = np.int16([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)


    SIFT_dic = {}
    for i in range(src_pts_int.shape[0]):
        SIFT_dic[tuple(src_pts_int[i])] = np.int((src_pts_int[i]-dst_pts_int[i])[0])

    pdb.set_trace()

    cv2.namedWindow('DISPARITY MAP', cv2.WINDOW_NORMAL)
    cv2.imshow('DISPARITY MAP', img_sift)
    cv2.waitKey(0)

    return SIFT_dic


################################### Cost computation ############################################
def SSD(Il,Ir,max_disp,l2r): # Squared Sum Difference
    h, w, ch = Il.shape
    cost = np.zeros((h,w,max_disp))
    for d in range(max_disp): 
        for i in range(h):
            for j in range(w):
                if l2r:
                    if j-d >= 0 :
                        vl = Il[i,j]            
                        vr = Ir[i,j-d]
                        cost[i,j,d] = sum((vl-vr)**2)
                else:
                    if j+d < w :
                        vl = Ir[i,j]            
                        vr = Il[i,j+d]
                        cost[i,j,d] = sum((vl-vr)**2)

        if d > 0 : # padding
            if l2r :
                for i in range(d):
                    cost[:,i,d] = cost[:,d,d]
            else:
                for i in range(d):
                    cost[:,(w-1)-i,d] = cost[:,(w-1)-d,d] 
          
    return cost

################################### Cost aggregation ############################################
def cost_volumne_filtering(raw_cost):
    h,w,max_disp = raw_cost.shape
    smoothed_cost = np.zeros((h,w,max_disp))
    for d in range(max_disp):
        smoothed_cost[:,:,d] = cv2.blur(raw_cost[:,:,d],(9,9))
        #smoothed_cost[:,:,d] = cv2.GaussianBlur(raw_cost[:,:,d], (9, 9), 0)
        #smoothed_cost[:,:,d] = cv2.bilateralFilter(np.float32(raw_cost[:,:,d]), 5, 21, 21)

        # raw_cost_g = raw_cost[:,:,d].reshape((h,w)).astype(np.float32)
        # smoothed_cost[:,:,d] = cv2.ximgproc.guidedFilter(guide=raw_cost_g, src=raw_cost_g, radius=16, eps=1000, dDepth=-1)
    
    return smoothed_cost

###################################Disparity optimization############################################
def WTA(Cost):



    h,w,d=Cost.shape
    labels = np.zeros((h,w))    
    # Winner-take-all.
    for i in range(h):
        for j in range(w):
            cost = Cost[i,j,:]
            labels[i,j]= np.argmin(cost)

    return labels

###################################Disparity refinement############################################
def consistency_check(Dl,Dr):
    h,w = Dl.shape
    Dl = np.float64(Dl)
    Y , X = [] , []
    for y in range(h):
        for x in range(w):
            if x-int(Dl[y,x]) >= 0:
                if  Dl[y,x] != Dr[y,x-int(Dl[y,x])]:
                    Dl[y,x] = 0
                    X.append(x)
                    Y.append(y)
    return  Dl,Y,X

def hole_filling(labels,Y,X):
    L = len(Y)
    for l in range(L):
        y , x = Y[l] , X[l]
        l_slice,r_slice = labels[y,0:x],labels[y,x+1:labels.shape[1]]
        l_can , r_can = -1, -1
        i , j = len(l_slice)-1 , 0
        while(l_can <= 0 and i >= 0 ):
            l_can = l_slice[i]
            i = i - 1
        while(r_can <= 0 and j < len(r_slice)):
            r_can = r_slice[j]
            j = j + 1            
        if l_can <=0 :
            labels[y,x] = r_can
        elif r_can <=0 :
            labels[y,x] = l_can
        else:        
            labels[y,x] = min(l_can,r_can)
    return labels

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    Il = Il.astype('float32')
    Ir = Ir.astype('float32')
    labels = np.zeros((h,w))
    
    # >>> Cost computation
    tic = time.time()
    SDl = SSD(Il,Ir,max_disp,1)
    SDr = SSD(Il,Ir,max_disp,0)    
    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    Cl = cost_volumne_filtering(SDl)
    Cr = cost_volumne_filtering(SDr)    
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))
    
    # >>> Disparity optimization
    tic = time.time()    
    Ll = WTA(Cl)
    Lr = WTA(Cr)
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))
    
    # >>> Disparity refinement
    tic = time.time()
    Ll = cv2_x.weightedMedianFilter(Il.astype('uint8'),Ll.astype('uint8'),15,5,cv2_x.WMF_JAC)
    Lr = cv2_x.weightedMedianFilter(Ir.astype('uint8'),Lr.astype('uint8'),15,5,cv2_x.WMF_JAC)    
    labels,Y,X = consistency_check(Ll,Lr)# Left-right consistency check 
    
    labels = hole_filling(labels,Y,X)
    labels = cv2_x.weightedMedianFilter(Il.astype('uint8'),labels.astype('uint8'),15,5,cv2_x.WMF_JAC)
    toc = time.time()

    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))
    
    return labels

def main():
    # print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')

    # img_left = cv2.copyMakeBorder(img_left,0,0,0,5,cv2.BORDER_CONSTANT,value=0)
    # img_right = cv2.copyMakeBorder(img_right,0,0,5,0,cv2.BORDER_CONSTANT,value=0)

    # SIFT(img_left, img_right)
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))
    
    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('venus.png', np.uint8(labels * scale_factor))
    
    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))

    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('cones.png', np.uint8(labels * scale_factor))
    
if __name__ == '__main__':
    main()
