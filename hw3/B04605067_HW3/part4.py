import numpy as np
import cv2
import sys
from main import solve_homography, transform

MIN_MATCH_COUNT = 10

def find_boundary(frame, template, ref_image):

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    # keypoint and descriptor
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(frame, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, tree = 5)
    search_params = dict(checks = 50)

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

        h, w = template.shape[0], template.shape[1]
        pts = np.float32([[0 ,0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        

    else:
        matchesMask = None

    ref_h, ref_w = ref_image.shape[0], ref_image.shape[1]
    ref_corners = np.float32([[0, 0], [0, ref_h-1], [ref_w-1, ref_h-1],[ref_w-1, 0]])
    M, _ = cv2.findHomography(ref_corners, dst, cv2.RANSAC, 5.0)
    warp_img = cv2.warpPerspective(ref_image, M, (frame.shape[1], frame.shape[0]))
    cv2.fillConvexPoly(frame, dst.astype(int), 0, 16)
    frame = frame + warp_img

    

    return frame




def main(ref_image,template,video):
    ref_image = cv2.imread(ref_image)  ## load gray if you need.
    template = cv2.imread(template)  ## load gray if you need.
    video = cv2.VideoCapture(video)
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) #1080, 1920
    film_fps = video.get(cv2.CAP_PROP_FPS)
    print(film_fps)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videowriter = cv2.VideoWriter("ar_video.mp4", fourcc, film_fps, (film_w, film_h))
    i = 0


    while(video.isOpened()):
        ret, frame = video.read()
        print('Processing frame {}'.format(i))
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            ## TODO: homography transform, feature detection, ransanc, etc.
            frame = find_boundary(frame, template, ref_image)

            videowriter.write(frame)

        else:
            break
            
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ## you should not change this part
    ref_path = './input/sychien.jpg'
    template_path = './input/marker_org1.png'
    video_path = sys.argv[1]  ## path to ar_marker.mp4
    main(ref_path,template_path,video_path)










