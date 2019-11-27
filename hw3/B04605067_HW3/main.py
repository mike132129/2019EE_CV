import numpy as np
import cv2
import time


# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')


    A = np.array([[u[0][0], u[0][1], 1, 0, 0, 0, -u[0][0]*v[0][0], -u[0][1]*v[0][0], -v[0][0]],
    			  [0, 0, 0, u[0][0], u[0][1], 1, -u[0][0]*v[0][1], -u[0][1]*v[0][1], -v[0][1]],
    			  [u[1][0], u[1][1], 1, 0, 0, 0, -u[1][0]*v[1][0], -u[1][1]*v[1][0], -v[1][0]],
    			  [0, 0, 0, u[1][0], u[1][1], 1, -u[1][0]*v[1][1], -u[1][1]*v[1][1], -v[1][1]],
    			  [u[0][0], u[2][1], 1, 0, 0, 0, -u[2][0]*v[2][0], -u[2][1]*v[2][0], -v[2][0]],
    			  [0, 0, 0, u[2][0], u[2][1], 1, -u[2][0]*v[2][1], -u[2][1]*v[2][1], -v[2][1]],
    			  [u[3][0], u[3][1], 1, 0, 0, 0, -u[3][0]*v[3][0], -u[3][1]*v[3][0], -v[3][0]],
    			  [0, 0, 0, u[3][0], u[3][1], 1, -u[3][0]*v[3][1], -u[3][1]*v[3][1], -v[3][1]],
    				])

    _, _, V = np.linalg.svd(A)

    H = V[8].reshape(3, 3)

    return H


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    img = np.array(img)
    h, w, ch = img.shape
    orig_corner = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    H = solve_homography(orig_corner, corners)
    
    # apply H to every point
    for i in range(h):
        for j in range(w):
            tmp = np.dot(H, np.array([[j, i, 1]]).T)
            x, y = int(tmp[0][0] / tmp[2][0]), int(tmp[1][0] / tmp[2][0])
            canvas[y][x] = img[i][j]

    return canvas

def backward_warpping(img, output, corners):
    h, w, ch = output.shape
    img_corner = np.array([[0, 0], [w-1, 0], 
                           [0, h-1], [w-1, h-1]
                          ])
    homography_matrix = solve_homography(img_corner, corners)
    for y in range(h):
        for x in range(w):
            new_pos = np.dot(homography_matrix, np.array([[x, y, 1]]).T)
            new_x, new_y = new_pos[0][0] / new_pos[2][0], new_pos[1][0] / new_pos[2][0]
            res = interpolation(img, new_x, new_y)
            output[y][x] = res

    return output

def interpolation(img, new_x, new_y):
    fx = round(new_x - int(new_x), 2)
    fy = round(new_y - int(new_y), 2)

    p = np.zeros((3,))
    p += (1 - fx) * (1 - fy) * img[int(new_y), int(new_x)]
    p += (1 - fx) * fy * img[int(new_y) + 1, int(new_x)]
    p += fx * (1 - fy) * img[int(new_y), int(new_x) + 1]
    p += fx * fy * img[int(new_y) + 1, int(new_x) + 1]

    return p


def main():
    #Part 1
    ts = time.time()
    canvas = cv2.imread('./input/Akihabara.jpg')
    img1 = cv2.imread('./input/lu.jpeg')
    img2 = cv2.imread('./input/kuo.jpg')
    img3 = cv2.imread('./input/haung.jpg')
    img4 = cv2.imread('./input/tsai.jpg')
    img5 = cv2.imread('./input/han.jpg')

    canvas_corners1 = np.array([[779,312],[1014,176],[739,747],[978,639]])
    canvas_corners2 = np.array([[1194,496],[1537,458],[1168,961],[1523,932]])
    canvas_corners3 = np.array([[2693,250],[2886,390],[2754,1344],[2955,1403]])
    canvas_corners4 = np.array([[3563,475],[3882,803],[3614,921],[3921,1158]])
    canvas_corners5 = np.array([[2006,887],[2622,900],[2008,1349],[2640,1357]])

    canvas = transform(img1, canvas, canvas_corners1)
    canvas = transform(img2, canvas, canvas_corners2)
    canvas = transform(img3, canvas, canvas_corners3)
    canvas = transform(img4, canvas, canvas_corners4)
    canvas = transform(img5, canvas, canvas_corners5)

    cv2.imwrite('part1.png', canvas)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))
    #==================Part1 END=================#

    # Part 2
    ts = time.time()
    img = cv2.imread('./input/QR_code.jpg')

    output2 = np.zeros((150, 150, 3))
    screen_corner = np.array([[1981, 1251], [2043, 1223], [2029, 1411], [2085, 1376]])
    print(screen_corner.shape)
    outpu2 = backward_warpping(img, output2, screen_corner)
    cv2.imwrite('part2.png', output2)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))
    #==========end============

    # # Part 3
    ts = time.time()

    img_front = cv2.imread('./input/crosswalk_front.jpg')

    top_corners = np.array([[138,132],[578,122], [0,255],[723,245]])
    output3 = np.zeros((200, 200, 3))
    output3 = backward_warpping(img_front, output3, top_corners)

    cv2.imwrite('part3.png', output3)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

if __name__ == '__main__':
    main()
