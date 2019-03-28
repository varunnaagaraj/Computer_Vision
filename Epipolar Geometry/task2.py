import cv2
import numpy as np
import random


def epipolar_geometry():
    original_1 = cv2.imread('../data/tsucuba_left.png')
    original_2 = cv2.imread('../data/tsucuba_right.png')
    img1 = cv2.imread('../data/tsucuba_left.png', 0)
    img2 = cv2.imread('../data/tsucuba_right.png', 0)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    print (img1.dtype)
    print (img2.dtype)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    keypoint1 = original_1.copy()
    keypoint2 = original_2.copy()
    cv2.drawKeypoints(original_1, kp1, keypoint1)
    cv2.drawKeypoints(original_2, kp2, keypoint2)
    cv2.imwrite("../Task2/task2_sift1.jpg", keypoint1)
    cv2.imwrite("../Task2/task2_sift2.jpg", keypoint2)
    # create FLANN matcher object
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])

    img_knn = cv2.drawMatches(original_1, kp1, original_2, kp2, good, None)
    cv2.imwrite("../Task2/task2_matches_knn.jpg", img_knn)

    F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.RANSAC)
    print("F :{}".format(F))

    stereoMatcher = cv2.StereoBM_create()
    stereoMatcher.setMinDisparity(6)
    stereoMatcher.setNumDisparities(48)
    stereoMatcher.setBlockSize(21)

    stereoMatcher.setSpeckleRange(15)
    stereoMatcher.setSpeckleWindowSize(40)

    disparity = stereoMatcher.compute(img1, img2)

    cv2.imwrite("../Task2/disparity.jpg", disparity)

    src_pts = np.int32(src_pts)
    dst_pts = np.int32(dst_pts)
    pts1 = src_pts[mask.ravel()==1]
    pts2 = dst_pts[mask.ravel()==1]

    def drawlines(img1,img2,lines,pts1,pts2):
        r,c = img1.shape
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
            img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2

    lines1 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    lines_to_draw = random.sample(lines1, 10)
    img5,img6 = drawlines(img1,img2,lines_to_draw,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    lines_to_draw = random.sample(lines2, 10)
    img3,img4 = drawlines(img2,img1,lines_to_draw,pts2,pts1)
    cv2.imwrite("../Task2/task2_epi_right.jpg", img5)
    cv2.imwrite("../Task2/task2_epi_left.jpg", img3)


epipolar_geometry()