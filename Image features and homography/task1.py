import cv2
import numpy as np
import random

def find_keypoints_and_match():
    original_1 = cv2.imread('./data/mountain1.jpg')
    original_2 = cv2.imread('./data/mountain2.jpg')
    img1 = cv2.imread('./data/mountain1.jpg', 0)
    img2 = cv2.imread('./data/mountain2.jpg', 0)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    keypoint1 = original_1.copy()
    keypoint2 = original_2.copy()
    cv2.drawKeypoints(original_1, kp1, keypoint1)
    cv2.drawKeypoints(original_2, kp2, keypoint2)
    cv2.imwrite("./Task1/task1_sift1.jpg", keypoint1)
    cv2.imwrite("./Task1/task1_sift2.jpg", keypoint2)
    # create FLANN matcher object
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    img_knn = cv2.drawMatches(original_1, kp1, original_2, kp2, good, None)
    cv2.imwrite("./Task1/task1_matches_knn.jpg", img_knn)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(M)
    matchesMask = mask.ravel().tolist()
    matchesMask = random.sample(matchesMask, 10)

    draw_params = dict(matchColor=(255, 0, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    good_new_random = random.sample(good, 10)
    img3 = cv2.drawMatches(original_1, kp1, original_2, kp2, good_new_random, None, **draw_params)
    cv2.imwrite('./Task1/task1_matches.jpg', img3)


    def warpImages(img1, img2, H):
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]

        list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
        temp_points = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)

        list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
        list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

        output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
        output_img[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img1
        return output_img

    output_img = warpImages(original_2, original_1, M)
    cv2.imwrite('./Task1/task1_pano.jpg',output_img)
    cv2.waitKey()

find_keypoints_and_match()

