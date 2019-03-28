# Created by Varun at 16/11/18

import cv2
import numpy as np
import matplotlib.pyplot as plt


def point_segmentation():
    img = cv2.imread("../original_imgs/point.jpg", 0)
    img = cv2.blur(img, (5,5))
    mask2 = [0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0]
    row, col = img.shape # img is a grayscale image
    y = np.zeros((256), np.uint64)
    for i in range(0,row):
        for j in range(0,col):
            y[img[i,j]] += 1
    x = np.arange(0,256)
    plt.bar(x,y,color="gray",align="center")
    plt.savefig("Histogram_plot.jpg")
    point_list_2 = []
    sum_list = []
    threshold = 101
    for i in range(2, img.shape[0]):
        for j in range(2, img.shape[1]):
            try:
                sum = (
                        (mask2[0] * img[i - 2][j - 2]) +
                        (mask2[1] * img[i - 2][j - 1]) +
                        (mask2[2] * img[i - 2][j]) +
                        (mask2[3] * img[i - 2][j + 1]) +
                        (mask2[4] * img[i - 2][j + 2]) +
                        (mask2[5] * img[i - 1][j - 2]) +
                        (mask2[6] * img[i - 1][j - 1]) +
                        (mask2[7] * img[i - 1][j]) +
                        (mask2[8] * img[i - 1][j + 1]) +
                        (mask2[9] * img[i - 1][j + 2]) +
                        (mask2[10] * img[i][j - 2]) +
                        (mask2[11] * img[i][j - 1]) +
                        (mask2[12] * img[i][j]) +
                        (mask2[13] * img[i][j + 1]) +
                        (mask2[14] * img[i][j + 2]) +
                        (mask2[15] * img[i + 1][j - 2]) +
                        (mask2[16] * img[i + 1][j - 1]) +
                        (mask2[17] * img[i + 1][j]) +
                        (mask2[18] * img[i + 1][j + 1]) +
                        (mask2[19] * img[i + 1][j + 2]) +
                        (mask2[20] * img[i + 2][j - 2]) +
                        (mask2[21] * img[i + 2][j - 1]) +
                        (mask2[22] * img[i + 2][j]) +
                        (mask2[23] * img[i + 2][j + 1]) +
                        (mask2[24] * img[i + 2][j + 2]))
                sum_list.append(sum)

                if abs(sum) >= threshold and abs(sum) < threshold+2:
                    point_list_2.append((i, j))
            except IndexError:
                continue
    print("The threshold being used here is: {}".format(threshold))
    print("The coordinates of the points detected using 5x5 Kernel are: {}".format(point_list_2))
    img_copy_2 = np.zeros_like(img)
    for point in point_list_2:
        img_copy_2[point[0]][point[1]] = 255
    cv2.imwrite("res_point.jpg", img_copy_2)


point_segmentation()