# Created by Varun at 22/11/18
import cv2
import numpy as np


def median_and_mean_filtering():
    """
    Perform the median and mean filtering on the image
    """
    img = cv2.imread("../original_imgs/noise.jpg", 0)
    cv2.imshow("original", img)
    new_img_median = np.zeros_like(img)
    new_img_mean = np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            pixel_arr = sorted(
                [img[i - 1][j - 1], img[i - 1][j], img[i - 1][j + 1], img[i][j - 1], img[i][j], img[i][j + 1],
                 img[i + 1][j - 1], img[i + 1][j], img[i + 1][j + 1]])
            median = np.median(pixel_arr)
            mean = np.mean(pixel_arr)
            new_img_mean[i][j] = mean
            new_img_median[i][j] = median
    cv2.imwrite("res_noise1.jpg", new_img_median)
    cv2.imwrite("res_noise2.jpg", new_img_mean)

def dilate(img):
    """
    Dilates the input image and returns the enlarged version of the input image
    :param img: image
    :return:
    """
    img_copy = img.copy()
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            if img[i][j] == 255:
                try:
                    img_copy[i-1][j] = img_copy[i][j+1] = img_copy[i][j-1] = img_copy[i+1][j] = 255
                except IndexError:
                    continue
    return img_copy

def draw_boundaries(img, tag, k=2):
    """
    Finds the boundaries and returns an image with just boundaries and hollow figures
    :param img: image
    :param tag: name of the image
    :param k: number of iterations to perform dilation
    """
    img_copy = img.copy()
    for i in range(k):
        img = dilate(img)
    cv2.imwrite(tag, img-img_copy)


median_and_mean_filtering()
img = cv2.imread("res_noise1.jpg", 0)
draw_boundaries(img, "res_bound1.jpg", k=2)
img1 = cv2.imread("res_noise2.jpg", 0)
draw_boundaries(img1, "res_bound2.jpg", k=3)