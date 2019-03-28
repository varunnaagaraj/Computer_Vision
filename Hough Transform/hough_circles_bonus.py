# Created by Varun at 30/11/18
import cv2
import numpy as np
import matplotlib.pyplot as plt


def hough_circles(img, threshold, region, radius=None):
    """
    Find the hough circles present in the image
    :param img: input image
    :param threshold: decides the number of accumulator arrays to consider
    :param region: region of the image to consider for every iteration
    :param radius: tuple of max radius and min radius
    :return: Matrix containing the values accumulated through finding process
    """
    (M, N) = img.shape
    if radius == None:
        max_radius = np.max((M, N))
        min_radius = 3
    else:
        [max_radius, min_radius] = radius

    radius = max_radius - min_radius
    accumulator = np.zeros((max_radius, M + 2 * max_radius, N + 2 * max_radius))
    B = np.zeros((max_radius, M + 2 * max_radius, N + 2 * max_radius))
    theta = np.arange(0, 360) * np.pi / 180
    edges = np.argwhere(img[:, :])
    for val in range(radius):
        r = min_radius + val
        bprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
        (m, n) = (r + 1, r + 1)
        for angle in theta:
            x = int(np.round(r * np.cos(angle)))
            y = int(np.round(r * np.sin(angle)))
            bprint[m + x, n + y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x, y in edges:
            X = [x - m + max_radius, x + m + max_radius]
            Y = [y - n + max_radius, y + n + max_radius]
            accumulator[r, X[0]:X[1], Y[0]:Y[1]] += bprint
        accumulator[r][accumulator[r] < threshold * constant / r] = 0

    for r, x, y in np.argwhere(accumulator):
        temp = accumulator[r - region:r + region, x - region:x + region, y - region:y + region]
        try:
            p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
        except:
            continue
        B[r + (p - region), x + (a - region), y + (b - region)] = 1

    return B[:, max_radius:-max_radius, max_radius:-max_radius]


def draw_circles(accumulator, file_path):
    """
    Creates the circles around the coins present in the image
    :param accumulator: Accumulator array containing information about the circle positions
    :param file_path: File path of the input file
    """
    img = cv2.imread(file_path)
    fig = plt.figure()
    # plt.imshow(img)
    circleCoordinates = np.argwhere(accumulator)
    circle = []
    for r, x, y in circleCoordinates:
        cv2.circle(img, (y,x), r, color=(0,255,0), thickness=2)
        # circle.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
        # fig.add_subplot(111).add_artist(circle[-1])
    # plt.savefig("hough_circles.jpg")
    cv2.imwrite("hough_circles.jpg", img)

img = cv2.imread("../original_imgs/hough.jpg", 0)

# cv2.imshow('Original Image', img)

img = cv2.GaussianBlur(img, (3, 3), 0)

# cv2.imshow('Gaussian Blur', img)

edges = cv2.Canny(img, 128, 200)

# cv2.imshow('Edges Detected', edges)

res = hough_circles(edges, 8.1, 15, radius=[25, 10])
draw_circles(res, "../original_imgs/hough.jpg")
