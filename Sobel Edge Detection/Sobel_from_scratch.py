import cv2
import numpy as np

image = cv2.imread('task1.png', 0)
sobelx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)
sobely = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float)

rows = image.shape[0]
columns = image.shape[1]
print(image.shape)
sobel_horizontal_values = np.zeros((600, 900))
sobel_vertical_values = np.zeros((600, 900))
sobel_gradient_values = np.zeros((600, 900))


def calculate_horizontal_value():
    return (sobely[0][0] * image[i - 1][j - 1]) + (sobely[0][1] * image[i - 1][j]) + (
            sobely[0][2] * image[i - 1][j + 1]) + \
           (sobely[1][0] * image[i][j - 1]) + (sobely[1][1] * image[i][j]) + (sobely[1][2] * image[i][j + 1]) + \
           (sobely[2][0] * image[i + 1][j - 1]) + (sobely[2][1] * image[i + 1][j]) + (
                   sobely[2][2] * image[i + 1][j + 1])


def calculate_vertical_values():
    return (sobelx[0][0] * image[i - 1][j - 1]) + (sobelx[0][1] * image[i - 1][j]) + (
            sobelx[0][2] * image[i - 1][j + 1]) + \
           (sobelx[1][0] * image[i][j - 1]) + (sobelx[1][1] * image[i][j]) + (sobelx[1][2] * image[i][j + 1]) + \
           (sobelx[2][0] * image[i + 1][j - 1]) + (sobelx[2][1] * image[i + 1][j]) + (
                   sobelx[2][2] * image[i + 1][j + 1])


for i in range(1, rows - 1):
    for j in range(1, columns - 1):
        gx = calculate_vertical_values()

        gy = calculate_horizontal_value()

        sobel_horizontal_values[i - 1][j - 1] = gy
        sobel_vertical_values[i - 1][j - 1] = gx

        g = np.sqrt(gx * gx + gy * gx)
        sobel_gradient_values[i - 1][j - 1] = g
        if sobel_horizontal_values[i - 1][j - 1] > 255:
            print("YAY")

cv2.imwrite('Horizontal_filter.png', sobel_horizontal_values)
cv2.imwrite('Vertical_filter.png', sobel_vertical_values)
cv2.imwrite('Gradient.png', sobel_gradient_values)
cv2.waitKey()
cv2.destroyAllWindows()
