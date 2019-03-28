# Created by Varun at 27/11/18

# Segmentation using Thresholding
import cv2
import numpy as np

img = cv2.imread("../original_imgs/segment.jpg", 0)
width, height = img.shape
g1 = np.zeros_like(img)
threshold = 202
for i in range(width):
    for j in range(height):
        if img[i][j] > threshold:
            g1[i][j] = 255
        else:
            g1[i][j] = 0
edge_g = cv2.Canny(g1, 100, 200)
cv2.imwrite("edge_output.jpg", edge_g)
rot_edge_g = edge_g.T
white_pixels = np.array(np.where(edge_g == 255))
first_white_pixel = white_pixels[:,0]
last_white_pixel = white_pixels[:,-1]

white_pixels_y = np.array(np.where(rot_edge_g == 255))
first_white_pixel_y = white_pixels_y[:,0]
last_white_pixel_y = white_pixels_y[:,-1]

print(first_white_pixel[0], first_white_pixel[1], last_white_pixel[0], last_white_pixel[1])
print(first_white_pixel_y[0], first_white_pixel_y[1], last_white_pixel_y[0], last_white_pixel_y[1])
im_col = cv2.imread("../original_imgs/segment.jpg")
cv2.rectangle(im_col, (first_white_pixel_y[1], first_white_pixel[0]), (last_white_pixel_y[0], last_white_pixel[0]), (0,255,255), thickness=3)
cv2.imwrite("all_bones_box.jpg",im_col)
cv2.imwrite("res_segment.jpg", g1)
