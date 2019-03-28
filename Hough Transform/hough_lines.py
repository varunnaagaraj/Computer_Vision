# Created by Varun at 24/11/18
import numpy as np
import cv2


def hough_line(img, theta_range, step_size=1):
    """
    Finds the hough lines present in the image
    :param img: input image
    :param theta_range: Range of theta values that can be checked
    :param step_size: increment of theta values
    :return: accumulator array, theta values and rho values
    """
    thetas = np.deg2rad(np.arange(-theta_range, theta_range, step_size))
    width, height = img.shape
    diag_len = np.ceil(np.sqrt(width * width + height * height))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    accumulator = np.zeros((int(2 * diag_len), num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accumulator[int(rho)][int(t_idx)] += 1
    return accumulator, thetas, rhos


if __name__ == '__main__':
    original_image = cv2.imread("../original_imgs/hough.jpg")

    # Binarize and edge detect the image using Canny edge detector
    img1 = cv2.Canny(original_image, 300, 400)
    cv2.imshow("canny", img1)
    img2 = cv2.Canny(original_image, 100, 200)
    cv2.imshow("canny_1", img2-img1)
    img = img2-img1

    # change the step size to 1 to get red lines and in arg sort keep the idx to
    # have values till -10 and the theta range to be 10.0
    accumulator, thetas, rhos = hough_line(img, 70.0, 2)
    idx = np.argsort(accumulator.ravel(), axis=None)[-31:]
    rho = rhos[idx / accumulator.shape[1]]
    theta = thetas[idx % accumulator.shape[1]]
    for r,t in zip(rho,theta):
        a = np.cos(t)
        b = np.sin(t)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        print(t)
        cv2.line(original_image,(x1,y1),(x2,y2),(0,255,0),2)
        print(x1,x2,y1,y2)
    cv2.imwrite("blue_lines.jpg", original_image)
    cv2.waitKey()
