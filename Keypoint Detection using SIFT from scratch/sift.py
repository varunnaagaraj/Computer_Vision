from __future__ import division
import cv2
import numpy as np

white_image = cv2.imread('white.png')
black_image = ~white_image
black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
image_original = cv2.imread('task2.jpg')
image = cv2.imread('task2.jpg', 0)
cv2.imshow("original", image)
print(image.shape)

# Initializing the 20 Guassian Kernels, dimension = (7x7)
G1 = [([0] * 7) for g in range(7)]
G2 = [([0] * 7) for g in range(7)]
G3 = [([0] * 7) for g in range(7)]
G4 = [([0] * 7) for g in range(7)]
G5 = [([0] * 7) for g in range(7)]
G6 = [([0] * 7) for g in range(7)]
G7 = [([0] * 7) for g in range(7)]
G8 = [([0] * 7) for g in range(7)]
G9 = [([0] * 7) for g in range(7)]
G10 = [([0] * 7) for g in range(7)]
G11 = [([0] * 7) for g in range(7)]
G12 = [([0] * 7) for g in range(7)]
G13 = [([0] * 7) for g in range(7)]
G14 = [([0] * 7) for g in range(7)]
G15 = [([0] * 7) for g in range(7)]
G16 = [([0] * 7) for g in range(7)]
G17 = [([0] * 7) for g in range(7)]
G18 = [([0] * 7) for g in range(7)]
G19 = [([0] * 7) for g in range(7)]
G20 = [([0] * 7) for g in range(7)]

gaussian_kernel = [[G1, G2, G3, G4, G5], [G6, G7, G8, G9, G10], [G11, G12, G13, G14, G15], [G16, G17, G18, G19, G20]]
maxima = {}


def show_images(image_arrays, count):
    for i in range(0,5):
        cv2.imwrite("Octave_{}{}.jpg".format(count,i),np.asarray(image_arrays[i]))


def scale_down_image(image, orig_height, orig_width, scaled_width, scaled_height):

    # Using Nearest Neighbor Algorithm to scale the image
    orig_width = int(orig_width)
    orig_height = int(orig_height)
    scaled_height = int(scaled_height)
    scaled_width = int(scaled_width)
    scaled_image = [([0] * scaled_width) for s in range(scaled_height)]
    for h in range(0, scaled_height):
        for w in range(0, scaled_width):
            n_w = int(round(float(w) / float(scaled_width) * float(orig_width)))
            n_h = int(round(float(h) / float(scaled_height) * float(orig_height)))
            index_w = min(n_w, orig_width - 1)
            index_h = min(n_h, orig_height - 1)
            scaled_image[h][w] = image[index_h][index_w]

    return scaled_image


def kernel_generation(rows, columns, image, gaussian_kernel_list):
    rows = int(rows)
    columns = int(columns)
    h1 = [([0] * columns) for h in range(rows)]
    h2 = [([0] * columns) for h in range(rows)]
    h3 = [([0] * columns) for h in range(rows)]
    h4 = [([0] * columns) for h in range(rows)]
    h5 = [([0] * columns) for h in range(rows)]
    for i in range(3, rows - 3):
        for j in range(3, columns - 3):
            for u in range(7):
                for v in range(7):
                    h1[i][j] += float(gaussian_kernel_list[0][u][v] * image[i + u - 3][j + v - 3])
                    h2[i][j] += float(gaussian_kernel_list[1][u][v] * image[i + u - 3][j + v - 3])
                    h3[i][j] += float(gaussian_kernel_list[2][u][v] * image[i + u - 3][j + v - 3])
                    h4[i][j] += float(gaussian_kernel_list[3][u][v] * image[i + u - 3][j + v - 3])
                    h5[i][j] += float(gaussian_kernel_list[4][u][v] * image[i + u - 3][j + v - 3])

    for i in range(3, rows - 3):
        for j in range(3, columns - 3):
            h1[i][j] = (h1[i][j] / find_max(h1[i]))*255.0
            h2[i][j] = (h2[i][j] / find_max(h2[i]))*255.0
            h3[i][j] = (h3[i][j] / find_max(h3[i]))*255.0
            h4[i][j] = (h4[i][j] / find_max(h4[i]))*255.0
            h5[i][j] = (h5[i][j] / find_max(h5[i]))*255.0
    return [h1, h2, h3, h4, h5]

def find_max(arr):
    max_of_items = 0
    for item in arr:
        if item > max_of_items:
            max_of_items = item
    if max_of_items:
        return max_of_items
    else:
        return 1


def gaussian_filter(sigma, gaussian_kernel_list):
    c_value = [0 for g in range(5)]
    for z in range(-3, 4):
        for k in range(-3, 4):
            gaussian_kernel_list[0][z + 3][k + 3] = np.math.exp(-1 * ((z ** 2 + k ** 2) / (2 * (sigma[0] ** 2)))) / (
                    2 * np.math.pi * (sigma[0] ** 2))
            c_value[0] += gaussian_kernel_list[0][z+3][k+3]
            gaussian_kernel_list[1][z + 3][k + 3] = np.math.exp(-1 * ((z ** 2 + k ** 2) / (2 * (sigma[1] ** 2)))) / (
                    2 * np.math.pi * (sigma[1] ** 2))
            c_value[1] += gaussian_kernel_list[1][z + 3][k + 3]
            gaussian_kernel_list[2][z + 3][k + 3] = np.math.exp(-1 * ((z ** 2 + k ** 2) / (2 * (sigma[2] ** 2)))) / (
                    2 * np.math.pi * (sigma[2] ** 2))
            c_value[2] += gaussian_kernel_list[2][z + 3][k + 3]
            gaussian_kernel_list[3][z + 3][k + 3] = np.math.exp(-1 * ((z ** 2 + k ** 2) / (2 * (sigma[3] ** 2)))) / (
                    2 * np.math.pi * (sigma[3] ** 2))
            c_value[3] += gaussian_kernel_list[3][z + 3][k + 3]
            gaussian_kernel_list[4][z + 3][k + 3] = np.math.exp(-1 * ((z ** 2 + k ** 2) / (2 * (sigma[4] ** 2)))) / (
                    2 * np.math.pi * (sigma[4] ** 2))
            c_value[4] += gaussian_kernel_list[4][z + 3][k + 3]
    return gaussian_kernel_list, c_value


def diff_of_gaussian(im_oct_1, im_oct_2, row, col):
    dog = [([0]*int(col)) for d in range(int(row))]
    for i in range(0, int(row)):
        for j in range(0, int(col)):
            dog[i][j] = im_oct_1[i][j] - im_oct_2[i][j]
    return np.asarray(dog)


def find_keypoints(main_dog, top_dog, bottom_dog, rows, columns, count):
    rows = int(rows)
    columns = int(columns)
    maxima_list = []
    for row in range(1, rows - 1):
        for col in range(1, columns - 1):
            if (main_dog[row][col] > main_dog[row - 1][col - 1]) and (main_dog[row][col] > main_dog[row - 1][col]) and (
                    main_dog[row][col] > main_dog[row - 1][col + 1]) and (
                    main_dog[row][col] > main_dog[row][col - 1]) and (main_dog[row][col] > main_dog[row][col + 1]) and (
                    main_dog[row][col] > main_dog[row + 1][col - 1]) and (
                    main_dog[row][col] > main_dog[row + 1][col]) and (main_dog[row][col] > main_dog[row + 1][col + 1]):
                if (main_dog[row][col] > top_dog[row - 1][col - 1]) and (
                        main_dog[row][col] > top_dog[row - 1][col]) and (
                        main_dog[row][col] > top_dog[row - 1][col + 1]) and (
                        main_dog[row][col] > top_dog[row][col - 1]) and (main_dog[row][col] > top_dog[row][col]) and (
                        main_dog[row][col] > top_dog[row][col + 1]) and (
                        main_dog[row][col] > top_dog[row + 1][col - 1]) and (
                        main_dog[row][col] > top_dog[row + 1][col]) and (
                        main_dog[row][col] > top_dog[row + 1][col + 1]):
                    if (main_dog[row][col] > bottom_dog[row - 1][col - 1]) and (
                            main_dog[row][col] > bottom_dog[row - 1][col]) and (
                            main_dog[row][col] > bottom_dog[row - 1][col + 1]) and (
                            main_dog[row][col] > bottom_dog[row][col - 1]) and (
                            main_dog[row][col] > bottom_dog[row][col]) and (
                            main_dog[row][col] > bottom_dog[row][col + 1]) and (
                            main_dog[row][col] > bottom_dog[row + 1][col - 1]) and (
                            main_dog[row][col] > bottom_dog[row + 1][col]) and (
                            main_dog[row][col] > bottom_dog[row + 1][col + 1]):
                        if count != 0:
                            maxima_list.append((int(row*count/0.75),int(col*count/0.75)))
                        else:
                            maxima_list.append((row, col))
                            
            elif (main_dog[row][col] < main_dog[row - 1][col - 1]) and (main_dog[row][col] < main_dog[row - 1][col]) and (
                    main_dog[row][col] < main_dog[row - 1][col + 1]) and (
                    main_dog[row][col] < main_dog[row][col - 1]) and (main_dog[row][col] < main_dog[row][col + 1]) and (
                    main_dog[row][col] < main_dog[row + 1][col - 1]) and (
                    main_dog[row][col] < main_dog[row + 1][col]) and (main_dog[row][col] < main_dog[row + 1][col + 1]):
                if (main_dog[row][col] < top_dog[row - 1][col - 1]) and (
                        main_dog[row][col] < top_dog[row - 1][col]) and (
                        main_dog[row][col] < top_dog[row - 1][col + 1]) and (
                        main_dog[row][col] < top_dog[row][col - 1]) and (main_dog[row][col] < top_dog[row][col]) and (
                        main_dog[row][col] < top_dog[row][col + 1]) and (
                        main_dog[row][col] < top_dog[row + 1][col - 1]) and (
                        main_dog[row][col] < top_dog[row + 1][col]) and (
                        main_dog[row][col] < top_dog[row + 1][col + 1]):
                    if (main_dog[row][col] < bottom_dog[row - 1][col - 1]) and (
                            main_dog[row][col] < bottom_dog[row - 1][col]) and (
                            main_dog[row][col] < bottom_dog[row - 1][col + 1]) and (
                            main_dog[row][col] < bottom_dog[row][col - 1]) and (
                            main_dog[row][col] < bottom_dog[row][col]) and (
                            main_dog[row][col] < bottom_dog[row][col + 1]) and (
                            main_dog[row][col] < bottom_dog[row + 1][col - 1]) and (
                            main_dog[row][col] < bottom_dog[row + 1][col]) and (
                            main_dog[row][col] < bottom_dog[row + 1][col + 1]):
                        if count != 0:
                            maxima_list.append((int((row/0.75)*count),int((col/0.75)*count)))
                        else:
                            maxima_list.append((row, col))
    maxima.update({count: maxima_list})


def normalize_kernel(kernel_array, c_value, count):
    for row in range(5):
        for col in range(5):
            kernel_array[row][col] /= c_value
    return kernel_array


octaves = np.asarray([[1.0 / np.sqrt(2), 1.0, np.sqrt(2.0), 2.0, 2.0 * np.sqrt(2)],
                      [np.sqrt(2.0), 2.0, 2.0 * np.sqrt(2), 4.0, 4.0 * np.sqrt(2)],
                      [2.0 * np.sqrt(2), 4.0, 4.0 * np.sqrt(2), 8.0, 8.0 * np.sqrt(2)],
                      [4.0 * np.sqrt(2), 8.0, 8.0 * np.sqrt(2), 16.0, 16.0 * np.sqrt(2)]], dtype=np.float)
count = 0
for sigma_values in octaves:
    gaussian_kernel[count], c_value = gaussian_filter(sigma_values, gaussian_kernel[count])
    for values in c_value:
        gaussian_kernel[count] = normalize_kernel(gaussian_kernel[count], values, c_value.index(values))
    count += 1

filtered_images = {}
count = 0

# Reduce the image to 0.75 of the previous output
filtered_images[count] = kernel_generation(458, 750,image, gaussian_kernel[count])
image_out = scale_down_image(image, 458, 750, 563, 344)
show_images(filtered_images[count], count)
count += 1

# Reduce the image to 0.75 of the previous output
filtered_images[count] = kernel_generation(344, 563, image_out, gaussian_kernel[count])
image_out_1 = scale_down_image(image_out, 344, 563, 423, 258)
show_images(filtered_images[count], count)
count += 1

# Reduce the image to 0.75 of the previous output
filtered_images[count] = kernel_generation(258, 423, image_out_1, gaussian_kernel[count])
image_out_2 = scale_down_image(image_out_1, 258, 423, 318, 194)
show_images(filtered_images[count], count)
count += 1

# Reduce the image to 0.75 of the previous output
filtered_images[count] = kernel_generation(194, 318, image_out_2, gaussian_kernel[count])
image_out_3 = scale_down_image(image_out_1, 194, 318, 239, 146)
show_images(filtered_images[count], count)
count += 1


def normalize(arr, row, col):
    max_value = 0
    for i in range(row):
        for j in range(col):
            if max_value < arr[i][j]:
                max_value = arr[i][j]
    for i in range(row):
        for j in range(col):
            arr[i][j] = (arr[i][j] / max_value)*255
    return arr

# Calculate the DoG's between images
L_11 = diff_of_gaussian(filtered_images[0][0], filtered_images[0][1], 458, 750)
L_12 = diff_of_gaussian(filtered_images[0][1], filtered_images[0][2], 458, 750)
L_13 = diff_of_gaussian(filtered_images[0][2], filtered_images[0][3], 458, 750)
L_14 = diff_of_gaussian(filtered_images[0][3], filtered_images[0][4], 458, 750)
normalize(L_11, 458,750)
normalize(L_12, 458, 750)
normalize(L_13,458, 750)
normalize(L_14, 458, 750)
cv2.imwrite("dog1.jpg",np.asarray(L_11))
cv2.imwrite("dog2.jpg",np.asarray(L_12))
cv2.imwrite("dog3.jpg",np.asarray(L_13))
cv2.imwrite("dog4.jpg",np.asarray(L_14))

L_21 = diff_of_gaussian(filtered_images[1][0], filtered_images[1][1], 344,563)
L_22 = diff_of_gaussian(filtered_images[1][1], filtered_images[1][2], 344,563)
L_23 = diff_of_gaussian(filtered_images[1][2], filtered_images[1][3], 344,563)
L_24 = diff_of_gaussian(filtered_images[1][3], filtered_images[1][4], 344,563)
normalize(L_21, 344,563)
normalize(L_22, 344, 563)
normalize(L_23,344, 563)
normalize(L_24, 344, 563)
cv2.imwrite("dog5.jpg",np.asarray(L_21))
cv2.imwrite("dog6.jpg",np.asarray(L_22))
cv2.imwrite("dog7.jpg",np.asarray(L_23))
cv2.imwrite("dog8.jpg",np.asarray(L_24))

L_31 = diff_of_gaussian(filtered_images[2][0], filtered_images[2][1], 258, 423)
L_32 = diff_of_gaussian(filtered_images[2][1], filtered_images[2][2], 258, 423)
L_33 = diff_of_gaussian(filtered_images[2][2], filtered_images[2][3], 258, 423)
L_34 = diff_of_gaussian(filtered_images[2][3], filtered_images[2][4], 258, 423)
normalize(L_31, 258,423)
normalize(L_32, 258, 423)
normalize(L_33,258, 423)
normalize(L_34, 258, 423)
cv2.imwrite("dog9.jpg",np.asarray(L_31))
cv2.imwrite("dog10.jpg",np.asarray(L_32))
cv2.imwrite("dog11.jpg",np.asarray(L_33))
cv2.imwrite("dog12.jpg",np.asarray(L_34))

L_41 = diff_of_gaussian(filtered_images[3][0], filtered_images[3][1], 194, 318)
L_42 = diff_of_gaussian(filtered_images[3][1], filtered_images[3][2], 194, 318)
L_43 = diff_of_gaussian(filtered_images[3][2], filtered_images[3][3], 194, 318)
L_44 = diff_of_gaussian(filtered_images[3][3], filtered_images[3][4], 194, 318)
normalize(L_41, 194,318)
normalize(L_42, 194, 318)
normalize(L_43,194, 318)
normalize(L_44, 194, 318)
cv2.imwrite("dog13.jpg",np.asarray(L_41))
cv2.imwrite("dog14.jpg",np.asarray(L_42))
cv2.imwrite("dog15.jpg",np.asarray(L_43))
cv2.imwrite("dog16.jpg",np.asarray(L_44))

# Find Keypoints of the image
find_keypoints(L_12,L_11,L_13, 458, 750, 0)
find_keypoints(L_13,L_12,L_14, 458, 750, 0)

find_keypoints(L_22,L_21,L_23, 344, 563, 1)
find_keypoints(L_23,L_22,L_24, 344, 563, 1)

find_keypoints(L_32,L_31,L_33, 258, 423, 2)
find_keypoints(L_33,L_32,L_34, 258, 423, 2)

find_keypoints(L_42,L_41,L_43, 194, 318, 3)
find_keypoints(L_43,L_42,L_44, 194, 318, 3)

counter = 0
for count,loc in maxima.items():
    counter += len(loc)
    for point in loc:
        try:
            image_original[point[0]][point[1]] = 255
            black_image[point[0]][point[1]] = 255
        except IndexError:
            continue
cv2.imwrite("keypoints.png",image_original)
cv2.imwrite("blacked.png",black_image)
# cv2.waitKey()

print("Number of Keypoints: {}".format(counter))
count = 0
first_five_points = []
for i in range(black_image.shape[0]):
    for j in range(6):
        if black_image[i][j] == 255 and len(first_five_points)<5:
            first_five_points.append((j, i))
            count += 1
    if count >= 5:
        break
print("The first five keypoints from the left are : {}".format(first_five_points))