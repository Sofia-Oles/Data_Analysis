import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

N_LIDERS = 2


def read_image_in_black_white(file):
    image = cv.imread(file, 0)  # Using 0 to read image in grayscale mode
    cv.imwrite("gray.jpeg", image)
    edges: np.ndarray = cv.Canny(image, 50, 200)
    cv.imwrite("edges.jpeg", edges)
    # print(edges.tolist())
    print(edges.shape)  # M*N
    return image, edges


def calc_distance(start_points, matrix):
    counter = 0
    queue = deepcopy(start_points)
    while len(queue) > 0:
        next_points = set()
        for x, y in queue:
            if matrix[x, y] != -1:
                continue
            matrix[x, y] = counter
            if x + 1 < IMAGE_SIZE[0]:
                next_points.add((x + 1, y))
            if y + 1 < IMAGE_SIZE[1]:
                next_points.add((x, y + 1))
            if x - 1 >= 0:
                next_points.add((x - 1, y))
            if y - 1 >= 0:
                next_points.add((x, y - 1))
        counter += 1
        queue = next_points
    return matrix


def get_score(matrix, image_size, template_size):
    for x in range(image_size[0] - template_size[0]):
        for y in range(image_size[1] - template_size[1]):
            sub_img_sum = matrix[x + templates_white_points[0], y + templates_white_points[1]].sum()
            yield sub_img_sum, x, y


image, edges = read_image_in_black_white("/Users/soles/Desktop/git/Data_Analysis/messi.jpeg")
white_points = np.array(np.where(edges == 255)).T
IMAGE_SIZE = image.shape

print(np.zeros(edges.shape) - 1)

M: np.ndarray = calc_distance(white_points, np.zeros(edges.shape) - 1)

tempalte, templates_edges = read_image_in_black_white("/Users/soles/Desktop/git/Data_Analysis/template.jpeg")
templates_white_points = np.array(np.where(templates_edges == 255))
TEMPLATE_SIZE = tempalte.shape

scorebord = sorted(get_score(M, IMAGE_SIZE, TEMPLATE_SIZE), key=lambda x: x[0])

for i in range(N_LIDERS):
    score, x, y = scorebord[i]
    #  (image, start_point, end_point, color, thickness)
    cv.rectangle(image, (x, y)[::-1], (x + TEMPLATE_SIZE[0], y + TEMPLATE_SIZE[1])[::-1], 255, 2)
    cv.rectangle(edges, (x, y)[::-1], (x + TEMPLATE_SIZE[0], y + TEMPLATE_SIZE[1])[::-1], 255, 2)

plt.subplot(221), plt.imshow(tempalte, cmap='gray')
plt.title('Original Template'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(templates_edges, cmap='gray')
plt.title('Edge Template'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
