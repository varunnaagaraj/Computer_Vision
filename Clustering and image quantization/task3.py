import cv2
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')

X = np.array(
    [[5.9, 3.2], [4.6, 2.9], [6.2, 2.8], [4.7, 3.2], [5.5, 4.2], [5.0, 3.0], [4.9, 3.1], [6.7, 3.1], [5.1, 3.8],
     [6.0, 3.0]])


# Uncomment below code if you need to see the scatter plot of the initial values.
# plt.scatter(X[:,0], X[:,1])
# plt.show()

colors = 10 * ['b', 'g', 'r', 'c', 'k']


class KMeans_custom():
    """
    Main Class which performs the Kmeans clustering.
    """
    def __init__(self, num_clusters=3, tolerance=0.0001, epochs=30, centroids={}):
        self.num_clusters = num_clusters
        self.tolerance = tolerance
        self.epochs = epochs
        self.centroids = centroids
        self.first_round_centroids = {}
        self.second_round_centroids = {}

    def fit(self, data):
        if self.centroids == {}:
            for i in range(self.num_clusters):
                self.centroids[i] = data[i]

        for i in range(self.epochs):
            self.classifications = {}

            for cluster in range(self.num_clusters):
                self.classifications[cluster] = []

            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(features)

            prev_centroid = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.mean(self.classifications[classification], axis=0)

            optimized = True
            color = {0: "blue", 1: "green", 2: "red"}
            for cent in self.centroids:
                original_centroid = prev_centroid[cent]
                current_centroid = self.centroids.get(cent)
                if np.sum((original_centroid - current_centroid) / original_centroid * 100) > self.tolerance:
                    optimized = False
                if self.epochs == 1 or self.epochs == 2:
                    plt.scatter(current_centroid[0], current_centroid[1], color=color.get(cent),
                                facecolors=color.get(cent), marker=mmarkers.MarkerStyle(marker='o', fillstyle='none'),
                                edgecolors=color.get(cent))
            plt.savefig("task3_iter{}_b.jpg".format(self.epochs))
            if optimized:
                break

    def predict(self, data):
        classification = list()
        for point in data:
            distances = [np.linalg.norm(point - self.centroids[centroid]) for centroid in self.centroids]
            classification.append(distances.index(min(distances)))
        return np.array(classification)


def part1_2_and_3(epochs):
    """
    Does the task subparts 1,2, and 3
    :param epochs: Number of iterations to run Kmeans
    """
    classifier = KMeans_custom(epochs=epochs, centroids={0: [6.2, 3.2], 1: [6.6, 3.7], 2: [6.5, 3.0]})
    classifier.fit(X)

    for cent in classifier.centroids:
        plt.scatter(classifier.centroids[cent][0], classifier.centroids[cent][1],
                    marker=mmarkers.MarkerStyle(marker='o', fillstyle='full'), color="b", s=150,
                    linewidths=5, facecolors=[1, 1, 1], edgecolors="black", alpha=0.5)

    for classification in classifier.classifications:
        color = colors[classification]
        for feature in classifier.classifications[classification]:
            plt.scatter(feature[0], feature[1], marker=mmarkers.MarkerStyle(marker='^', fillstyle='full'), color=color,
                        s=150, linewidths=5, facecolors=color,
                        edgecolors=color, alpha=0.5)

    print(classifier.classifications)
    print(classifier.centroids)
    plt.savefig("task3_iter{}_a.jpg".format(epochs))


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    image = np.zeros((w, h, 3))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    print("Done with recreation")
    return image


def part_4(num_colors):
    """
    Performs Image Quantization.
    :param num_colors: Specifies the number of colors the image needs to be reduced to.
    """
    image_matrix = cv2.imread('../data/baboon.jpg')
    image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
    image = np.array(image_matrix, dtype=np.float64) / 255
    w, h, d = tuple(image.shape)
    image_array = np.reshape(image, (w * h, d))
    image_quantization = KMeans_custom(num_clusters=num_colors, epochs=30)
    image_quantization.fit(image_array)
    labels = image_quantization.predict(image_array)
    print(image_quantization.centroids)
    new_image = recreate_image(image_quantization.centroids, labels, w, h)
    import matplotlib
    matplotlib.image.imsave('task3_baboon_{}.jpg'.format(num_colors), new_image)


if __name__ == '__main__':
    # Change the number of epochs when you need to run for more number of times.
    part1_2_and_3(epochs=1)

    # Specify the number of colors to which image needs to be reduced to.
    part_4(num_colors=3)
