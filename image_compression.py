from random import shuffle

from skimage import io
import numpy as np

from sklearn.metrics.cluster import completeness_score, v_measure_score

def flatten(t):
    return [item for sublist in t for item in sublist]


# CONFIG
from KmeansClustering import perform_clustering, get_kmeans_clustering, get_kmeans_clustering_local

image_file_name = "kaladin.jpg"


#Read the image
image = io.imread('images\\' + image_file_name)
io.imshow(image)
# io.show()

#Dimension of the original image
rows = image.shape[0]
cols = image.shape[1]

#Flatten the image
image = image.reshape(rows*cols, 3)
# array of [R, G, B] arrays

# Shuffle so that close pixels are split over the coresets
# shuffle(image)
indices = [i for i in range(len(image))]
shuffle(indices)
mapping = {}
image_shuffled = [None] * len(image)
for old_index, new_index in enumerate(indices):
    mapping[new_index] = old_index
    color_array = image[old_index]
    image_shuffled[new_index] = color_array


# #Implement k-means clustering to form k clusters
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(image)

# #Replace each pixel value with its nearby centroid
# compressed_image = kmeans.cluster_centers_[kmeans.labels_]

print("Start clustering")
(kmeans_clustering, elapsed_time) = get_kmeans_clustering(image_shuffled, 32)
# kmeans_clustering = get_kmeans_clustering_local(result, 64)
print("Finished clustering")

kmeans_clustering_processed = kmeans_clustering.astype('uint8')

# Unshuffle
image_unshuffled = [None] * len(kmeans_clustering_processed)
for i in range(len(kmeans_clustering_processed)):
    color_array = kmeans_clustering_processed[i]
    old_index = mapping[i]
    image_unshuffled[old_index] = color_array

compressed_image = np.clip(image_unshuffled, 0, 255)

score = v_measure_score(flatten(image), flatten(compressed_image))
print("Elapsed time: " + str(elapsed_time) + " | score: " + str(score))

#Reshape the image to original dimension
compressed_image = compressed_image.reshape(rows, cols, 3)

#Save and display output image
io.imsave('images\\compr_' + image_file_name, compressed_image)
io.imshow(compressed_image)
io.show()