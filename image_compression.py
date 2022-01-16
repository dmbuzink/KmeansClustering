from skimage import io
import numpy as np

# CONFIG
from KmeansClustering import perform_clustering, get_kmeans_clustering

image_file_name = "Lena.png"


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

# #Implement k-means clustering to form k clusters
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(image)

# #Replace each pixel value with its nearby centroid
# compressed_image = kmeans.cluster_centers_[kmeans.labels_]

kmeans_clustering = get_kmeans_clustering(image, 64)

compressed_image = np.clip(kmeans_clustering.astype('uint8'), 0, 255)

#Reshape the image to original dimension
compressed_image = compressed_image.reshape(rows, cols, 3)

#Save and display output image
io.imsave('images\\compr_' + image_file_name, compressed_image)
io.imshow(compressed_image)
io.show()