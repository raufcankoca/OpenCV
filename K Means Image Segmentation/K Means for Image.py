import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('sample.png')

#convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = img.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

print('Image Shape : ',pixel_values.shape)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.2)

# number of clusters (K)
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values,3,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)


print(centers)

# flatten the labels array
labels = labels.flatten()

print(centers[labels.flatten()])

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(img.shape)
# show the image
plt.imshow(segmented_image)
plt.show()

