import cv2
import matplotlib.pyplot as plt
import numpy as np



cap=cv2.VideoCapture(0)


while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    frame=cv2.resize(frame,(400,400))

    # convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = frame.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)

    #print('Image Shape : ', pixel_values.shape)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    #print(centers)

    # flatten the labels array
    labels = labels.flatten()

    #print(centers[labels.flatten()])

    # convert all pixels to the color of the centroids
    segmented_frame = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_frame = segmented_frame.reshape(frame.shape)

    cv2.imshow('K-MEANS',segmented_frame)

    if cv2.waitKey(2)& 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
