import cv2
import numpy as np


""" Hough Circles
*To detect the circles,we have to work 8-bits images.
*8-bits:Gray images

#circles=cv2.HoughCircles(image,method,dp,minDist,param1,param2,maxRadius,minRadius)

image:source image 
method:the method of detecting circle.cv2.HOUGH_GRADIENT-Just one method
dp:image resolution(the more dp value,the less detected circles.)
param1:gradient value
param2:threshold value
minRadius*: the minimum radius for all circles
maxRadius*: the maximum radius for all circles

"""
#Read the image
img=cv2.imread('planets.jpg')
#Convert image to gray
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Apply blurring
img_blurred=cv2.medianBlur(img_gray,9)

circles=cv2.HoughCircles(img_blurred,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=30,minRadius=0,maxRadius=50)
circles=np.uint16(np.around(circles[0,:]))

#print(circles[0])

for x,y,r in circles:
    #Outer radius
    cv2.circle(img,(x,y),r,(255,0,0),3)
    #Center of the circle
    cv2.circle(img,(x,y),1,(255,0,0),3)

cv2.imshow('CIRCLES',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



