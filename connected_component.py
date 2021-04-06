# Import the cv2 library
import cv2
import numpy as np
import time
# Read the image you want connected components of
src = cv2.imread('two_wall.png', cv2.IMREAD_GRAYSCALE)
# Threshold it so it becomes binary
#ret, thresh = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# You need to choose 4 or 8 for connectivity type
connectivity = 4  
start = time.time()
# Perform the operation
output = cv2.connectedComponentsWithStats(src, connectivity, cv2.CV_32S)
print(time.time() - start)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]

# Map component labels to hue val
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue==0] = 0

kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(15,15))
labeled_img[:,1] = cv2.morphologyEx(labeled_img[:,1], cv2.MORPH_CLOSE,kernel)
print(time.time() - start)

cv2.imshow("name", labeled_img)
cv2.waitKey(0)

