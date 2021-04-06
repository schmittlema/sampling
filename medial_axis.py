# Import the cv2 library
import cv2
import numpy as np
import time
from skimage.morphology import medial_axis, skeletonize

extract_cutoff = 0.15 # [0,1] higher value will give you less vertices
map_file = 'two_wall_simple.png'
scale_percent = 0.15 # how much to scale down original image for speedy computation
erosion_kernel = 1 # how big of a kernel to use to smooth out obstacles

src = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)

start = time.time()

# Scale
width = int(src.shape[1] * scale_percent)
height = int(src.shape[0] * scale_percent)
dsize = (width, height)
resized = cv2.resize(src, dsize)
print("resizing: ", time.time() - start)
middle = time.time()

# Threshold it so it becomes binary
ret, thresh = cv2.threshold(resized,210,255,cv2.THRESH_BINARY)
print("thresholding: ", time.time() - middle)
middle = time.time()

# Erode details to simplify for medial axis
kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(erosion_kernel, erosion_kernel))
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)
erode = cv2.erode(closing, kernel, iterations=1)
print("erosion: ", time.time() - middle)
middle = time.time()

#########################################
# Medial axis calc
#skel, distance = medial_axis(erode, return_distance=True)
#skel = skel*distance

# Skeletonize is way faster
erode[erode==255] = 1
skel = skeletonize(erode)
print("medial axis or Skeletonize: ", time.time() - middle)
middle = time.time()
#########################################

# Detect vertices 
dst = cv2.cornerHarris(np.float32(skel),2,3, 0.04)
vertices = np.argwhere(dst>extract_cutoff*dst.max())
print("corners: ", time.time() - middle)
middle = time.time()

print("total: ", time.time() - start)

# visualize
# convert src to 3 channel and resize bc maps are big
final = cv2.resize(cv2.cvtColor(src,cv2.COLOR_GRAY2BGR), (int(src.shape[1]*0.35), int(src.shape[0]*0.35)))

# add lines
lines = np.argwhere(skel)
for linepoint in lines:
    scaled = np.uint16(np.flip(linepoint) * (0.35/scale_percent))
    final = cv2.circle(final, tuple(scaled), 1, (196, 205, 54), 2)

# Add vertices
print("Num Vertices: {}".format(len(vertices)))
for vertex in vertices:
    scaled = np.uint16(np.flip(vertex) * (0.35/scale_percent))
    final = cv2.circle(final, tuple(scaled), 1, (255,0,255), 2)

cv2.imshow("color", final)
cv2.waitKey(0)
cv2.destroyAllWindows()

