import cv2 
import numpy as np

img = cv2.imread("src/digits.png")
# img = cv2.imread("src/gai2.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cells = [np.hsplit(row,100) for row in np.vsplit(imgGray,50)]
# x = np.array(cells)

imgCropped = img[:1000,:20]

k = np.arange(3)
# train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
# test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
train_labels = np.repeat(k,5)[:,np.newaxis]
print(train_labels)
cv2.imshow("imda",imgCropped)
cv2.waitKey(0)
# print(x[0])

# cv2.imwrite("number1.png", x[10][0])


# cv2.imshow("imgGray", imgGray)
# cv2.waitKey(0)
