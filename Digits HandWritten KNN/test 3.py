import cv2
import numpy as np

img = cv2.imread("src/digits.png")
imgTest = cv2.imread("src/2.png")
imgTest_Gray = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cells = [np.hsplit(row, 100) for row in np.vsplit(imgGray,50)]
np_cells = np.array(cells)
#chuẩn bị dữ liệu test ( dữ liệu test cũng phải xử lý để được đưa về 1 hàng)
data_test = np.array(imgTest_Gray)
testing = data_test[:,:].reshape(-1, 400).astype(np.float32)

#chuẩn bị dữ liệu để train
train = np_cells[:1,:1].reshape(-1,400).astype(np.float32)
print(train)
# dán nhãn cho dữ liệu để train
k = np.arange(10)
trainLabels = np.repeat(k,9)[:, np.newaxis]
# print(trainLabels)
