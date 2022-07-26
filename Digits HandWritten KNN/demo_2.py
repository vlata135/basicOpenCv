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
train = np_cells[:,:100].reshape(-1,400).astype(np.float32)
print(train.shape)
# dán nhãn cho dữ liệu để train
k = np.arange(10)
trainLabels = np.repeat(k,500)[:, np.newaxis]
print(trainLabels)
#shape của train là (5000, 400) tức là có 5000 dòng, mỗi dòng chứa mỗi ảnh được đưa về dạng 400 pixel năm ngang
#còn shape của Train label là (5000,1) tức là có 5000 dòng, mỗi số 1 được in 500 lần để dãn nhãn 
# hàm newaxis ( tạo trục mới ) tạo ra để đưa các nhãn về thành mảng 1 chiều để train 
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, trainLabels)
ret,result,neighbours,dist = knn.findNearest(testing,k=3)

# print(result)


# hàng xịn





