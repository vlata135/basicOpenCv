import cv2
import numpy as np
# Import Data
img = cv2.imread("src/digits.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Import Test
imgTest = cv2.imread("src/2.png")
imgTest_Gray = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
# Processing data
cells = [np.hsplit(row, 100) for row in np.vsplit(imgGray,50)] # cvt img to cell size 20x20 v cất ngang, h cắt dọc
np_cells = np.array(cells) # cvt to np cells 
train = np_cells[:,:100].reshape(-1,400).astype(np.float32) # cvt to size (2500,400) (400 chính là đưa ảnh 20x20 về (1,400))
# Labels for data
k = np.arange(10) # tạo các số từ 0 - 9 để dán nhãn
trainLabels = np.repeat(k,500)[:, np.newaxis] # đưa các số về mảng 1 chiều có dạng (2500,1) ( để ý số 1 và 400)
#Processing Test
data_test = np.array(imgTest_Gray) # đưa data về dạng chuỗi của np
testing = data_test[:,:].reshape(-1, 400).astype(np.float32) # đưa ảnh test 20x20 về dạng (1,400)

#Training 
knn = cv2.ml.KNearest_create() # hàm khởi tạo thuật toán train
knn.train(train, cv2.ml.ROW_SAMPLE, trainLabels) # chạy training
#Result 
ret,result,neighbours,dist = knn.findNearest(testing,k=3)
# có 4 tham số là 
    #ret: in ra số không phải ở dạng mảng 
    #result: in ra số ở dạng mảng  
    #neighbour: mấy số gần nó (tham số k chính là số chữ số gần nó lấy ra để so sánh)
    #dist: khoảng cách của nó đến các số gần nhất
print(ret)
print(result)
print(neighbours)
print(dist)



