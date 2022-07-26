# Thuật toán KNN
 - tìm người hàng xóm gần nhất
 - Nó sẽ tính khoảng cách của đối tượng cần nhận dạng với dữ liệu đã được train
    -> xem nó gần cái nào nhiều nhất thì nó thuộc cái đấy

- Người ta thường chọn số lượng các điểm cần so sáng là số lẻ chứ k chọn số chẵn vì cơ bản số chẵn sẽ chia được cho 2 (phát triển lên: k chọn số chia hết cho số lượng đối tượng trong dữ liệu đã train)


## 1. Thuật toán KNN là gì
- K-nearest neighbor là một trong những thuật toán supervised-learning đơn giản nhất trong Machine Learning
- Khi training, thuật toán này _không học_ một điều gì từ dữ liệu training -> được xếp vào loại lazy -  learning

## 2. KNN in Handwritten Digits Recognition
### 2.1 Bài toán 
- Data: 1 bức ảnh chứa 50x100 bức ảnh 20x20 pixel chứa các số từ 0 đến 9 dưới dạng viết tay 
- INPUT: bức ảnh chứa con số cần nhận dạng
- OUPUT: Sử dụng các hàm trong thư viên OpenCV để viết chương trình nhận dạng chữ số từ dữ liệu đã cho 

### 2.2 Các bước thực hiện
#### a. Xử lý dữ liệu 
- Dữ liệu đang ở dạng ảnh 1000x2000 pixel, mỗi số 20x20 
=> Cần phải cắt ảnh ban đầu thành các thành phần nhỏ hơn 
- Đưa các ảnh nhỏ về array trong numpy có shape (50,100,20,20)
- Tạo dữ liệu để train:
    - Để có thể sử dụng hàm có sẵn trong cv2, ta phải đưa dữ liệu từ mảng (50,100,20,20) về dạng dạng mảng 2 chiều có shape (2500,400)
    - Hiểu 1 cách đơn giản là đưa các bức ảnh (20,20) về np.array
    (1,400) 

#### b. Tạo nhãn cho từng loại 
- Sau khi xử lý được dữ liệu đưa về dạng mảng 2 chiều (2500, 400) thì ta cần tạo 1 mảng chứa nhãn phải có cùng số hàng, tức là ở dạng (2500, 1)
#### c. Training
- Để train dữ liệu, ta sử dụng các cú pháp sau đây
`knn = cv2.ml.KNearest_create() # hàm khởi tạo thuật toán train`
`knn.train(train, cv2.ml.ROW_SAMPLE, trainLabels) # chạy training`
    - Tham số: cv2.ml.ROW_SAMPLE: chỗ này theo em hiểu là mỗi một hàng của nhãn sẽ tương ứng với một hàng trong data
    - Tương tự với cv2.ml.COL_SAMPLE nhưng là theo cột
 #### d. Xử lý ảnh đưa vào
 - Ở đây ta mặc định ảnh được đưa vào là con số đã được cắt và đưa về dạng 20x20 pixel 
 - Ảnh đưa vào cũng được xử lý giống như những ảnh từng số trong dữ liệu training, tức là cũng sẽ được đưa về mảng (1,400) và type là _np.float32(phần này sẽ được giải thích ở đâu đó [Kiểu dữ liệu](#Ex))
 - Để lấy kết quả ta dùng hàm:
`ret,result,neighbours,dist = knn.findNearest(testing,k=3)`
    - có 4 tham số là :
        - ret: in ra số không phải ở dạng mảng 
        - result: in ra số ở dạng mảng  
        - neighbour: mấy số gần nó (tham số k chính là số chữ số gần nó lấy ra để so sánh)
        - dist: khoảng cách của nó đến các số gần nhất
    - testing: ảnh cần xử lý
    - k: số mẫu gần nó nhất ( ở đây lấy bằng 3)

### 3. Thử nghiệm và đánh giá
#### 3.1 Thử nghiệm
- Khi chạy thử, chương trình đáp ứng 10/10 case trong đó có
  - 5 case em cắt ảnh từ ảnh gốc ban đầu
  - 5 case em tự vẽ trong paint 
#### 3.2 Đánh giá
##### 3.2.1 Ưu và nhược của KNN [em tham khảo tại đây](https://machinelearningcoban.com/2017/01/08/knn/#:~:text=c%E1%BB%A7a%20c%C3%A1c%20class.-,Nh%C6%B0%E1%BB%A3c%20%C4%91i%E1%BB%83m%20c%E1%BB%A7a%20KNN,c%C3%B3%20nhi%E1%BB%81u%20%C4%91i%E1%BB%83m%20d%E1%BB%AF%20li%E1%BB%87u.)
- Ưu điểm:
    -   Độ phức tạp tính toán của quá trình training bằng 0: em thấy nó chỉ đơn giản là dãn nhãn vào dữ liệu mình đưa vào mà chả phải xử lý 1 cái gì cả
    -   Việc dự đoán dữ liệu đưa vào rất đơn giản
    -   **Không cần giả sử gì về phân khối của các class**: với kiến thức hiện tại của em thì em không hiểu câu này lắm 
- Nhược điểm:
  -   KNN rất nhạy cảm với những k nhỏ: cái này được viết rất kĩ trong phần đầu của [bài viết này](https://machinelearningcoban.com/2017/01/08/knn/#:~:text=c%E1%BB%A7a%20c%C3%A1c%20class.-,Nh%C6%B0%E1%BB%A3c%20%C4%91i%E1%BB%83m%20c%E1%BB%A7a%20KNN,c%C3%B3%20nhi%E1%BB%81u%20%C4%91i%E1%BB%83m%20d%E1%BB%AF%20li%E1%BB%87u.)
  -   Đối với các dữ liệu lớn, thời gian xử lý của KNN trong thư viện này cũng rất lớn: Khi chạy code, KNN sẽ tính khoảng cách tới từng Data Point rồi đi so sánh, thời gian sẽ còn lâu hơn nếu K lớn 
  

 