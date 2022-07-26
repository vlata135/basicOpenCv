
import cv2
import numpy as np

path = 'src\chessboard.png'

img = cv2.resize(cv2.imread(path), (0,0), fx=0.5, fy=0.5)
img2 = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(img,100,0.2,10)
corners = np.int0(corners) # chuyển từ dạng float sang int để vẽ hình tròn 

#param: 
    #1: source Gray img
    #2: the max corners you want
    #3: the quality of corner
    #4: the minimun dis Euclide

for corner in corners:
    x, y =corner.ravel()
    cv2.circle(img2, (x,y), 3, (0,255,0), -1)

cv2.imshow("img", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()