import matplotlib.pyplot as plt
import numpy as np
import cv2 
from sklearn.cluster import KMeans
import imutils


'''img = cv2.imread('../photos/aba.png')
Y, X= img.shape[:2]
list_pst=[]




def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    list_xy = []
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        list_xy.append(x)
        list_xy.append(y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 0), thickness=1)
        cv2.imshow("original_img", img)
        list_pst.append(list_xy)
        if(len(list_pst)==4):
            # Четыре угла книги на исходном изображении (верхний левый, верхний правый, нижний левый и нижний правый) и положение преобразованной матрицы
            pts1 = np.float32(list_pst)
            pts2 = np.float32([[0, 0], [X, 0], [0, Y], [X, Y]])

            # Создать матрицу перспективного преобразования; выполнить перспективное преобразование
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(img, M, (X, Y))
            cv2.imshow('out_path', dst)
            



cv2.namedWindow("original_img")

cv2.setMouseCallback("original_img", on_EVENT_LBUTTONDOWN)

cv2.imshow("original_img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''
retval, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)


x,y,w,h,s=stats[1,:]

print(x,y,w,h,s)

cv2.rectangle(img, (x, y), (x+w, y+h),(0, 0, 255), 1)
cv2.imshow('name.png', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
u = 'sjdnjsd sjdnjsd jsdnjs'
t = [1, 3, 4]

class Zalupa:

    def __init__(self, photo_path) -> None:
        self.photo = photo_path
        self.out_path = '../result/' + self.photo.split('/')[-1]

    def change(self, name):
        self.photo = name
        self.out_path = '../result/' + self.photo.split('/')[-1]
        



t = Zalupa('../photos/tt.jpg')
t.change('../photos/pp.lp')


print(t.photo)
print(t.out_path)



