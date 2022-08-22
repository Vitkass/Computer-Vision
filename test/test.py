from cProfile import label
from cmath import rect
from email.mime import image
from importlib.resources import path
from ssl import _create_default_https_context
import matplotlib.pyplot as plt
import cv2 
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import utils




img = cv2.imread('../photos/3d.jpg')
Y, X= img.shape[:2]
list_pst=[]

out_path = '../result/persp.jpg'


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
            cv2.imwrite(out_path, dst)
            return dst
            



cv2.namedWindow("original_img")
cv2.setMouseCallback("original_img", on_EVENT_LBUTTONDOWN)

cv2.imshow("original_img", img)


cv2.waitKey(0)
cv2.destroyAllWindows()
