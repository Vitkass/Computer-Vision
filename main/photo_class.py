from asyncio import BaseTransport
from importlib.resources import path
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import cv2
import numpy as np
from pathlib import Path
import logging
import argparse
import os
import utils




class PhotoReadactor:

    def __init__(self, photo_path) -> None:
        self.photo = photo_path
        self.out_path = '../result/' + photo_path.split('/')[-1]

    
    def change(self, name):
        self.photo = name
        self.out_path = '../result/' + self.photo.split('/')[-1]



    #Обработка яркости и контрастности
    def brightness(self):
        image = cv2.imread(self.photo)
        img = cv2.imread(self.photo)
        #path = self.photo.split('/')
        #name = path[len(path)-1]
        
            
        print("Enter value of brightness in range (-200, 200): ")
        brightness = float(input())
        print("Enter value of contrast in range (-200, 200): ")
        contrast = float(input())


        img = np.int16(image)
        img = img * (contrast/127+1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        
        
        #out_path = '../result/' + name
        cv2.imwrite(self.out_path, img)



        
    #Поиск кромок
    def edges(self):
        
        image = cv2.imread(self.photo)
        
            
        print('Enter lower threshold value: ')
        t_lower = int(input())
        print('Enter upper threshold value: ')
        t_upper = int(input())
        print('Enter Aperture size: ')
        aperture_size = int(input())
        L2Gradient = False
            
            
        edge = cv2.Canny(image, t_lower, t_upper, apertureSize = aperture_size, L2gradient = L2Gradient)
                                
        cv2.imwrite(self.out_path, edge)


    #Отображает результаты работы одной из функций
    def show_result(self):
        original = cv2.imread(self.photo)
        original = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
        change = cv2.imread(self.out_path)
        change = cv2.cvtColor(change, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(14, 8))

        plt.subplot(121),plt.imshow(change)
        plt.title('Change Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(original)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([]) 
        plt.show()

        

    #Функция, убирающая шум на изображениях
    def smooth(self):
        image = cv2.imread(self.photo)
        print('Enter parameter regulating filter strength for black and white color')
        h = int(input())
        print('Enter parameter regulating filter strength for other colors')
        photo_render = int(input())
        dst = cv2.fastNlMeansDenoisingColored(image,None,h,photo_render,7,21)
        cv2.imwrite(self.out_path, dst)

    
    #Функция, меняющая перспективу фотографии, при запуске необходимо выбрать четыре угла на изображении, порядок приведен ниже
    def perspective(self):
        img = cv2.imread(self.photo)
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
                    cv2.imwrite(self.out_path, dst)
                    return dst
            



        cv2.namedWindow("original_img")
        cv2.setMouseCallback("original_img", on_EVENT_LBUTTONDOWN)

        cv2.imshow("original_img", img)


        cv2.waitKey(0)
        cv2.destroyAllWindows()


        

    #Функция сокращающая цвета на изображении до выбранного количества
    def reduce_color(self):
        image = cv2.imread(self.photo)
        Z = image.reshape((-1,3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        print('Enter the number of colors you want')
        K = int(input())
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image.shape))
        cv2.imwrite(self.out_path, res2)

    # Замазывает выбранный цвет белым
    def cover_up(self):

        bgr_image = cv2.imread(self.photo)
        k = int(input('Enter the number of colors you want to highlight: '))

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        plt.figure()
        plt.axis("off")
        plt.imshow(rgb_image)

        image = rgb_image.reshape((-1,3))
        clt = KMeans(n_clusters = k)
        labels = clt.fit_predict(image)


        hist = utils.centroid_histogram(labels)
        bar = utils.plot_colors(hist, clt.cluster_centers_)
        # show our color bart
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.show()

        print('Enter number of color that you want to cover up')
        color_num = int(input())
        color_num -= 1

        for i in range(len(labels)):
            if color_num == labels[i]:
                image[i] = [255, 255,255]

        image = image.reshape((bgr_image.shape))

        plt.figure()
        plt.axis("off")
        plt.imshow(image)
        plt.show()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(self.out_path, image)

    # Поиск окружностей на изображении
    def find_sercl(self):
        image = cv2.imread(self.photo)

        img_grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        rows = img_grey.shape[0]
        circles = cv2.HoughCircles(img_grey, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                param1=90, param2=45,
                                minRadius=1, maxRadius=100)

        res = np.zeros(image.shape)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(image, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(image, center, radius, (255, 0, 255), 3)

                cv2.imwrite(self.out_path, image)

    
    # Изменение размера
    def resize(self):
        image = cv2.imread(self.photo)


        print("Do you want to decrease (d) or increase (i) ?")
        ans = input()
        interpolation, proper = (cv2.INTER_AREA, 'decrease') if ans == 'd' else (cv2.INTER_LINEAR, 'increase')

        print(f'By what percentage do you want to {proper} the height?')
        heigth_procent = int(input())
        print(f'By what percentage do you want to {proper} the width?')
        width_procent = int(input())

        if ans == 'd':

            width = int(image.shape[1] * (100- width_procent) / 100)
            height = int(image.shape[0] * (100-heigth_procent) / 100)
            dsize = (width, height)
            resized = cv2.resize(image, dsize, interpolation = interpolation)
        
        else:

            width = int(image.shape[1] * (100+width_procent) / 100)
            height = int(image.shape[0] * (100+heigth_procent) / 100)
            dsize = (width, height)
            resized = cv2.resize(image, dsize, interpolation = interpolation)



        cv2.imwrite(self.out_path, resized)





    





