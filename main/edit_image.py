from asyncio import BaseTransport
from importlib.resources import path
from matplotlib import pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import logging
import argparse
import os



class PhotoReadactor:

    def __init__(self, photo_path) -> None:
        self.photo = photo_path



    def brightness(self):
        image = cv2.imread(self.photo)
        img = cv2.imread(self.photo)
        path = self.photo.split('/')
        name = path[len(path)-1]
        
            
        print("Enter value of brightness: ")
        alpha = float(input())
        print("Enter value of contrast: ")
        beta = float(input())


        img = np.uint8(np.clip((alpha * image + beta), 0, 255))
        
        
        out_path = '../result/' + name
        cv2.imwrite(out_path, img)


        

    def edges(self):
        
        image = cv2.imread(self.photo)
        path = self.photo.split('/')
        name = path[len(path)-1]
        
            
        print('Enter lower threshold value: ')
        t_lower = int(input())
        print('Enter upper threshold value: ')
        t_upper = int(input())
        print('Enter Aperture size: ')
        aperture_size = int(input())
        L2Gradient = False
            
            
        edge = cv2.Canny(image, t_lower, t_upper, apertureSize = aperture_size, L2gradient = L2Gradient)
                                
        out_path = '../result/' + name
        cv2.imwrite(out_path, edge)



    def show_result(self):
        original = cv2.imread(self.photo)
        path = self.photo.split('/')
        name = path[len(path)-1]
        out_path = '../result/' + name
        change = cv2.imread(out_path)

        plt.figure(figsize=(14, 8))

        plt.subplot(121),plt.imshow(change)
        plt.title('Change Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(original)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([]) 
        plt.show()

        


    def smooth(self):
        image = cv2.imread(self.photo)
        path = self.photo.split('/')
        name = path[len(path)-1]
        print('Enter parameter regulating filter strength for black and white color')
        h = int(input())
        print('Enter parameter regulating filter strength for other colors')
        photo_render = int(input())
        dst = cv2.fastNlMeansDenoisingColored(image,None,h,photo_render,7,21)
        out_path = '../result/' + name
        cv2.imwrite(out_path, dstf)





    
    def resize(self):
        image = cv2.imread(self.photo)
        path = self.photo.split('/')
        name = path[len(path)-1]


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



        out_path = '../result/' + name
        cv2.imwrite(out_path, resized)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('photo', type=str, help="Path to the picture")
    args = parser.parse_args()

    print('Welcom to the photo-redactor!')

    menu = '''If you want to change brightness and contrast, enter 1,
If you want to find edges, enter 2,
If you want to change photo size, enter 3,
If ypu want to remove noise, enter 4,
If you want to change photo path, print change,
If you want to see last changes, print show,
Else, enter quit'''


    path = args.photo

    edit = PhotoReadactor(photo_path=path)
    
    while True:


        print(menu)
        key = input('Your answer: ')
        
        if key == '1':
            edit.brightness()
            print('Do you want to exit? (yes/no)')
            ans_key = input()
            print('\n')
            
            if ans_key == 'yes':
                break
            elif ans_key == 'no':
                pass


        elif key == '2':
            edit.edges()
            print('Do you want to exit? (yes/no)')
            ans_key = input()
            print('\n')
            
            if ans_key == 'yes':
                break
            elif ans_key == 'no':
                pass

        elif key == '3':
            edit.resize()
            print('Do you want to exit? (yes/no)')
            ans_key = input()
            print('\n')
            
            if ans_key == 'yes':
                break
            elif ans_key == 'no':
                pass

        elif key == '4':
            edit.smooth()
            print('Do you want to exit? (yes/no)')
            ans_key = input()
            print('\n')
            
            if ans_key == 'yes':
                break
            elif ans_key == 'no':
                pass

        
        
        elif key == 'chage':
            print('Enter new path')
            edit.photo = str(input())
            print('\n')

        elif key == 'show':           
            edit.show_result()
            print('\n')


        
        elif key == 'quit':
            break

        else:
            print('\n')
            print('Enter another key')
            print('Help:')
    


    





