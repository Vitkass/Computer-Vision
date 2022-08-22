from asyncio import BaseTransport
from importlib.resources import path
from matplotlib import pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import logging
import argparse
import os
from photo_class import PhotoReadactor


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('photo', type=str, help="Path to the picture")
    args = parser.parse_args()

    print('Welcom to the photo-redactor!')

    menu = '''If you want to change brightness and contrast, enter 1,
If you want to find edges, enter 2,
If you want to change photo size, enter 3,
If you want to remove noise, enter 4,
If you want to reduce the number of colors, enter 5,
If you want to cover up some colors, enter 6,
If ypu want to find circle, enter 7,
If you want to turn photo, enter 8,
If you want to correct perspective, enter 9,
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

        elif key == '5':
            edit.reduce_color()
            print('Do you want to exit? (yes/no)')
            ans_key = input()
            print('\n')
            
            if ans_key == 'yes':
                break
            elif ans_key == 'no':
                pass
        
        elif key == '6':
            edit.cover_up()
            print('Do you want to exit? (yes/no)')
            ans_key = input()
            print('\n')
            
            if ans_key == 'yes':
                break
            elif ans_key == 'no':
                pass

        elif key == '7':
            edit.find_sercl()
            print('Do you want to exit? (yes/no)')
            ans_key = input()
            print('\n')
            
            if ans_key == 'yes':
                break
            elif ans_key == 'no':
                pass
        
        elif key == '8':
            edit.perspective()
            print('Do you want to exit? (yes/no)')
            ans_key = input()
            print('\n')
            
            if ans_key == 'yes':
                break
            elif ans_key == 'no':
                pass

        elif key == '9':
            edit.perspective()
            print('Do you want to exit? (yes/no)')
            ans_key = input()
            print('\n')
            
            if ans_key == 'yes':
                break
            elif ans_key == 'no':
                pass

        
        elif key == 'change':
            print('Enter new path')
            new_path = str(input())
            edit.change(new_path)            
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
    


    





