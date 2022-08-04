from __future__ import print_function
from __future__ import division
from email.mime import image
import cv2
import numpy as np
from matplotlib import pyplot as plt


original = cv2.imread('../photos/noise.jpg')
#path = self.photo.split('/')
#name = path[len(path)-1]
#out_path = '../result/' + name
doc = cv2.imread('../result/doc.jpg')



dst = cv2.fastNlMeansDenoisingColored(doc,None,10,10,7,21)


plt.figure(figsize=(13, 8))

plt.subplot(121),plt.imshow(dst)
plt.title('Change Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(doc)
plt.title('Original Image'), plt.xticks([]), plt.yticks([]) 
plt.show()



