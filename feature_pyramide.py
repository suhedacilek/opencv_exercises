# -*- coding: utf-8 -*-
"""


opencv'de image.shape --> height,width, channels döndürür.


"""

import cv2
import matplotlib.pyplot as plt



def image_pyramide(image, scale = 1.5, minSize=(224,224)):
    
    yield image
    
    while True:
        
        w = int(image.shape[1]/scale)
        image = cv2.resize(image, dsize=(w,w))
        
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        yield image
        
img = cv2.imread("lena.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
im = image_pyramide(img, 1.5, (10,10))
for i, image in enumerate(im):
    print(i)
    
    if i==3:
        plt.imshow(image)