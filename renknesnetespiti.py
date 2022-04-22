
import cv2
import numpy as np
from collections import deque

buffer_size = 16
pts = deque(maxlen = buffer_size)

skinLower = (84,98,0)
skinUpper = (179,255,255)

cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    
    success, imgOriginal = cap.read()
    
    if success:
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0)        
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)      
        cv2.imshow("hsv image",hsv)
        
        mask = cv2.inRange(hsv,skinLower,skinUpper)
        cv2.imshow("mask",mask)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        cv2.imshow("mask+genisleme", mask)
        
        #kontur
        (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None 
        
        if len(contours) > 0:
            
            c =  max(contours, key = cv2.contourArea) #take max countour
            rect = cv2.minAreaRect(c)
            ((x,y), (width,height), rotation) = rect
            
            s = "x:{}, y:{}, width:{}, height:{}, rotation:{}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(s)
            
            #kutucuk
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            
            M = cv2.moments(c) #moment(görüntü pixellerinin agirlikli ortalamasidir.)
            center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            cv2.drawContours(imgOriginal, [box], 0 , (0,255,255), 2)
            
            cv2.circle(imgOriginal,center,5, (255,0,255), -1) #merkeze bir adet nokta cizelim.pembe
            cv2.putText(imgOriginal, s, (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0,0,0),2) #bilgileri ekrana yazdir.
            
            #deque
            
            pts.appendleft(center) 
            for i in range(1,len(pts)):
                if pts[i-1] is None or pts[i] is None: continue
                
                cv2.line(imgOriginal,pts[i-1], pts[i], (0,255,0),3)
        cv2.imshow("Original Tespit", imgOriginal)
            
    if cv2.waitKey(1) & 0xFF == ord("q"): break
