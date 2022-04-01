#%% Resim yeniden boyutlandırma ve kırpma.
import cv2

img = cv2.imread("lena.png")
print("Resim boyutu", img.shape)
cv2.imshow("Orjinal",img)

imgResized = cv2.resize(img, (800,800))
print("Resized image shape", imgResized.shape)
cv2.imshow("Resized image", imgResized)

#crop

imgCropped = img[:200,:300] #normally width-height but in opencv height-width 
cv2.imshow("Cropped image", imgCropped)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Şekil ve metin ekleme
import cv2
import numpy as np

img1 = np.zeros((512,512,3), np.uint8) #siyah bir resim
print(img1.shape)

cv2.imshow("Siyah", img1)

#çizgi
#resim başlangıç noktası, bitiş noktası, renk, kalınlık

cv2.line(img1,(100,100),(100,300),(0,255,0),3) #RGB yerine BRG kullanılır. (0,255,0)
cv2.imshow("Cizgi", img1)

#dikdörtgen
#(resim, başlangıç, bitiş,renk)

cv2.rectangle(img1,(0,0), (256,256), (255,0,0), cv2.FILLED) #♠cv2.filled içini doldurur.
cv2.imshow("Dikdortgen", img1)


#cember
#(resim,yaricapi,renk) içini doldurursam daire olur.

cv2.circle(img1,(300,300), 45, (0,0,255), cv2.FILLED)
cv2.imshow("Daire", img1)

#metin
#(resim,başlangıç,bitiş,font ,kalınlık,renk)

cv2.putText(img1,"Image",(350,350), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255))
cv2.imshow("Text", img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
#%% Görüntülerin birleştirilmesi

import cv2
import numpy as np

#resimi ice aktar.
img = cv2.imread("lena.png")
cv2.imshow("original",img)

#yatay
hor = np.hstack((img,img))
cv2.imshow("Horizontal",hor) #veriyi çeşitlendirme için, kedi köpeklerin birleştirilmesi gibi, nesne tespitinde kullanılabilir.

#dikey
ver = np.vstack((img,img))
cv2.imshow("Vertical",ver)

cv2.waitKey(0)
cv2.destroyAllWindows()
#%% Perspektifleri Çarpıtma (Ayarlama)

#Bir perspektifi değiştirmek görüntü işlemede tespit,sınıflandırma algoritmalarında farklı perspektiflerin verileri olması gerekir.

import cv2 
import numpy as np

img = cv2.imread("kart.png")
cv2.imshow("Original",img)

width = 597
height = 833
pts1 = np.float32([[0,0],[0,height],[width,0],[width,height]])
pts2 = np.float32([[260,1],[1,500],[540,180],[410,647]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
print(matrix)
imgOutput = cv2.warpPerspective(img,matrix,(width,height))
cv2.imshow("Ddsfs", imgOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Görüntüleri karıştırmak

import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("foto1.jpg")
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.imread("foto2.jpg")
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

print(img1.shape)
print(img2.shape)

img1 = cv2.resize(img1, (800,400))
img2 = cv2.resize(img2, (800,400))

print(img1.shape)
print(img2.shape)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

#karıştırılmış resim = alpha katsayısı*img1+beta*img2

blended = cv2.addWeighted(src1 = img1, alpha=0.1, src2=img2, beta=0.8, gamma=0)
#alpha=1,beta=0 olsaydı sadece 1.resim ile karşılaşacaktık.

plt.figure()
plt.imshow(blended)


cv2.waitKey(0)
cv2.destroyAllWindows()

#%% GÖRÜNTÜ EŞİKLEME

import cv2 
import matplotlib.pyplot as plt

img = cv2.imread("foto1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#color mapte siyah beyaz yerler daha farklı görünüyor, siyah beyaz görüntü elde etmek için cmap değiştirerek siyah beyaz gösteriyoruz.
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

#threshold = eşikleme

_, thresh_img = cv2.threshold(img, thresh = 200, maxval = 255, type = cv2.THRESH_BINARY)

plt.figure()
plt.imshow(thresh_img, cmap="gray")
plt.show()

#uyarlamalı = adaptive threshold --> global bir eşik değeri yerine algoritma görüntünün küçük bölgesi için
#pikselleri hesaplar. Aynı görüntünün farklı bölgeleri için farklı thresholdlar hesaplar.

thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8) #8 bizim C sabitimiz. Ağırlıklı ortalamadan çıkartılabilecek bir değer. 
                                                                                                                            
plt.figure()
plt.imshow(thresh_img2,cmap="gray")
plt.axis("off")
plt.show()

#%% Bulanıklaştırma - Blurring

import cv2
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#blurring detayı azaltır, gürültüyü engeller.

img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 3
columns = 2

fig.add_subplot(rows,columns,1)
plt.imshow(img), plt.axis("off"), plt.title("Orjinal")

#Ortalama bulanıklaştırma yöntemi

dst2 = cv2.blur(img, ksize=(3,3))
fig.add_subplot(rows,columns,2)
plt.imshow(dst2), plt.axis("off"), plt.title("Ortalama Blur"), plt.show()


#Gaussian Blur

gb  = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=7)
fig.add_subplot(rows,columns,3)
plt.imshow(gb), plt.axis("off"), plt.title("Gaussian Blur"), plt.show()


#Medyan 

mb = cv2.medianBlur(img, ksize=3)
fig.add_subplot(rows,columns,4)
plt.imshow(mb), plt.axis("off"), plt.title("Medyan Blur")

def gaussianNoise(image):
    
    row, col, ch = image.shape
    mean = 0
    var = 0.05
    sigma = var**0.5
    
    gauss = np.random.normal(mean, sigma, (row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    
    return noisy

#içe aktar ve normalize et
#Oluşturulan gürültü 0 ortalamalı bir gürültü olduğu için normalize etmemiz gerekiyor.

img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255

gaussianNoisyImage= gaussianNoise(img)

fig.add_subplot(rows,columns,5)
plt.imshow(gaussianNoisyImage), plt.axis("off"), plt.title("Gauss Noisy"), plt.show()

#gauss blur

gb2 = cv2.GaussianBlur(gaussianNoisyImage, ksize=(3,3), sigmaX=7)

fig.add_subplot(rows,columns,6)
plt.imshow(gb2), plt.axis("off"), plt.title("with Gauss Blur")


#%% Morfolojik Operasyonlar 
"""
    1.EROZYON -> Ön plandaki resmin sınırlarını aşındırır.
    2.GENİŞLEME -> Erozyonun zıttıdır. Beyaz bölgeleri artırır.
    3.AÇMA -> Erozyon+genişleme = gürültünün giderilmesinde faydalıdır.
    4.KAPATMA -> AÇmanın tam tersi, genişleme+erozyon. Nesne üzerindeki siyah noktaları kapatmak için kullanışlıdır.
    5.MORFOLOJİK GRADYAN -> Genişleme - Erozyondur. 
"""


import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("suheda.png", 0) #resmi siyah beyaz olarak içe aktarmalıyız. Gürültü eklerken sorun çıkmasın.

fig = plt.figure(figsize=(15, 15))
fig.add_subplot(5,2,1), plt.imshow(img, cmap = 'gray'), plt.axis("off"), plt.title("Orjinal Resim") 

#erozyon    
kernel = np.ones((5,5), dtype = np.uint8)
result = cv2.erode(img, kernel, iterations = 1) #iteration kaç kez erozyon yapacagıdır. 
fig.add_subplot(5,2,2), plt.imshow(result, cmap = 'gray'), plt.axis("off"), plt.title("Erozyon Resim") 

#Genişleme - Dilation
result2 = cv2.dilate(img, kernel, iterations = 1) #iteration kaç kez erozyon yapacagıdır. 
fig.add_subplot(5,2,3), plt.imshow(result2, cmap = 'gray'), plt.axis("off"), plt.title("Erozyon Resim") 

#white noise-> siyahlar ve beyazlardan oluşan bir resim elde ederek orjinal resme gürültü ekleyerek gürültü görüntü elde edeceğiz.
whiteNoise = np.random.randint(0,2,size = img.shape[:2]) #resime beyaz gürültü ekleme. = 0 ve 2 arasında  resmin boyutu kadar olan bir np.array oluşturuyoruz.
whiteNoise = whiteNoise*255 # 0 ve 2 arasında üretilen array normalize edilmiş hali olur 255 ile istediğim skalaya çıkartıyorum.
fig.add_subplot(5,2,4), plt.imshow(whiteNoise, cmap = 'gray'), plt.axis("off"), plt.title("Beyaz Gürültülü Resim") 

noise_img = whiteNoise + img
fig.add_subplot(5,2,5), plt.imshow(noise_img, cmap = 'gray'), plt.axis("off"), plt.title("Gürültülü(Beyaz) Resim") 

#açılma
opening = cv2.morphologyEx(noise_img.astype(np.float32), cv2.MORPH_OPEN, kernel)
fig.add_subplot(5,2,6), plt.imshow(opening, cmap = 'gray'), plt.axis("off"), plt.title("Acilma") 

#black noise-> siyah gürültü elde edeceğiz. 
blackNoise = np.random.randint(0,2,size = img.shape[:2]) #resime siyah gürültü ekleme. = 0 ve 2 arasında  resmin boyutu kadar olan bir np.array oluşturuyoruz.
blackNoise = blackNoise*-255 # 0 ve 2 arasında üretilen array normalize edilmiş hali olur 255 ile istediğim skalaya çıkartıyorum.
fig.add_subplot(5,2,7), plt.imshow(blackNoise, cmap = 'gray'), plt.axis("off"), plt.title("Siyah Gürültülü Resim") 

black_noise_img = blackNoise + img
black_noise_img[black_noise_img <= -245] = 0
fig.add_subplot(5,2,8), plt.imshow(black_noise_img, cmap = 'gray'), plt.axis("off"), plt.title("Gürültülü(Siyah) Resim")

#kapatma
closing = cv2.morphologyEx(black_noise_img.astype(np.float32), cv2.MORPH_CLOSE, kernel)
fig.add_subplot(5,2,9), plt.imshow(closing, cmap = 'gray'), plt.axis("off"), plt.title("Kapama") 

#gradient = edge detection, kenar tespiti
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
fig.add_subplot(5,2,10), plt.imshow(gradient, cmap = 'gray'), plt.axis("off"), plt.title("Gradient") 

#%% Gradyanlar - Kenar Algılamada kullanabiliriz.

import cv2
import matplotlib.pyplot as plt
 
img = cv2.imread("sudoku.png", 0)
fig = plt.figure(figsize=(6,6))
fig.add_subplot(2,2,1), plt.imshow(img, cmap="gray"), plt.axis("off"), plt.title("Original Image")

#x gradyan
sobelx = cv2.Sobel(img, ddepth= cv2.CV_16S, dx = 1, dy = 0, ksize=5)
fig.add_subplot(2,2,2), plt.imshow(sobelx, cmap="gray"), plt.axis("off"), plt.title("Sobel X Image")

#y gradyan
sobely = cv2.Sobel(img, ddepth= cv2.CV_16S, dx = 0, dy = 1, ksize=5)
fig.add_subplot(2,2,3), plt.imshow(sobely, cmap="gray"), plt.axis("off"), plt.title("Sobel Y Image")

#laplacian gradyan

laplacian= cv2.Laplacian(img, ddepth = cv2.CV_64F)
fig.add_subplot(2,2,4), plt.imshow(laplacian, cmap="gray"), plt.axis("off"), plt.title("laplacian Image")

#%% Histogram - Görüntüdeki renk dağılımını gösterir.

import cv2 
import matplotlib.pyplot as plt
import numpy as np
 
img = cv2.imread("images/red_blue.png")
img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(12,12))
fig_X= 4
fig_Y= 2

fig.add_subplot(fig_X, fig_Y, 1), plt.imshow(img_vis), plt.title("Original Image")

print(img.shape)

img_hist = cv2.calcHist([img], channels= [0], mask= None, histSize = [256], ranges= [0,256])
print(img_hist.shape)
fig.add_subplot(fig_X, fig_Y, 2), plt.plot(img_hist), plt.title("Hist Image")

color = ("b","g","r")
fig.add_subplot(fig_X, fig_Y, 3)
for i,c in enumerate(color):
    hist = cv2.calcHist([img], channels=[i], mask=None, histSize=[256], ranges=[0,256])
    plt.plot(hist,color=c)

#####

golden_gate = cv2.imread("images/golden_gate.jpg")
golden_gate_vis = cv2.cvtColor(golden_gate, cv2.COLOR_BGR2RGB)
fig.add_subplot(fig_X, fig_Y, 4), plt.imshow(golden_gate_vis), plt.title("Golden Gate Image")

print(golden_gate.shape)

mask = np.zeros(golden_gate.shape[:2], np.uint8) 
fig.add_subplot(fig_X, fig_Y,5), plt.imshow(mask, cmap="gray"), plt.title("Mask")

mask[500:800, 200:500] = 255
fig.add_subplot(fig_X, fig_Y,6), plt.imshow(mask, cmap="gray"), plt.title("Masked Region")

masked_img_vis = cv2.bitwise_and(golden_gate_vis, golden_gate_vis, mask=mask)
fig.add_subplot(fig_X, fig_Y,7), plt.imshow(masked_img_vis, cmap="gray"), plt.title("Masked Image")

masked_img = cv2.bitwise_and(golden_gate, golden_gate, mask=mask)
masked_img_hist = cv2.calcHist([golden_gate], channels= [2], mask= mask, histSize = [256], ranges= [0,256]) #channels degerlerini degistirerek farklı renkk hist. dagılımlarını gorsellestirebilirsiniz.
fig.add_subplot(fig_X, fig_Y, 8), plt.plot(masked_img_hist), plt.title("Hist Masked Image")

### histogram eşitleme , karşıtlık artırma

img = cv2.imread("images/lena_histogram.png", 0)

fig2 = plt.figure(figsize=(10,10))
fig2_X= 2
fig2_Y= 2

fig2.add_subplot(fig2_X, fig2_Y,1), plt.imshow(img, cmap="gray")

img_hist = cv2.calcHist([img], channels= [0], mask= None, histSize = [256], ranges= [0,256])
fig2.add_subplot(fig2_X, fig2_Y,2), plt.plot(img_hist)

eq_hist = cv2.equalizeHist(img)
fig2.add_subplot(fig2_X, fig2_Y,3), plt.imshow(eq_hist, cmap="gray")

eq_img_hist = cv2.calcHist([eq_hist], channels= [0], mask= None, histSize = [256], ranges= [0,256])
fig2.add_subplot(fig2_X, fig2_Y,4), plt.plot(eq_img_hist)






























 



