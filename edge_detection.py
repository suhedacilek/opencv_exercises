import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("images/london.jpg", 0)
fig = plt.figure(figsize=(10,10))

fig_X = 3 
fig_Y = 2
fig.add_subplot(fig_X,fig_Y, 1), plt.imshow(img, cmap="gray"), plt.axis("off")

edges = cv2.Canny(image = img, threshold1=0, threshold2= 255)
fig.add_subplot(fig_X,fig_Y, 2), plt.imshow(edges, cmap="gray"), plt.axis("off")


med_val = np.median(img)
print(med_val)

low = int(max(0, (1-0.33)*med_val))
high = int(min(255,(1+0.33)*med_val))

print(low)
print(high)


edges = cv2.Canny(image = img, threshold1= low, threshold2=high)
fig.add_subplot(fig_X,fig_Y, 3), plt.imshow(edges, cmap="gray"), plt.axis("off")


#blur

blurred_img = cv2.blur(img, ksize=(5,5))
fig.add_subplot(fig_X,fig_Y, 4), plt.imshow(blurred_img, cmap="gray"), plt.axis("off")

med_val = np.median(blurred_img)
print(med_val)

low = int(max(0, (1-0.33)*med_val))
high = int(min(255,(1+0.33)*med_val))

print(low)
print(high)


edges = cv2.Canny(image = blurred_img, threshold1= low, threshold2=high)
fig.add_subplot(fig_X,fig_Y, 5), plt.imshow(edges, cmap="gray"), plt.axis("off")
