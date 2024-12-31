#!/usr/bin/env python
# coding: utf-8

# In[3]:


from PIL import Image
import matplotlib.pyplot as plt

image=Image.open()
plt.imshow()
image.size
image.format
image.modefrom PIL import ImageOps
image_gray=ImageOps.grayscale(image)
image_gray.mode ---------- gives luminiscence
image_gray.save('lenna.jpg')

import cv2
image=cv2.imread(my_image)
image.shape
import matplotlob.pyplot as plt
plt.imshow(image)

# to change color from bgr to rgb
new_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(new_image)image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imwrite('gaurav.jpg',image_gray)im_gray=cv2.imread('barba.png',cv2.IMREAD_GRAYSCALE)
baboon=cv2.imread('baboon.png')
blue,green,red=baboon[:,:,0],baboon[:,:,1],baboon[:,:,2]
# to flip an image
im_flip=cv2.flip(image,0)
im_flip=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)import numpy as np
a=np.zeros((10,10))
a[2:5,5:6]=255
plt.imshow(a)
# # pixel transofmration
import cv2
goldhill=cv2.imread('goldhill')
hist=cv2.calcHist([goldhill],[0],None,[256],[0,255]) n
image=cv2.imread('gaurav.png',cv2.IMREAD_GRAYSCALE)
new_image=cv2.convertScaleAbs(image,alpha=1,beta=100)

#histogram equalisarion
image=cv2.imread('image',cv2.IMREAD_GRAYSCALE)
image=cv2.equalizeHist(image)#thresholding and simple segmentation
def thresholding(input_image,threshold,max_)val=255,min_val=0):
    N,M=input_image.shape
    image_out=np.zeros((N,M).dtype=unint8)
    for i in range(N):
        for j in range(M):
            if input_img[i,j]>threshold:
                image_out[i,j]=max_value
            else:
                image_out[i.j]=min_value
    return image_OUT             image=cv2.imread('gaurav.png',cv2.IMREAD_GRAYSCALE)
 hist=cv2.calcHist(image,[0],None,[255],[0,255])
intensity_values=np.array([x for x in range(hist.shape[0])])
plt.bar(intensity_values,hist[:,0],width=5)
 max_value=255
 threshold=87
 ret,new_image=cv2.threshold(image,threshold,max_value,cv2.THRESH_BINARY)
 ret,otsu=cv2.threshold(image,0,255,cv2.THRESH_OTSU)#geometrric transformation


from PIL import Image
image=Image.open('new.png')
width=image[0]
height=image[1]
new_width=2*width
new_height=2*height
new_image=image.resize((new_width,new_height))
theta=45
image=new_image.rotate(theta)
# In[ ]:


import cv2
image=cv2.imread('image.png')
new_image=cv2.resize(image,None,fx=2,fy=1,interpolation=cv2.INTER_CUBIC)


# In[5]:


get_ipython().system('pip install opencv-python')


# In[3]:


import cv2
import matplotlib.pyplot as plt
img=cv2.imread("test 1.jpg")
plt.imshow(img)

#readdin videos
capture = cv2.VideoCapture(0)
while True:
    isTrue,frame=capture.read()
# In[10]:


#resize and rescale
def rescaleFrame(frame,scale=0.75):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dim=(width,height)
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)


# In[14]:


frame_new=rescaleFrame(frame,scale=0.75)
plt.imshow(frame_new)


# In[ ]:


def changeres(width,height):
    capture.set(4,width)
    capture.set(3,height)


# In[17]:


#draw or write on images
import numpy as np
blank=np.zeros((500,500),dtype='uint8')
plt.imshow(blank)


# In[21]:


#paint this image with certain color
blank=np.zeros((500,500,3),dtype='uint8')
blank[200:300,100:125]=0,0,255
plt.imshow(blank)


# In[28]:


#draw a rectangle
cv2.rectangle(frame,(0,0),(250,500),(0,255,0),thickness=cv2.FILLED)
plt.imshow(frame)


# In[33]:


#DRAW A CICRLCE
cv2.circle(frame,(0,0),(275,500),40,(0,255,0),thickness=3)


# In[36]:


#write text on image
new=np.zeros((500,500),dtype='uint8')
cv2.putText(new,'Heloo',(250,225),cv2.FONT_HERSHEY_TRIPLEX, 1.0,(0,0,255),2)
plt.imshow(new)


# In[37]:


#essential func in opencv



# In[39]:


#convert inmage to grayscale
img=cv2.imread("Image1.png")
plt.subplot(1,2,1)
plt.imshow(img)
gray_scale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.subplot(1,2,2)
plt.imshow(gray_scale)


# In[43]:


#blur an image : basically reducce the noise in an image
blur=cv2.GaussianBlur(img,(8,8),cv2.BORDER_DEFAULT)
plt.imshow(blur)


# In[45]:


#edge cascade
canny = cv2.Canny(img,125,175)
plt.imshow(canny)


# In[49]:


#dilating an image
dilated= cv2.dilate(canny,(7,7),iterations=3)
plt.imshow(dilated)



# In[52]:


#eroding the inmage
eroded= cv2.erode(dilated,(7,7),iterations=3)
plt.imshow(eroded)


# In[55]:


#resize 

new=cv2.resize(img,(640,640))
plt.imshow(new)


# In[56]:


#traslation
def translate(img,x,y):
    transmat=np.float32([[1,0,x],[0,1,y]])
    dimension=(img.shape[1],img.shape[0])
    return cv2.warpAffine(img,transmat,dimension)
#-x----left
#-y----up
#x----right
#y -----down
translated=translate(img,100,100)
plt.imshow(translated)


# In[58]:


def rotate_image(img,angle,rotpoint=None):
    (height,width)=img.shape[:2]
    if rotpoint is None:
        rotpoint=(width//2,height//2)
    rotmat=cv2.getRotationMatrix2D(rotpoint,angle,1.0)
    dim=(width,height)
    return cv2.warpAffine(img,rotmat,dim)


# In[59]:


rotated=rotate_image(img,45)
plt.imshow(rotated)


# In[60]:


#flip image
flip=cv2.flip(img,0)


# In[62]:


#contour detection
img=cv2.imread("Image1.png",0)
img=cv2.Canny(img,125,175)

plt.imshow(img)


# In[ ]:




