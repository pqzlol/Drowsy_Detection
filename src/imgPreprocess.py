#coding=utf-8
import cv2
import random as r
import numpy as np
import skimage
from skimage import util
from skimage import  measure
import math

#添加椒盐噪声函数
def sp(src,p=0.1):
    a=src.copy()
    for i in range(int(a.shape[0]*a.shape[1]*p)):
        r1=r.randint(0,a.shape[0]-1)
        r2=r.randint(0,a.shape[1]-1)
        r3=r.randint(0,1)*255
        #彩色图像添加噪声
        #a[r1,r2]=(r3,r3,r3)
        #灰度图像添加噪声
        a[r1,r2]=(r3)
    return a

#添加高斯噪声函数 sigma标准差默认1不明显可调大 mean为正可增加图片亮度
def gauss(src,means=0,sigma=1):
    a=src.copy()
    rows=a.shape[0]
    cols=a.shape[1]
    for i in range(rows):
        for j in range(cols):
            g=r.gauss(means,sigma)
            r1=np.where((g+a[i,j])>255,255,(g+a[i,j]))
            r2=np.where(r1<0,0,r1)
            a[i,j]=np.round(r2)
    return a

##########################################################
#计算PSNR和MSE(三种方法)
def psnr(target, ref, scale):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    # assume RGB image
    target_data = np.array(target)
    target_data = target_data[scale:-scale,scale:-scale]
 
    ref_data = np.array(ref)
    ref_data = ref_data[scale:-scale,scale:-scale]
 
    diff = ref_data - target_data
    #flatten()可以将二维的array展成一维的，flatten('C')和flatten('F')的区别在于行向量和列向量。
    diff = diff.flatten('C')
    rmse = math.sqrt( np.mean(diff ** 2.) )
    return 20*math.log10(1.0/rmse)

def psnr1(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0**2/mse)

def psnr2(img1, img2):
    mse = np.mean( (img1/255. - img2/255.) ** 2. )
    print "MSE:",mse
#     if mse < 1.0e-10:
#         return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
##################################################################
#读取图片
#img=cv2.imread('img/xqy.jpg',0)
img=cv2.imread('img/lena.png',0)
#cv2.imwrite("img/xqy1.jpg", img);
print '图像类型：',img.dtype

#添加椒盐噪声
t1=cv2.getTickCount()
#img1=util.random_noise(img,mode='s&p',amount=0.01,seed=None, clip=True)
img1=sp(img)
t2=cv2.getTickCount()
t=(t2-t1)/cv2.getTickFrequency()
cv2.imwrite("img/jy.jpg", img1);
print "添加椒盐噪声时间:",t
cv2.imshow('jiaoyan img',img1)

#均值滤波
t1=cv2.getTickCount()
blur1=cv2.blur(img1,(5,5))
t=(cv2.getTickCount()-t1)/cv2.getTickFrequency()
cv2.imwrite("img/jy-blur1.jpg", blur1);

print'*****均值滤波*******'
print '均值滤波时间：',t
psnr=psnr2(img1,blur1)
print 'PSNR：',psnr
cv2.imshow('jy-blur1 img',blur1)


#中值滤波
t1=cv2.getTickCount()
medianBlur1=cv2.medianBlur(img1,5)
t=(cv2.getTickCount()-t1)/cv2.getTickFrequency()
cv2.imwrite("img/jy-medianBlur1.jpg", medianBlur1);
print'*****中值滤波*******'
print '中值滤波时间：',t
psnr=psnr2(img1,medianBlur1)
print 'PSNR：',psnr
cv2.imshow('jy-medianBlur1 img',medianBlur1)
 
#高斯滤波
t1=cv2.getTickCount()
gaussBlur1=cv2.GaussianBlur(img1,(5,5),1)
t=(cv2.getTickCount()-t1)/cv2.getTickFrequency()
cv2.imwrite("img/jy-gaussBlur1.jpg", gaussBlur1);
 
print'*****高斯值滤波*******'
print '高斯滤波时间：',t
psnr=psnr2(img1,gaussBlur1)
print 'PSNR：',psnr
cv2.imshow('jy-gaussBlur1 img',gaussBlur1)
 
print'*****************'
#添加高斯噪声 均值为0.1 方差为20
t1=cv2.getTickCount()
#img2=util.random_noise(img, seed=None, clip=True)

img2=gauss(img,0,30)
t2=cv2.getTickCount()
t=(t2-t1)/cv2.getTickFrequency()
cv2.imwrite("img/g-img2.jpg", img2);
print "添加高斯噪声时间:",t
cv2.imshow('gaussian  img',img2)
 
#均值滤波
t1=cv2.getTickCount()
blur2=cv2.blur(img2,(5,5))
t=(cv2.getTickCount()-t1)/cv2.getTickFrequency()
cv2.imwrite("img/g-blur2.jpg", blur2);
 
print'*****均值滤波*******'
print '均值滤波时间：',t
psnr=psnr2(img2,blur2)
print 'PSNR：',psnr
cv2.imshow('g-blur2 img',blur2)
 
#中值滤波
t1=cv2.getTickCount()
medianBlur2=cv2.medianBlur(img2,5)
t=(cv2.getTickCount()-t1)/cv2.getTickFrequency()
cv2.imwrite("img/g-medianBlur2.jpg", medianBlur2);
 
print'*****中值滤波*******'
print '中值滤波时间：',t
psnr=psnr2(img2,medianBlur2)
print 'PSNR：',psnr
cv2.imshow('g-medianBlur2 img',medianBlur2)
 
#高斯滤波
t1=cv2.getTickCount()
gaussBlur2=cv2.GaussianBlur(img2,(5,5),0)
t=(cv2.getTickCount()-t1)/cv2.getTickFrequency()
cv2.imwrite("img/g-gaussBlur2.jpg", gaussBlur2);
print'*****高斯滤波*******'
print '高斯滤波时间：',t
psnr=psnr2(img2,gaussBlur2)
print 'PSNR：',psnr
cv2.imshow('g-gaussBlur2 img',gaussBlur2)


cv2.waitKey(0)

