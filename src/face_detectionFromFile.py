#coding=utf-8
#使用dlib库检测脸并框出人脸
#导入需要的包
from scipy.spatial import distance as dist #算法库和数学工具包
from imutils.video import FileVideoStream   #文件视频流
from imutils.video import VideoStream   #视频流
from imutils import face_utils
import numpy as np    #数学库，主要用于数组计算
import argparse #命令行解析包
import imutils #图像处理包
import time    #时间包
import dlib #特征点标记库
import cv2    #opencv库
import matplotlib #绘图库
import math

#################################定义变量############################

#1、设置文件路径
#opencv自带的训练好的xml分类器文件路径
#opencv自带的训练好的人脸检测xml分类器文件路径
face_cascade_path='model/haarcascade_frontalface_default.xml'
#dlib预训练的面部标志检测器的路径
facial_landmark_predictor_path='model/shape_predictor_68_face_landmarks.dat'
#加载我们的Haar级联分类器
detector = cv2.CascadeClassifier(face_cascade_path)
#读取图片
img=cv2.imread("img/face.jpg")
#4、转换为灰度图像进行预处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#检测灰度图像中的面部 
rects = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE)
#循环面部检测,提取rects检测的坐标和宽度+高度，OpenCV样式的边界框[即（x，y，w，h）
for (x, y, w, h) in rects:
    #显示面部边框(坐标int类型)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
#显示图像
cv2.imshow("face", img)
cv2.waitKey(0) 