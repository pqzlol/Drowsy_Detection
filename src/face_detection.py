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
face_cascade_path='model/haarcascade_frontalface_default.xml'
#dlib预训练的面部标志检测器的路径
facial_landmark_predictor_path='model/shape_predictor_68_face_landmarks.dat'
#警报的输入音频文件的路径
slightFA_path='alarm/SlightFA.wav'
moderateFA_path='alarm/ModerateFA.wav'
severeFA_path='alarm/SevereFA.wav'

#2、定义系统变量

##################################定义函数#############################################

##1、定义eye_aspect_ratio函数，该函数用于计算垂直眼睛地标与水平眼睛地标之间距离的比率###################
#当眼睛打开时，眼睛纵横比的返回值将近似恒定。该值将在眨眼期间快速减小到零。
#如果眼睛闭合，眼睛纵横比将再次保持近似恒定，但将远小于眼睛打开时的比率。
def eye_aspect_ratio(eye):
    # 计算两组垂直眼睛标志（x，y） - 坐标之间的欧氏距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算两组水平眼睛标志（x，y） - 坐标之间的欧氏距离
    C = dist.euclidean(eye[0], eye[3])
 
    # 计算眼睛纵横比
    ear = (A + B) / (2.0 * C)
 
    # 返回眼睛纵横比
    return ear


############人脸检测#############################
#加载我们的Haar级联和面部地标预测器文件
print("[INFO] loading OpenCV's Haar cascade facial landmark predictor...")
detector = cv2.CascadeClassifier(face_cascade_path)
#detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(facial_landmark_predictor_path)

#从一组面部标志中提取眼睛区域，我们只需要知道正确的阵列切片索引，使用这些索引，我们可以轻松地通过数组切片提取眼睛区域
# 分别抓取左眼和右眼的面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

#启动视频流线程
print("[INFO] starting video stream thread...")
#使用文件的视频流
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
#内置网络摄像头或USB摄像头
vs = VideoStream(src=0).start()
#使用Pi相机
# vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

# 视频流中循环帧
while True:
    #读取下一 帧
    frame = vs.read()
    #调整宽度为500像素并将其转换为灰度进行预处理
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #harr检测灰度图像中的面部 
    rects = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE)
    #没有检测到人脸则继续检测下一帧
    if len(rects)<=0:
        continue
    for (x, y, w, h) in rects:                
        #显示面部边框(坐标int类型)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    
#     # 应用dlib的面部检测器来查找和定位灰度图像中的面部
#     rects = detector(gray, 0)
#     #循环面部检测,提取rects检测的坐标和宽度+高度
#     for rect in rects:
#         # 将dlib的矩形转换为OpenCV样式的边界框[即（x，y，w，h）]，然后绘制面边框
#         (x, y, w, h) = face_utils.rect_to_bb(rect)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         
#         # 确定面部区域的面部标志并将面部标志（x，y） -坐标转换为NumPy阵列。
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
#         
#         #取出左右眼对应的特征点
#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         mouth = shape[mStart: mEnd]   
#         #绘制左眼边框
#         stratX=int(leftEye[0][0]-abs(leftEye[0][0]-leftEye[3][0])*0.3)
#         stratY=int(min(leftEye[0][1],leftEye[3][1])-abs(leftEye[1][1]-leftEye[5][1])*1.5)
#         endX=int(leftEye[3][0]+abs(leftEye[0][0]-leftEye[3][0])*0.3)
#         endY=int(min(leftEye[0][1],leftEye[3][1])+abs(leftEye[1][1]-leftEye[5][1])*1.5)
#         cv2.rectangle(frame, (stratX,stratY), (endX, endY), (255, 0, 0), 1)
#         print"(x1,y1):({},{}),(x2,y2):({},{})".format(stratX,stratY, endX,endY)
#         #绘制右眼边框
#         stratX=int(rightEye[0][0]-abs(rightEye[0][0]-rightEye[3][0])*0.3)
#         stratY=int(min(rightEye[0][1],rightEye[3][1])-abs(rightEye[1][1]-rightEye[5][1])*1.5)
#         endX=int(rightEye[3][0]+abs(rightEye[0][0]-rightEye[3][0])*0.3)
#         endY=int(min(rightEye[0][1],rightEye[3][1])+abs(rightEye[1][1]-rightEye[5][1])*1.5)
#         cv2.rectangle(frame, (stratX,stratY), (endX, endY), (255, 0, 0), 1)
         
#         #计算左眼和右眼的凸包，绘制左右眼轮廓
#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         #计算嘴部凸包，绘制嘴部轮廓
#         mouthHull=cv2.convexHull(mouth);
#                  
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        # 循环（x，y） - 面部地标的坐标并在图像上绘制它们
#         for (x, y) in shape:
#             cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    # 显示图像
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # 按q退出循环
    if key == ord("q"):
        break
 
# 关闭窗口
cv2.destroyAllWindows()
vs.stop()