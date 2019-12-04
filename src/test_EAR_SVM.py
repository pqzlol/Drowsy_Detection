#coding=utf-8 
# import numpy as np 
# from sklearn import svm
# from sklearn.externals import joblib


import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils.video import VideoStream   #视频流
from imutils import face_utils
import imutils #图像处理包
#from sklearn import svm
from sklearn.externals import joblib
  
VECTOR_SIZE = 13
def queue_in(queue, data):
    ret = None
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue
  
def eye_aspect_ratio(eye):
    # print(eye)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
  
shape_detector_path='model/shape_predictor_68_face_landmarks.dat'
  
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)
  
# 导入模型
clf = joblib.load("ear_svm.m")
 
#连续闭眼帧数
close_counter = 0
#眨眼帧数（连续3帧闭眼算一次眨眼）
blink_counter = 0
#眼睛横纵比向量数组
ear_vector = []
#EYE_AR_THRESH = 0.3# EAR阈值
EYE_AR_CONSEC_FRAMES = 3# 接连多少帧一定发生眨眼动作
  
# 对应特征点的序号
# 分别抓取左眼和右眼的面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
  
#cap = cv2.VideoCapture(1)
#开启视频流
cap = VideoStream(src=0).start()
 
 
while(1):
    #读取每一帧
    img = cap.read()
    img=imutils.resize(img, width=500)
     
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 应用dlib的面部检测器来查找和定位灰度图像中的面部
    rects = detector(gray, 0)
    #遍历检测到的人脸
    for rect in rects:
        print('-'*20)
        ##检测特征点
        shape = predictor(gray, rect)
        #面部特征点（x，y）坐标转换为NumPy阵列
        points = face_utils.shape_to_np(shape)
         
        #提取左眼和右眼坐标，然后使用坐标计算双眼的眼睛纵横比
        #取出左眼对应的特征点
        leftEye = points[lStart:lEnd]
        rightEye = points[rStart:rEnd]
        #计算横纵比
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        print('leftEAR = {0}'.format(leftEAR))
        print('rightEAR = {0}'.format(rightEAR))
  
        ear = (leftEAR + rightEAR) / 2.0
         
        #计算左眼和右眼的凸包，绘制左右眼轮廓
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
         
        #横纵比向量
        ret, ear_vector = queue_in(ear_vector, ear)
        #当特征向量为VECTOR_SIZE
        if(len(ear_vector) == VECTOR_SIZE):
            print(ear_vector)
            input_vector = []
            input_vector.append(ear_vector)
            res = clf.predict(input_vector)
             
            print(res)
  
            if res == "close":
                close_counter += 1
            else:
                if close_counter >= EYE_AR_CONSEC_FRAMES:
                    blink_counter += 1
                close_counter = 0
  
        cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
  
     
    cv2.imshow("Frame", img)
  
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
print("Blinks:{0}".format(blink_counter)) 
#cap.release()
cap.stop()
cv2.destroyAllWindows()
