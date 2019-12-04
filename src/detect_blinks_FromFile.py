# coding=utf-8 
#测试ZJU眨眼数据库(80个视频)
#3 3 4 3 4 6 4 5 5 3 3 10 2

import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
import os
#from numba import jit #加速
from imutils.video import VideoStream  # 视频流
from imutils import face_utils
import imutils  # 图像处理包
# from sklearn import svm
from sklearn.externals import joblib

# #13维向量 
# VECTOR_SIZE = 13

##dlib预训练的面部标志检测器的路径
shape_detector_path = 'model/shape_predictor_68_face_landmarks.dat'

#视频总帧数
all_frame_counter=0
#每个视频总帧数
frame_counter=0
# 每个视频眨眼帧数（连续3帧闭眼算一次眨眼）
blink_counter = 0
#眨眼总帧数
all_blink_counter=0
#t统计小于阈值数
ear_frame=0
EYE_AR_THRESH = 0.24# EAR阈值
EYE_AR_CONSEC_FRAMES =3 # 接连多少帧一定发生眨眼动作 连续3帧闭眼算一次眨眼

def eye_aspect_ratio(eye):
    # print(eye)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


#人脸检测器和面部标志检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)
 
# 对应特征点的序号
# 分别抓取左眼和右眼的面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


#遍历数据集文件
root='zjublinkdb'
for i in os.listdir(root):
    #获取文件路径
    path=os.path.join(root,i)
    if os.path.isfile(path):
        print(path)
        
        # 捕捉视频，未开始读取
        cap = cv2.VideoCapture(path) 
        #获取视频帧速率 
        #fps = cap.get(cv2.CAP_PROP_FPS) 
        
        #计算每个视频中眨眼次数（连续3帧闭眼算一次眨眼）
        blink_counter=0 
     
        #视频帧数
        frame_counter=0
        #读第一帧
        success, img = cap.read()  
        # 循环直到视频结束
        while success:  
            #计算每个视频帧数
            frame_counter+=1
            
            #调整大小
            img=imutils.resize(img, width=500)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 应用dlib的面部检测器来查找和定位灰度图像中的面部
            rects = detector(gray, 0)
            
            #遍历检测到的人脸
            for rect in rects:
                
                # 将dlib的矩形转换为OpenCV样式的边界框[即（x，y，w，h）]，然后绘制面边框
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                #print('-'*20)
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
                #print('leftEAR = {0}'.format(leftEAR))
                #print('rightEAR = {0}'.format(rightEAR))
          
                ear = (leftEAR + rightEAR) / 2.0
                 
                #计算左眼和右眼的凸包，绘制左右眼轮廓
                leftEyeHull = cv2.convexHull(leftEye)#返回的是凸包点在原轮廓点集中的索引
                rightEyeHull = cv2.convexHull(rightEye)
               
                cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
                # 循环（x，y） - 面部地标的坐标并在图像上绘制它们
#                 for (x, y) in points:
#                     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
                
                #如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
                if ear < EYE_AR_THRESH:
                	ear_frame += 1
                	#如果EAR大于阈值
                else:
					# EAR小于阈值的总帧数大于等于EYE_AR_CONSEC_FRAMES记一次眨眼
					if ear_frame >= EYE_AR_CONSEC_FRAMES:
						blink_counter += 1
		
					# 重置EAR小于阈值的总帧数
					ear_frame = 0
               
                #标记眨眼次数和眼睛横纵比
                cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                #cv2.putText(img, "EyeClose:{0}".format(each_close_counter), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                #cv2.putText(img, "frame_counter:{}".format(frame_counter), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
             
            cv2.imshow("Frame", img)
            #按q退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  
            # 获取下一帧
            success, img = cap.read() 
            
        cap.release()
        #cap.stop()
        cv2.destroyAllWindows()
        
        print("Blinks:{0}".format(blink_counter))
        print("frame_counter:{}".format(frame_counter))
        print('-'*20)
        
        #总眨眼次数
        all_blink_counter += blink_counter
        #总帧数
        all_frame_counter+=frame_counter
       
      
print('-'*20)
print("all_Blink_Counter:{0}".format(all_blink_counter)) 
print("sum_frame:{0}".format(all_frame_counter)) 
print("accuray:{0}".format(all_blink_counter/255.0)) 
        