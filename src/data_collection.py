#coding=utf-8 
#usage 按b开始或暂停数据采集
#VECTOR_SIZE表示你的特征向量维度多少。注意采集数据程序中的VECTOR_SIZE要和其他程序中一致

import numpy as np 
import os
import dlib
import cv2
from scipy.spatial import distance
from imutils import face_utils
import pickle

#队列特征向量
#VECTOR_SIZE表示你的特征向量维度多少，默认取3维的。注意采集数据程序中的VECTOR_SIZE要和其他程序中一致
VECTOR_SIZE =3


def queue_in(queue, data):
    ret = None
    #采集的特征向量为VECTOR_SIZE时取队列第一个元素
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue

#计算眼睛横纵比
def eye_aspect_ratio(eye):
    # print(eye)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

#pwd = os.getcwd()
#model_path = os.path.join(pwd, 'model')
#shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')
shape_detector_path='model/shape_predictor_68_face_landmarks.dat'

## 初始化dlib的人脸检测器（基于HOG），然后创建面部标志预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)

cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(0)

#采集眼睛睁开/闭合时的样本
print('Prepare to collect images with your eyes open')
print('Press b to begin collecting images.')
#print('Press s to stop collecting images.')
print('Press q to quit')
#判断是否要开始/暂停/结束采集数据
flag = 0

# 'r'：读
# 'w'：写
# 'a'：追加
# 'r+' == r+w（可读可写，文件若不存在就报错(IOError)）
# 'w+' == w+r（可读可写，文件若不存在就创建）
# 'a+' ==a+r（可追加可写，文件若不存在就创建）
# 对应的，如果是二进制文件，就都加一个b就好啦：
# 'rb'　　'wb'　　'ab'　　'rb+'　　'wb+'　　'ab+'

#未眨眼训练数据
#txt = open('train_open.txt', 'wb')
#txt = open('train_close.txt', 'wb')
#可追加可写，末尾追加再写，文件若不存在就创建
#txt = open('train_open.txt', 'ab+')
#眨眼训练数据
#txt = open('train_close.txt', 'ab+')

#测试数据
txt = open('test_open.txt', 'ab+')
#txt = open('test_close.txt', 'ab+')

#txt = open('close.txt', 'ab+')
data_counter = 0
ear_vector = []

# 分别抓取左眼和右眼的面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while(1):
    ret, frame = cap.read()
    key = cv2.waitKey(1)
    if key & 0xFF == ord("b"):
        print('begin collecting images.')
        flag = not flag
#     elif key & 0xFF == ord("s"):
#         print('Stop collecting images.')
#         flag = 0
    elif key & 0xFF == ord("q"):
        print('quit')
        break

    if flag == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #应用dlib的面部检测器来查找和定位灰度图像中的面部
        rects = detector(gray, 0)
        # 遍历每个检测到的面部
        for rect in rects:
            # 将dlib的矩形转换为OpenCV样式的边界框[即（x，y，w，h）]，然后绘制面边框
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #每个检测到的面部，我们应用dlib的面部标志检测器
            shape = predictor(gray, rect)
            #将结果转换为NumPy阵列
            shape = face_utils.shape_to_np(shape)
            
            # 循环（x，y） - 面部地标的坐标并在图像上绘制它们
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                
            #提取左眼和右眼对应特制点坐标，然后使用坐标计算双眼的眼睛纵横比
            #使用NumPy数组切片，我们可以分别提取左眼和右眼的  （x，y）坐标
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            #print('leftEAR = {0}'.format(leftEAR))
            #print('rightEAR = {0}'.format(rightEAR))
            #求双眼纵横比的均值
            ear = (leftEAR + rightEAR) / 2.0
            #计算左眼和右眼的凸包，然后可视化每只眼睛
            ## 寻找左眼轮廓
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # 绘制左右眼轮廓
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            #特征向量放入队列
            ret, ear_vector = queue_in(ear_vector, ear)
            #采集的特征向量为VECTOR_SIZE时
            if(len(ear_vector) == VECTOR_SIZE):
                # print(ear_vector)
                # input_vector = []
                # input_vector.append(ear_vector)

                txt.write(str(ear_vector))
                txt.write('\n')
                txt.flush()
                #计算取特征向量的组数
                data_counter += 1
                print(data_counter)

            cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("frame", frame)
txt.close()
cap.release()
cv2.destroyAllWindows()