#coding=utf-8

#导入需要的包
from scipy.spatial import distance as dist #算法库和数学工具包
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils  # 图像处理包
import numpy as np	#数学库，主要用于数组计算
import os
import dlib #特征点标记库
import cv2	#opencv库
import matplotlib #绘图库


def mouth_aspect_ratio(mouth):
	A = dist.euclidean(mouth[2],mouth[10])
	B = dist.euclidean(mouth[3],mouth[9])
	C = dist.euclidean(mouth[4],mouth[8])
	mar1 = (A+B+C)/3.0
	D= dist.euclidean(mouth[0],mouth[6])
	E=dist.euclidean(mouth[12],mouth[16])
	mar2=(D+E)/2.0
	mar=mar1 /mar2
	return mar

#define两个常数，一个用于纵横比以指示打哈欠，然后第二个常量用于比值高于阈值的连续帧数
# MAR阈值  MAR大于它，
THRESH = 0.7
# 当EAR大于阈值时，接连多少帧记一次打哈欠
CONSEC_FRAMES = 15

# 初始化打哈欠帧数总数和打哈欠总数
#frame_counter是纵横比大于THRESH的连续帧的总数
frame_counter = 0


#打哈欠总数
sum_yaw_counter=0

#初始化dlib的人脸检测器（基于HOG），然后创建面部标志预测器
print("[INFO] loading facial landmark predictor...")
# 人脸检测器
detector = dlib.get_frontal_face_detector()
#人脸特征点检测器
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

#获取嘴部的面部标志的索引
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


#遍历数据集文件
root='YawDD dataset/Dash'
for i in os.listdir(root):
	#获取文件路径
	path=os.path.join(root,i)
	if os.path.isfile(path):
		print(path)
		
		#计算每个视频哈欠次数
		yaw_counter = 0
		# 捕捉视频，未开始读取
		cap = cv2.VideoCapture(path) 
		#读第一帧
		success, frame = cap.read()  
		# 循环直到视频结束
		while success:  
	
			frame = imutils.resize(frame, width=450)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
			# 通过dlib的内置面部检测器检测灰度帧中的人脸
			rects = detector(gray, 0)
		
			# 遍历检测到的每个人脸，然后对每个人脸检测面部特征点
			# 遍历每一个人脸
			for rect in rects:
				#确定面部区域的面部特征点，然后将面部特征点（x，y）坐标转换为NumPy阵列
				#检测特征点
				shape = predictor(gray, rect)
				#面部特征点（x，y）坐标转换为NumPy阵列
				shape = face_utils.shape_to_np(shape)
		
				#提取嘴部坐标，然后使用坐标计算纵横比
				mouth = shape[mStart:mEnd]
				# 计算纵横比
				MAR = mouth_aspect_ratio(mouth)
				#print('MAR= {0}'.format(MAR))
		
				## 寻找嘴部轮廓
				MouyhHull = cv2.convexHull(mouth)
				# 绘制嘴部轮廓
				cv2.drawContours(frame, [MouyhHull], -1, (0, 0, 255), 1)
		
				# 如果MAR大于阈值，开始计算连续帧，只有连续帧计数超过CONSEC_FRAMES时，才会计一次打哈欠
				if MAR > THRESH:
					frame_counter += 1
		
				# 如果小于阈值
				else:
					# MAR大于阈值的总帧数大于等于CONSEC_FRAMES记一次打哈欠
					if frame_counter >= CONSEC_FRAMES:
						yaw_counter += 1
					# 重置MAR大于阈值的总帧数
					frame_counter = 0
		
				# 在图像上显示出眨打哈欠次数和MAR
				cv2.putText(frame, "yawn: {}".format(yaw_counter), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "MAR: {:.2f}".format(MAR), (300, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
			# show the frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			
			# 按“q“退出循环
			if key == ord("q"):
				break
			
			# 获取下一帧
			success, frame = cap.read() 
		
		#关闭所有窗口和视频流
		cap.release()
		cv2.destroyAllWindows()

	print("yaw:{0}".format(yaw_counter))
	print('-'*20)
	#总打哈欠数
	sum_yaw_counter+=yaw_counter
	
print('-'*20)
print("sum_yaw_counter:{0}".format(sum_yaw_counter)) 
#print("accuray:{0}".format(all_blink_counter/255.0)) 