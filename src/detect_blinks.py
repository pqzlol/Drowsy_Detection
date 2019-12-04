#coding=utf-8

#计算眼部纵横比及头部位置

#导入需要的包
from scipy.spatial import distance as dist #算法库和数学工具包
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np	#数学库，主要用于数组计算
import argparse #命令行解析包
import imutils #图像处理包
import time	#时间包
import dlib #特征点标记库
import cv2	#opencv库
import matplotlib #绘图库
def eye_aspect_ratio(eye):
	#计算两组垂直眼睛标记点（x，y） - 坐标之间的欧氏距离
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	#计算水平眼睛界标（x，y） - 坐标之间的欧氏距离
	C = dist.euclidean(eye[0], eye[3])

	#计算垂直眼睛界标（x，y） - 坐标之间的欧氏距离
	ear = (A + B) / (2.0 * C)

	# 返回眼睛纵横比
	return ear

# # 构造参数解析并解析参数
# ap = argparse.ArgumentParser()
# #定义了可选参数-p或--shape-predictor，通过解析后，其值保存在args.shape-predictor变量中
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-v", "--video", type=str, default="",
# 	help="path to input video file")
# #解析所有变量
# args = vars(ap.parse_args())

#define两个常数，一个用于眼睛纵横比以指示眨眼，然后第二个常量用于眼睛必须低于阈值的连续帧数
# EAR阈值  EAR大于它，则认为眼睛是睁开的；如果EAR小于它，则认为眼睛是闭上的。（这个阈值时可根据自己的调整）
EYE_AR_THRESH = 0.2
# 当EAR小于阈值时，接连多少帧一定发生眨眼动作
EYE_AR_CONSEC_FRAMES = 3
#储存ear比值
ear_vector = []
left_ear = []
right_ear = []

#存储头部位置 以左右眼的眼角中点来计算位置
head_location=[]

# 初始化闭眼帧数总数和眨眼总数
#frame_counter是眼睛纵横比小于EYE_AR_THRESH的连续帧的总数
frame_counter = 0
#眨眼总数
blink_counter = 0

#用下面的代码可以减少参数的输入
# pwd = os.getcwd()# 获取当前路径
# model_path = os.path.join(pwd, 'model')# 模型文件夹路径
# # 人脸特征点检测模型路径
# shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')

#初始化dlib的人脸检测器（基于HOG），然后创建面部标志预测器
print("[INFO] loading facial landmark predictor...")
# 人脸检测器
detector = dlib.get_frontal_face_detector()
#人脸特征点检测器
#predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

#分别获取左眼和右眼的面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
print("lStart:",lStart)

# 启动视频流线程
print("[INFO] starting video stream thread...")
#使用文件的视频流
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
#内置网络摄像头或USB摄像头
vs = VideoStream(src=0).start()
#使用Pi相机
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)
flag=True

# 循环视频流的每帧
while flag:
	#如果这是一个文件视频流，那么我们需要检查缓冲区中是否还有剩余的帧需要处理
	if fileStream and not vs.more():
		break

	# 视频流中读取下一帧，然后调整大小并将其转换为灰度
	frame = vs.read()
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

		#提取左眼和右眼坐标，然后使用坐标计算双眼的眼睛纵横比
		#取出左眼对应的特征点
		leftEye = shape[lStart:lEnd]
		# 取出右眼对应的特征点
		rightEye = shape[rStart:rEnd]
		# 计算左眼EAR
		leftEAR = eye_aspect_ratio(leftEye)
		# 计算右眼EAR
		rightEAR = eye_aspect_ratio(rightEye)
# 		print('leftEAR= {0}'.format(leftEAR))
# 		print('rightEAR= {0}'.format(rightEAR))
		#求双眼纵横比的均值
		ear = (leftEAR + rightEAR) / 2.0
		
		#计算头部垂直位置y
		heady_y=(leftEye[0][1]+rightEye[3][1])/2.0
		
		if len(ear_vector)>=80:
			print ear_vector
			print left_ear
			print right_ear
			print head_location
			#退出循环
			flag=False
		else:
			left_ear.append(leftEAR)
			right_ear.append(rightEAR)
			ear_vector.append(ear)
			head_location.append(heady_y)

		#计算左眼和右眼的凸包，然后可视化每只眼睛
		## 寻找左眼轮廓
		leftEyeHull = cv2.convexHull(leftEye)
		# 寻找右眼轮廓
		rightEyeHull = cv2.convexHull(rightEye)
		# 绘制左右眼轮廓
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

		# 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
		if ear < EYE_AR_THRESH:
			frame_counter += 1

		# 如果EAR大于阈值
		else:
			# EAR小于阈值的总帧数大于等于EYE_AR_CONSEC_FRAMES记一次眨眼
			if frame_counter >= EYE_AR_CONSEC_FRAMES:
				blink_counter += 1

			# 重置EAR小于阈值的总帧数
			frame_counter = 0

		# 在图像上显示出眨眼次数blink_counter和EAR
		cv2.putText(frame, "Blinks: {}".format(blink_counter), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "heady_y: {:.2f}".format(heady_y), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# 按“q“退出循环
	if key == ord("q"):
		break

# 关闭所有窗口和视频流
cv2.destroyAllWindows()
vs.stop()