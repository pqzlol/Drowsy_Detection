#coding=utf-8

#######从文件中读取视频 检测疲劳#########

#导入需要的包
from scipy.spatial import distance as dist #算法库和数学工具包
#from imutils.video import FileVideoStream   #文件视频流
from imutils.video import VideoStream   #视频流
from imutils import face_utils
from numba import jit #加速
import numpy as np    #数学库，主要用于数组计算

import imutils #图像处理包
import time    #时间包
import dlib #特征点标记库
import cv2    #opencv库
from Driver import *
from sklearn.externals import joblib
import playsound
from threading import Thread
import os

################################# 定义变量 ############################

###1、设置文件路径
#opencv自带的训练好的人脸检测xml分类器文件路径
face_cascade_path='model/haarcascade_frontalface_default.xml'
#dlib预训练的面部标志检测器的路径
facial_landmark_predictor_path='model/shape_predictor_68_face_landmarks.dat'

#警报的输入音频文件的路径
#slightFA_path='alarm/SlightFA.wav'
moderateFA_path='alarm/ModerateFA.wav'
severeFA_path='alarm/SevereFA.wav'

###2、定义系统变量
#初始化跟踪器
tracker = None
#暂停视频流
paused = True
    
# 眼睛横纵比向量数组
ear_vector = []
#向量维数
VECTOR_SIZE=3

#define两个常数，一个用于眼睛纵横比以指示眨眼，然后第二个常量用于眼睛必须低于阈值的连续帧数
# EAR阈值  EAR大于它，则认为眼睛是睁开的；如果EAR小于它，则认为眼睛是闭上的。（这个阈值时可根据自己的调整）
eyeThreshold = 0.24
# 当EAR小于阈值时，接连多少帧一定发生眨眼动作
EYE_CONSEC_FRAMES = 3

#嘴部
mouthThreshold=0.7
Mouth_CONSEC_FRAMES =15

#存储头部位置 以左右眼的眼角中点来计算位置
headLocation=[]

################################# end ############################

################################## 定义函数 #############################################
########帮助函数##############
@jit
def help():
    print("\nThis is a fatigue driving detection and warning system.\n"
            "Make sure your face in video camera.\n"
            "Every minute after the system , it will automatically check for fatigue.\n"
            "According to the result, the system will give the corresponding voice alert.")
    print("Hot keys: \n"
            "\tq - quit the system\n"
            "\ts - stop/begin the system\n"
            "\tb - begin the system\n"
            "Make sure your face in video camera then press 'b' to begin the program")
######## end ##############

########语音播报##############
def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)
    
######## end ##############

###########################
def queue_in(queue, data):
    ret = None
    #如果该
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue

#########################

###########图像锐化拉普拉斯算子###########################
#使用中心为5的8邻域拉普拉斯算子与图像卷积可以达到锐化增强图像的目的
@jit
def Laplace(image):
    #定义卷积核
    kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    dst = cv2.filter2D(image, -1,kernel)
    return dst
######## end ##############

###########光照补偿法（改进的参考白算法）###################
@jit
def Reference_white(image):
    #光照补偿后结果图
    result=image
    #图像参考白前5%像素点的个数
    image_rows=image.shape[0] # 图像的行数(高)
    image_cols=image.shape[1] # 图像的列数（宽）
    Reference_white_num = image_rows * image_cols * 0.05;
    
    #创建一个单通道图像(二维数组)，存储每个像素点的亮度值
    grayImg=np.zeros((image_rows,image_cols), dtype=int) 
   
    #创建一个一维数组，统计每个亮度值的个数
    histogram=np.zeros(256)
    #灰度直方图统计,计算每个像素点的亮度值
    for i in range(0, image_rows):
        for j in range(0, image_cols):
            #计算像素点灰度值
            #opencv默认的imread是以BGR的方式进行存储的
            gray=int((image[i,j,0]*114+image[i,j,1]* 587+image[i,j,2]*299)/1000)
            grayImg[i,j]=gray
            #灰度值（每个亮度值）统计
            histogram[gray]+=1
    
    #初始化前5%像素点的个数及亮度
    light=0
    sum=0
    for light in range(255,-1,-1):#从255-0查找
        #按亮度值按由大到小次序，找出亮度值处于前5%的像素点总个数
        sum+=histogram[light]
        
        #亮度值处于前5%的像素点亮度较强或较弱时进行光照补偿
        if sum >Reference_white_num:
            #光线较强或者较弱，则进行光照补偿
            if light >200 or light<60:
                break
            #否则不进行光照补偿
            else:
                return
    #输出亮度值
    #print("light: {}".format(light))
    
    #定义变量计算计算R、G、B三个分量像素总值
    sum_R=0.0
    sum_G=0.0
    sum_B=0.0
    
    #亮度值处于前5%的像素点，将其RGB分量全部设为255，这些像素点即为“参考白”
    for i in range(0, image_rows):
        for j in range(0, image_cols):
            #亮度值处于前5%的像素点，将其RGB分量全部设为255
            if grayImg[i,j] >=light:
                #将其RGB分量全部设为255
                result[i,j,0]=255
                result[i,j,1]=255
                result[i,j,2]=255
                #计算R、G、B三个分量的像素总值
                sum_R+=255
                sum_G+=255
                sum_B+=255
            else:
                sum_R+=image[i,j,2]
                sum_G+=image[i,j,1]
                sum_B+=image[i,j,0]
    
    #计算R、G、B三个分量在像素中所占比例最大值      
    factor_R = sum_R / (Reference_white_num * 255)
    factor_G = sum_G / (Reference_white_num * 255)
    factor_B = sum_B / (Reference_white_num * 255)
    
    #max_factor为R、G、B三个分量在像素中所占比例最大值
    max_factor = factor_R
    if max_factor < factor_G:
        max_factor = factor_G
    
    if max_factor < factor_B:
        max_factor = factor_B
    
    factor_R = max_factor/factor_R;
    factor_G = max_factor/factor_G;
    factor_B = max_factor/factor_B;
    
    #参考白算法
    for i in range(0, image_rows):
        for j in range(0, image_cols):
            if result[i,j,0]<255:
                result[i,j,2] = image[i,j,2] * factor_R
                result[i,j,1] = image[i,j,1] * factor_G
                result[i,j,0] = image[i,j,0] * factor_B
           
    #返回光照补偿后的图像
    return result
######## end ##############

#####计算垂直眼睛地标与水平眼睛地标之间距离的比率###################
#当眼睛打开时，眼睛纵横比的返回值将近似恒定。该值将在眨眼期间快速减小到零。
#如果眼睛闭合，眼睛纵横比将再次保持近似恒定，但将远小于眼睛打开时的比率。
@jit
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
######## end ##############

###########计算嘴部纵横比################
@jit
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
######## end ##############

#############人脸检测及预处理##################
@jit
def detect_face(frame):
    #####图像预处理###############
    start_time=time.time()
    #1、中值滤波去噪
    blurImg=cv2.medianBlur(frame,5)
    #cv2.imshow('medianBlur',blurImg)
    #2、图像锐化（aplacian算子）
    LaplaceImg=Laplace(blurImg)
    #cv2.imshow("Laplacian", LaplaceImg)
    #3、光照补偿（参考白算法）
    img=Reference_white(LaplaceImg)
    #cv2.imshow("Reference_white", img)
    #4、转换为灰度图像进行预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #预处理使用时间
    #Pretreatment_time=time.time()-start_time
    print "图像预处理时间: {}".format(Pretreatment_time)
    ##### end ###############              
                 
    ##### 人脸检测 ###############################
    start_time=time.time()
    #检测灰度图像中的面部 
    rects = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE)
    #print "人脸检测时间: {}".format(time.time()-start_time)
    ##### end ###############################
    
    return gray,rects;
######## end ##############

############疲劳判断##################

def judgeFatigueState(dA):
    
    print "闭眼持续最长时间：{}".format(dA.maxECT)
    print "闭眼帧数：{}".format(dA.closeFrame)
    #计算闭眼帧数占比
    dA.ECR=(dA.closeFrame*1.0)/dA.totalFrame
    print "闭眼眼睛闭合比：{}".format(dA.ECR)
    print "眨眼频率：{}".format(dA.BF)
    print "点头频率：{}".format(dA.NF)
    print "打哈欠频率：{}".format(dA.YF)
    print "最长张嘴时间:{}".format(dA.maxYT)
    #最长闭眼时间大于1.5s
    if dA.maxECT>=1.5:
        dA.fatigueRation+=0.2
    
    #闭眼比大于0.2
    if dA.ECR>=0.2:
        dA.fatigueRation+=0.1
    
    #眨眼频率>25或<10
    if dA.BF<10 or dA.BF>25:
        dA.fatigueRation+=0.2
    
    #点头频率
    if dA.NF>=8 :
        dA.fatigueRation+=0.3
    
    #打哈欠频率3次及最长张嘴时间4s
    if dA.YF >=3 or dA.maxYT>=4:
        dA.fatigueRation+=0.2
    
    print "疲劳值：{}".format(dA.fatigueRation)
    
    if dA.fatigueRation<0.3:
        print "正常驾驶"
    #疲劳
    if dA.fatigueRation>=0.3 and dA.fatigueRation<0.7 :
        print "您已处于疲劳状态，请尽快停车休息！"
        #创建一个线程播放语音
        t = Thread(target=sound_alarm,args=(moderateFA_path,))
        t.deamon = True
        t.start()
        #sound_alarm(moderateFA_path)
    #重度疲劳
    if dA.fatigueRation>=0.7:
        print "您已严重疲劳，请立即停车休息"
        #sound_alarm(severeFA_path)
        t = Thread(target=sound_alarm,args=(severeFA_path,))
        t.deamon = True
        t.start()
        
############end##################

##############头部运动分析##################
@jit
def getHeadMovement(headLocation):
    #极值点个数
    extremumNum = 0;
    #点头次数
    NodFreq = 0;
    for i in range(len(headLocation)-2):
        #计算极值点
        if (headLocation[i]-headLocation[i+1])*(headLocation[i+1]-headLocation[i+2])<0: 
            extremumNum+=1
            if headLocation[i+1]-headLocation[0]>=50:
                NodFreq+=1
        #如果没有极值点。则判断是否为单调函数
        if extremumNum == 0:
            if headLocation[0] - headLocation[1] < 0 :
                NodFreq = 1;
    return NodFreq
#####################################

############核心代码###############################
if __name__ == '__main__':
    
    #帮助提示
    #help()
    
    #加载我们的Haar级联分类器和面部地标预测器文件
    print "[INFO] loading OpenCV's Haar cascade facial landmark predictor..."
    detector = cv2.CascadeClassifier(face_cascade_path)
    predictor = dlib.shape_predictor(facial_landmark_predictor_path)
    #导入训练的人眼睁闭状态识别模型
    clf = joblib.load("ear_svm.m")
    #从一组面部标志中提取眼睛区域，，使用这些阵列切片索引，我们可以轻松地通过数组切片提取眼睛区域
    # 分别抓取左眼和右眼的面部标志的索引
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    
    
    ######### 启动视频流线程 ######################
   
    #遍历数据集文件
    root='NTHU/glasses'
    for i in os.listdir(root):
        #获取文件路径
        path=os.path.join(root,i)
        if os.path.isfile(path):
            print "***********************"
            print(path)
            #创建类对象
            dA=Driver()
            #统计周期总时间
            totalTime=0.0
            #人眼识别时间
            eye_time=0.0
            #嘴部识别时间
            mouth_time=0.0
            #头部识别时间
            head_time=0.0
            #特征点定位时间
            FP_time=0.0
            #错检帧数
            errorFrameNum=0
            #存储眼部状态（0为闭眼，1为睁眼）
            eyeState=[]
            
            #计算眼睛连续小于阈值的帧数
            eyeFrameNum=0
            #计算嘴部连续小于阈值的帧数
            mouthFrameNum =0  
            #连续闭眼帧数
            eyeCloseNum=0
            
            # 捕捉视频，未开始读取
            cap = cv2.VideoCapture(path) 
            #获取视频帧率
            fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
            print "fps:",fps
            success=True
            
            # 视频流中循环帧
            while success:
                #读第一帧
                success, frame = cap.read() 
                #总帧数+1
                dA.totalFrame+=1 
                
                #开始时间 计算时间周期 每个时间周期计算一次疲劳状况
                runTime=time.time()
                #print "totalTime:",totalTime
                #如果时间周期结束，进行疲劳判断
                if(totalTime+(errorFrameNum+dA.totalFrame) * 0.004>=60):
                    #总帧数
                    print "totalFrameNum:{}".format(dA.totalFrame)
                    print "erroeFrameNum：{}".format(errorFrameNum)
                    print "有效帧数：{}".format(dA.totalFrame-errorFrameNum)
                    print "特征点定位时间：{}".format(FP_time)
                    #头部运动分析
                    head_time=time.time()
                    dA.NF=getHeadMovement(headLocation)
                    print "头部状态识别时间：{}".format(time.time()-head_time)
                    print "眼部识别时间：{}".format(eye_time)
                    print "嘴部识别时间:{}".format(mouth_time)
                    
                    #疲劳判断及预警
                    fatigue_time=time.time()
                    judgeFatigueState(dA)
                    print "疲劳识别时间：{}".format(time.time()-fatigue_time)
                    
                    #重置特征参数
                    dA.init()
                    totalTime=0.0                
                    errorFrameNum=0
                    #计算眼睛连续小于阈值的帧数
                    eyeFrameNum=0
                    #计算嘴部连续小于阈值的帧数
                    mouthFrameNum =0  
                    #连续闭眼帧数
                    eyeCloseNum=0
                    headLocation=[]
                    
                
                #若视频结束则退出循环
                if frame is None:
                    print "totalTime:",totalTime
                    print("frame is empty")
                    break
                
                    
                #调整宽度为500像素
                frame = imutils.resize(frame, width=500)
                
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #跟踪器对象为 None，检测人脸再跟踪对象
                if tracker is None:
                    ######人脸检测######################################
                    #gray,rects=detect_face(frame)
                    #####图像预处理###############
                    start_time=time.time()
                    #1、中值滤波去噪
                    blurImg=cv2.medianBlur(frame,5)
                    #cv2.imshow('medianBlur',blurImg)
                    #2、图像锐化（aplacian算子）
                    LaplaceImg=Laplace(blurImg)
                    #cv2.imshow("Laplacian", LaplaceImg)
                    #3、光照补偿（参考白算法）
                    img=Reference_white(LaplaceImg)
                    #cv2.imshow("Reference_white", img)
                    #4、转换为灰度图像进行预处理
                    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    #预处理使用时间
                    Pretreatment_time=time.time()-start_time
                    #print "图像预处理时间: {}".format(Pretreatment_time)
                    ##### end ###############              
                                     
                    ##### 人脸检测 ###############################
                    start_time=time.time()
                    #检测灰度图像中的面部 
                    rects = detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE)
                    #print "人脸检测时间: {}".format(time.time()-start_time)
                    ##############end#################################
                    
                    #没有检测到人脸则继续检测下一帧
                    if len(rects)<=0:
                        tracker=None
                        errorFrameNum+=1
                        totalTime+=time.time()-runTime
                        continue
                        
                    #循环面部检测,提取rects检测的坐标和宽度+高度，OpenCV样式的边界框[即（x，y，w，h）
                    for (x, y, w, h) in rects:
                        t1=time.time()
                        #建立我们的dlib对象跟踪器并提供边界框坐标
                        #创建一个跟踪类
                        tracker = dlib.correlation_tracker()
                        #从Haar级联边界框构造一个dlib矩形对象(坐标int类型)
                        rect = dlib.rectangle(x, y, x + w, y + h,)
                        #开启跟踪器,设置图片中的要跟踪物体的框
                        tracker.start_track(frame, rect)
                        #print "人脸跟踪时间：{}".format(time.time()-t1)
                            
                        #显示面部边框(坐标int类型)
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        #print "检测到的面部位置：x:{}，y:{},x+w:{},y+h:{},w:{},h:{}".format(x,y,x+w,y+h,w,h)
                #已经锁定到对象进行跟踪
                else:
                    #更新跟踪器,实时跟踪下一帧对象,返回峰值与旁瓣比率，值越大表示置信度越高
                    t1=time.time()
                    is_found=tracker.update(frame)
                    
                    #print "跟踪时间：{}".format(time.time()-t1)
                    #print "置信度:{}".format(is_found)
                        
                    #得到跟踪到的目标的位置   
                    pos = tracker.get_position()
                    #print "跟踪位置: {}".format(pos)
                        
                    # 解压位置对象
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    #若丢失跟踪目标 重新检测人脸
                    if  is_found <15 or startX<0 or startY<0 or endX<0 or endY<0:
                        tracker=None
                        errorFrameNum+=1
                        totalTime+=time.time()-runTime
                        continue
                        
                    # 从关联对象跟踪器中绘制边界框
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 255, 0), 2)   
                        
                    #构造一个dlib矩形对象(坐标int类型)
                    rect = dlib.rectangle(startX,startY,endX,endY) 
                    
                    t1=time.time()
                    #确定面部区域的面部标志并将面部标志（x，y） -坐标转换为NumPy阵列。
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    FP_time+=time.time()-t1
                    
                    #取出左右眼对应的特征点
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    mouth = shape[mStart: mEnd]   
                        
                    #绘制左眼边框
                    stratX=int(leftEye[0][0]-abs(leftEye[0][0]-leftEye[3][0])*0.3)
                    stratY=int(min(leftEye[0][1],leftEye[3][1])-abs(leftEye[1][1]-leftEye[5][1])*1.5)
                    endX=int(leftEye[3][0]+abs(leftEye[0][0]-leftEye[3][0])*0.3)
                    endY=int(min(leftEye[0][1],leftEye[3][1])+abs(leftEye[1][1]-leftEye[5][1])*1.5)
                    cv2.rectangle(frame, (stratX,stratY), (endX, endY), (255, 0, 0), 1)
                        
                    #绘制右眼边框
                    stratX=int(rightEye[0][0]-abs(rightEye[0][0]-rightEye[3][0])*0.3)
                    stratY=int(min(rightEye[0][1],rightEye[3][1])-abs(rightEye[1][1]-rightEye[5][1])*1.5)
                    endX=int(rightEye[3][0]+abs(rightEye[0][0]-rightEye[3][0])*0.3)
                    endY=int(min(rightEye[0][1],rightEye[3][1])+abs(rightEye[1][1]-rightEye[5][1])*1.5)
                    cv2.rectangle(frame, (stratX,stratY), (endX, endY), (255, 0, 0), 1)
                        
                    #计算左眼和右眼的凸包，绘制左右眼轮廓
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    #计算嘴部凸包，绘制嘴部轮廓
                    mouthHull=cv2.convexHull(mouth);
                         
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                    ##########################计算眼部状态########################################## 
                    t1=time.time()
                    # 计算左眼EAR
                    leftEAR = eye_aspect_ratio(leftEye)
                    # 计算右眼EAR
                    rightEAR = eye_aspect_ratio(rightEye)
                    #求双眼纵横比的均值
                    ear = (leftEAR + rightEAR) / 2.0
                        
                    ##########眨眼检测
                    #眼睛纵横比连续小于阈值的帧数eyeThreshold=0.7
                    if ear < eyeThreshold:
                        eyeFrameNum += 1
                
                    # 如果EAR大于阈值
                    else:
                        # EAR小于阈值的总帧数大于等于EYE_AR_CONSEC_FRAMES=15记一次眨眼
                        if eyeFrameNum >= EYE_CONSEC_FRAMES:
                            dA.BF += 1
                        # 重置EAR小于阈值的总帧数
                        eyeFrameNum = 0
                            
                    ############睁闭眼状态检测#################
                    #横纵比向量
                    ret, ear_vector = queue_in(ear_vector, ear)
                    #当特征向量为VECTOR_SIZE
                    if(len(ear_vector) == VECTOR_SIZE):
                        #输出3维特征向量
                        input_vector = []
                        input_vector.append(ear_vector)
                        res = clf.predict(input_vector)
                        #输出检测结果
                        if res == "close":
                            #连续闭眼次数+1
                            eyeCloseNum += 1
                            #闭眼总帧数+1
                            dA.closeFrame+=1
                        else:
                            #计算最长连续闭眼时间
                            if (eyeCloseNum/fps) > dA.maxECT:
                                dA.maxECT = (eyeCloseNum/fps)
                            eyeCloseNum=0
                    eye_time+=time.time()-t1
                    ################end########################
                        
                    #################分析头部运动#########################
                    #根据眼部位置计算头部垂直位置y
                    heady_y=(leftEye[0][1]+rightEye[3][1])/2.0
                    headLocation.append(heady_y)
                    ###################end#############################
                        
                    #####################嘴部状态###########################
                    t1=time.time()
                    # 计算纵横比
                    MAR = mouth_aspect_ratio(mouth)
                    # 如果MAR大于阈值，开始计算连续帧
                    if MAR > mouthThreshold:
                        mouthFrameNum += 1
                    # 如果MAR小于阈值
                    else:
                        # MAR大于阈值的总帧数大于等于EYE_AR_CONSEC_FRAMES记一次眨眼
                        if mouthFrameNum >= Mouth_CONSEC_FRAMES:
                            dA.YF += 1
                        #求打哈欠最长持续时间
                        if (mouthFrameNum/fps)>dA.maxYT:
                            dA.maxYT=(mouthFrameNum/fps)
                                
                        # 重置MAR大于阈值的总帧数    
                        mouthFrameNum = 0
                    mouth_time+=time.time()-t1
                    ####################end##############################
                        
        #               # 循环（x，y） - 面部地标的坐标并在图像上绘制它们
        #               for (x, y) in shape:
        #                   cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
                #显示图像
                cv2.imshow("detection", frame)
                #获取周期内运行总时间
                totalTime += time.time()-runTime
                
                key = cv2.waitKey(1) & 0xFF
                # 按“q“退出循环
                if key == ord("q"):
                    break   
                
  
                    
            # 关闭窗口及视频流
            cv2.destroyAllWindows()
            cap.release()