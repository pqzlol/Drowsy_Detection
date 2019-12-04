#coding=utf-8
# import the necessary packages
from imutils import face_utils
import numpy as np
import cv2
 
class FaceAligner:
    # 我们的构造函数有4个参数：
    # 预测器  ：面部标志预测模型。
    # desiredLeftEye  ：显示默认值的可选（x，y）元组，指定所需的左眼输出位置。
    #     对于该变量，通常看到百分比在20-40％的范围内。这些百分比控制对齐后可见的脸部数量。使用的确切百分比将因应用程序而异。
    #     使用20％你将基本上获得面部的“放大”视图，而使用更大的值时，面部将更加“缩小”。
    # desiredFaceWidth  ：另一个可选参数，用于以像素为单位定义所需的面。我们将此值默认为256像素。
    # desiredFaceHeight  ：最终的可选参数，以像素为单位指定所需的面高度值
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        #存储面部标志预测器，所需输出左眼位置和所需输出面宽+高
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
 
        #决定面部框是正方形图像或矩形的
        # 如果所需的面高度为无，则将其设置为所需的面宽（正常行为）
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
    
    #参数:RGB输入图像，灰度输入图像,由dlib的HOG人脸检测器生成的边界框矩形
    def align(self, image, gray, rect):
        # 应用dlib的面部地标预测器，并将地标转换为NumPy格式的（x，y）坐标。
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
 
        # 提取左眼和右眼（x，y） - 坐标
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        
        # 分别通过平均每只眼睛的所有（x，y）点来计算每只眼睛的质心
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
 
        # 计算眼睛质心之间的角度
        #给定眼睛中心，我们可以计算（x，y） -坐标的差异并采用反正切来获得眼睛之间的旋转角度。该角度将允许我们校正旋转。
        #为了确定角度，我们首先计算y方向 的delta，即dY
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        #计算x方向上的delta 
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        #计算面部旋转的角度。使用带有参数dY和 dX的 NumPy的 arctan2函数  ，然后转换为度数，同时减去180以获得角度。
        #numpy.arctan(x)：三角反正切  numpy.degrees(x)：弧度转换为度
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        # 根据左眼所需的x坐标计算所需的右眼x坐标
        #1.0减去 desiredLeftEye[0]因为 desiredRightEyeX值应该与图像的右边缘等距，因为相应的左眼x坐标是从其左边缘开始的。
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
 
        #＃通过获取* current *图像中眼睛之间的距离与*所需*图像中眼睛之间距离的比率来确定新结果图像的比例
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        #计算的欧氏距离比值。使用右眼和左眼x值之间的差值，我们计算所需的距离
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        #根据所需的宽度来缩放我们的眼睛距离。
        desiredDist *= self.desiredFaceWidth
        #通过将desiredDist除以  先前计算的 dist来计算我们的比例
        scale = desiredDist / dist
        
        #现在有旋转角度和比例 ,需要在计算仿射变换之前采取几个步骤。
        #这包括找到眼睛之间的中点以及计算旋转矩阵并更新其平移组件
        # 计算中心（x，y） - 输入图像中两只眼睛之间的坐标（即中间点），这个中点位于鼻子顶部，是我们旋转脸部的点
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
        #获取旋转矩阵以旋转和缩放面部
        #cv2.getRotationMatrix2D参数：
        #    眼睛中心  ：眼睛之间的中点是我们将面部旋转的点。
        #    角度  ：我们将角度旋转到的角度，以确保眼睛沿着相同的水平线。
        #    尺度  ：我们将放大或缩小图像的百分比，确保图像缩放到所需的大小。
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
 
        # 更新矩阵的转换组件
        #取一半 desiredFaceWidth   并将值存储为 tX  ，即x方向的转换
        tX = self.desiredFaceWidth * 0.5
        #计算 tY，y方向的平移，我们将desiredFaceHeight乘以   所需的左眼y值， desiredLeftEye[1]  。
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        #使用 tX 和 tY ，我们通过从相应的眼睛中点值eyesCenter中减去每个值来更新矩阵的平移分量 。
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        #仿射变换来调整脸部
        #为方便起见，我们将desiredFaceWidth   和 desiredFaceHeight   分别存储   到 w   和 h中
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        #调用cv2.warpAffine。此函数调用需要3个参数和1个可选参数：
        #图像  ：脸部图像。
        #M  ：平移，旋转和缩放矩阵。M作为仿射变换矩阵，一般反映平移或旋转的关系
        #（w ，h ）  ：输出图像的大小。
        #flags  ：用于warp的插值算法，在本例中为 INTER_CUBICflages表示插值方式，
        # 默认为 flags=cv2.INTER_LINEAR，表示线性插值 cv2.INTER_NEAREST（最近邻插值）  
        # cv2.INTER_AREA （区域插值）  cv2.INTER_CUBIC（三次样条插值）   cv2.INTER_LANCZOS4（Lanczos插值）
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
 
        # 返回对齐的人脸图像
        return output
