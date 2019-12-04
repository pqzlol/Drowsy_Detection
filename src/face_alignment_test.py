#coding=utf-8
# import the necessary packages
from face_alignment import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2

#dlib预训练的面部标志检测器的路径
facial_landmark_predictor_path='model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(facial_landmark_predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=256)


image = cv2.imread("img/1.jpg")
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 

cv2.imshow("Input", image)
rects = detector(gray, 2)

#遍历 rects  ，对齐每个面，并显示原始和对齐的图像
for rect in rects:
    # 提取*原始*脸部的ROI，然后使用面部地标对齐脸部
    (x, y, w, h) = rect_to_bb(rect)
    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
    faceAligned = fa.align(image, gray, rect)
 
    # 显示图像
    cv2.imshow("Original", faceOrig)
    cv2.imshow("Aligned", faceAligned)
    cv2.waitKey(0)