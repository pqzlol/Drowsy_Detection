# coding=utf-8
# 测试ZJU眨眼数据库(80个视频)

import numpy as np
import cv2
import imutils  # 图像处理包

# 从视频中获取
# path='zjublinkdb/000020F_UNN.avi'
path = 'NTHU/night_glassses/03_Normal.avi'
# 捕捉视频，未开始读取
#cap = cv2.VideoCapture(path)
# 调用摄像头获取
cap = cv2.VideoCapture(0)
# 视频帧数
# frame_counter=0
# frame_counter=138
# frame_counter=274
# frame_counter=416
# frame_counter=552
frame_counter = 57930
# 读第一帧
success, img = cap.read()
num = frame_counter
# 循环直到视频结束
while success:
    frame_counter += 1
    # 调整大小
    img = imutils.resize(img, width=500)
    cv2.imwrite("test_data/" + str(frame_counter) + ".jpg", img)
    #cv2.imshow("img", img)
    # 按q退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # 获取下一帧
    success, img = cap.read()

cap.release()

cv2.destroyAllWindows()
print "frame_counter:{}".format(frame_counter)
# 视频帧数
print "num:{}".format(frame_counter - num)
