import numpy as np
import cv2
import cv2.aruco as aruco
 
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000) #创建aruco字典, 250个标记，标记大小为6x6位
print(aruco_dict)
# 第二个参数是id号，最后一个参数是总图像大小
# img = aruco.drawMarker(aruco_dict, 650, 400)
# 2--标记id，因为选择的字典多达250。因此id的取值范围为0 ~ 249;700x700是像素大小

img = aruco.generateImageMarker(aruco_dict, 650, 400)
print(img.shape)
cv2.imwrite("./test_marker11_big.jpg", img)
print(img.shape)
cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()