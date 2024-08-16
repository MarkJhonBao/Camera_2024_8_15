import numpy as np
import cv2
import cv2.aruco as aruco
from aruco_lib import *
import time

cap = cv2.VideoCapture(0)
# det_aruco_list = {}
while (True):
	ret,frame = cap.read()
	frame = cv2.imread(r"D:\Infant_Multi\Scripts\xy-2024-07013-LaneDetection\image\ImageSets\000061.jpg")
	frame = cv2.resize(frame, (512, 512))
	det_aruco_list = detect_Aruco(frame)
	if(det_aruco_list):
		img = mark_Aruco(frame, det_aruco_list)
		robot_state = calculate_Robot_State(img, det_aruco_list)
	cv2.imshow('image',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		# break
		exit(0)
	exit(0)
cap.release()
cv2.destroyAllWindows()
