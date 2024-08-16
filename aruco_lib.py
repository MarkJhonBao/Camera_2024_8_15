import numpy as np
import cv2
import cv2.aruco as aruco
import sys
import math

'''
函数说明:
* angle_calculate(pt1,pt2, trigger = 0) - 返回两个点之间的角度函数
* detect_Aruco(img) - 返回检测到的id:corners的aruco列表字典
* mark_Aruco(img, aruco_list) - 功能标记中心并显示id
* calculate_Robot_State(img,aruco_list) - 给出bot的状态 (centre(x), centre(y), angle)
'''

def angle_calculate(pt1,pt2, trigger = 0):  #返回0-359范围内两个点之间的角度
    angle_list_1 = list(range(359,0,-1))
    #angle_list_1 = angle_list_1[90:] + angle_list_1[:90]
    angle_list_2 = list(range(359,0,-1))
    angle_list_2 = angle_list_2[-90:] + angle_list_2[:-90]
    x=pt2[0]-pt1[0] # unpacking tuple
    y=pt2[1]-pt1[1]
    angle=int(math.degrees(math.atan2(y,x))) #取2点相对于水平轴在范围(-180,180)内
    if trigger == 0:
        angle = angle_list_2[angle]
    else:
        angle = angle_list_1[angle]
    return int(angle)

def detect_Aruco(img):  #返回检测到的id: corners的aruco列表字典
    aruco_list = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)   #创建5位的aruco_dict，最多250个id, id范围从0到249
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # parameters = aruco.DetectorParameters_create()  #检测器初始化
    #检测参数设置
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshConstant = 10
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    parameters.minMarkerDistanceRate = 0.05
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    #id的列表以及属于每个id的角
    # corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    #corners is the list of corners(numpy array) of the detected markers. For each marker, its four corners are returned in their original order (which is clockwise starting with top left). So, the first corner is the top left corner, followed by the top right, bottom right and bottom left.
    # print corners[0]
    #gray = aruco.drawDetectedMarkers(gray, corners,ids)
    #cv2.imshow('frame',gray)
    #print (type(corners[0]))
    if len(corners):    #返回arucos的id
        #print (len(corners))
        #print (len(ids))
        print(type(corners))
        print(corners[0][0])
        for k in range(len(corners)):
            temp_1 = corners[k]
            temp_1 = temp_1[0]
            temp_2 = ids[k]
            temp_2 = temp_2[0]
            aruco_list[temp_2] = temp_1
        return aruco_list

def mark_Aruco(img, aruco_list):    #功能标记中心并显示id
    key_list = aruco_list.keys()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for key in key_list:
        dict_entry = aruco_list[key]    #dict_entry is a numpy array with shape (4,2)
        centre = dict_entry[0] + dict_entry[1] + dict_entry[2] + dict_entry[3]#so being numpy array, addition is not list addition
        centre[:] = [int(x / 4) for x in centre]    #finding the centre
        #print centre
        orient_centre = centre + [0.0, 5.0]
        #print orient_centre
        centre = tuple(centre)  
        orient_centre = tuple((dict_entry[0]+dict_entry[1])/2)
        #print centre
        #print orient_centre
        # 将元组中的元素转换为整数
        # 将元组转换为 numpy.ndarray
        x, y = int(centre[0].astype(int)), int(centre[1].astype(int))
        x0, y0 = int(orient_centre[0].astype(int)), int(orient_centre[1].astype(int))
        cv2.circle(img, (x, y), 1, (0, 0, 255), 8)
        #cv2.circle(img,tuple(dict_entry[0]),1,(0,0,255),8)
        #cv2.circle(img,tuple(dict_entry[1]),1,(0,255,0),8)
        #cv2.circle(img,tuple(dict_entry[2]),1,(255,0,0),8)
        #cv2.circle(img,orient_centre,1,(0,0,255),8)
        cv2.line(img, (x, y), (x0, y0),(255, 0, 0), 4) #marking the centre of aruco
        #cv2.line(img,centre,orient_centre,(255,0,0),4)
        cv2.putText(img, str(key), (int(centre[0] + 20), int(centre[1])), font, 1, (0,0,255), 2, cv2.LINE_AA) # displaying the idno
    return img

def calculate_Robot_State(img,aruco_list):  #给出Robot状态(中心(x)，中心(y)，角度)
    robot_state = {}
    key_list = aruco_list.keys()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for key in key_list:
        dict_entry = aruco_list[key]
        pt1 , pt2 = tuple(dict_entry[0]), tuple(dict_entry[1])
        centre = dict_entry[0] + dict_entry[1] + dict_entry[2] + dict_entry[3]
        centre[:] = [int(x / 4) for x in centre]
        centre = tuple(centre)
        angle = angle_calculate(pt1, pt2)
        cv2.putText(img, str(angle), (int(centre[0] - 80), int(centre[1])), font, 1, (0,0,255), 2, cv2.LINE_AA)
        robot_state[key] = (int(centre[0]), int(centre[1]), angle)#HOWEVER IF YOU ARE SCALING IMAGE AND ALL...THEN BETTER INVERT X AND Y...COZ THEN ONLY THE RATIO BECOMES SAME
    #print (robot_state)
    return robot_state    
