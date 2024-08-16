import cv2
import torch
from torchvision.models.detection import ssd300_vgg16
from tools.train import *
from aruco_lib import *

def aruco_detect(names, frame):
    robot_state = {}
    frame = cv2.resize(frame, (512, 512))
    det_aruco_list = detect_Aruco(frame)
    if (det_aruco_list):
        img = mark_Aruco(frame, det_aruco_list)
        robot_state = calculate_Robot_State(img, det_aruco_list)
        frame = img
    return robot_state, frame

def calculate_midpoint(pt):
    x1, y1, x2, y2 = pt
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2
    return [midpoint_x, midpoint_y]

def image_detect(model, image):
    # 使用transforms.ToTensor将PIL Image转换为torch.Tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # 将图片转换为Tensor
    image = transform(image)
    # 增加维度,模型期望的输入格式是 (batch_size, channels, height, width),需要在第0维增加一个维度
    image = image.unsqueeze(0)
    predictions = model(image)
    image = (image[0] * 255).to(torch.uint8).cpu()  # draw_bounding_boxes函数的输入为0-255
    boxes = predictions[0]["boxes"].cpu()
    labels = predictions[0]["labels"].cpu().detach().numpy()
    labels = np.where(labels >= len(index2name), 0, labels)  # 标签不在范围内时标记为0
    names = [index2name[label.item()] for label in labels]
    # show_boxes(image, boxes, names)
    boxes = []
    names = []
    for i, box in enumerate(predictions[0]["boxes"]):
        score = predictions[0]["scores"][i].cpu().detach().numpy()
        if score > 0.5:  # 抽出得分大于0.5的部分
            boxes.append(box.cpu().tolist())
            label = predictions[0]["labels"][i].item()
            if label >= len(index2name):  # 标签不在范围的情况下为0
                label = 0
            name = index2name[label]
            names.append(name)
    # boxes = torch.tensor(boxes)
    # 可视化检测结果
    # show_boxes(image, boxes, names)
    # 应用函数到每个内部列表
    boxes = [calculate_midpoint(pt) for pt in boxes]
    print(boxes)
    return names, boxes

# 参数导入与距离计算、相机矩阵和畸变系数（这些是从相机标定过程中获得的）
def img_rectify(image):
    camera_matrix = np.array([[-261.63, 0, 523.48],
                              [0, -395.100, 700.28],
                              [0, 0, 1]])
    dist_coeffs = np.array( [[-4.792711299063557], [17.273473625894518],
                             [-26.870043899005253], [11.94322427701912]])  # 替换为实际的畸变系数
    # 读取图像尺寸
    h, w = image.shape[:2]
    # 优化相机矩阵（可选）
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    # 使用cv2.undistort进行畸变校正
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    # 裁剪图像以去除可能的黑色边缘
    # x, y, w, h = roi
    # undistorted_image = undistorted_image[y:y+h, x:x+w]
    return undistorted_image

def distance_compute(x, y, pixel_width_of_object = 200):
    # 已知物体的宽度（例如，使用校准图案测量的棋盘格宽度）
    known_object_width = 0.1  # 例如，10厘米
    # 物体在图像中的像素宽度（从图像处理获得）
    pixel_width_of_object = pixel_width_of_object  # 假设值
    # 假设你已经有了相机矩阵和畸变系数
    camera_matrix = np.array([[-261.63, 0, 523.48],
                              [0, -395.100, 700.28],
                              [0, 0, 1]])
    dist_coeffs = np.array([[-4.792711299063557], [17.273473625894518], [-26.870043899005253], [11.94322427701912]])  # 替换为实际的畸变系数
    # 计算焦距，假设已知相机到物体的实际距离（例如，使用其他方法测量）
    known_distance_to_object = 1.0  # 例如，1米
    focal_length = (pixel_width_of_object * camera_matrix[0, 0]) / known_object_width
    known_object_width = 0.1
    focal_length = (pixel_width_of_object * camera_matrix[0, 0]) / known_object_width
    # 图像中的像素坐标点
    pixel_points = np.array([[x, y]], dtype=np.float64)  # 替换x和y为实际的像素坐标
    # 归一化像素坐标
    normalized_points = cv2.undistortPoints(pixel_points, camera_matrix, dist_coeffs, P=None, R=None)
    # 去畸变后的点坐标
    x_ud, y_ud = normalized_points[0][0]
    # 计算去畸变点到相机的实际距离，假设物体的宽度在图像中是已知的，并且物体是平行于图像平面的，使用相似三角形原理来估计距离
    estimated_distance = (known_object_width * focal_length) / pixel_width_of_object
    print(f"Undistorted Coordinates: ({x_ud}, {y_ud})")
    print(f"Estimated Distance to Camera: {estimated_distance} meters")
    return estimated_distance

if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # 调用函数计算距离
    distance = distance_compute(20, 30)
    # print(f"The actual distance from the image center to the camera is: {distance} meters.")
    # exit(0)
    # 检测模型载入
    model_path = r".\\tools\complete_model.pth"
    model = ssd300_vgg16(pretrained=model_path).eval()
    # 打开视频文件
    cap = cv2.VideoCapture(0)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
    # 间隔帧数、获取视频的FPS（每秒帧数）
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 计算每帧的展示时间（以毫秒为单位）
    frame_interval = int((1 / fps) * 1000)
    # frame_interval = 0
    while True:
        # 读取帧
        ret, frame = cap.read()
        # 如果读取视频，注释掉此行即可
        # frame = cv2.imread(r"D:\E-StudyData\Camera_Measure_Distance\aruco\images\1.jpg")
        frame = cv2.resize(frame, [512, 512])
        frame = img_rectify(frame)
        # 如果正确读取帧，ret为True
        if not ret:
            print("Error: No more frames to read.")
        # 每隔i帧处理一次
        if frame_interval == 0:
            # 处理帧的代码
            names, boxes = image_detect(model, frame)
            robot_state, img = aruco_detect(names, frame)
            # 使用zip函数将两个列表的元素配对
            for item1, item2 in zip(names, boxes):
                print(f"Class: {item1}, Location: {item2}")
            for item in robot_state:
                print(f"Aruco: {item}")
            # 显示帧, 按'q'退出
            # cv2.imshow('Frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            frame_interval -= 1
        exit(0)
    # 释放视频捕获对象
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()