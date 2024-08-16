# 此文件用于检测模型的训练和保存
import os
import cv2
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection._utils import retrieve_out_channels

import numpy as np
import matplotlib.pyplot as plt

global index2name
# 1: green plants, 2: flowerpot, 3, aruco code
index2name = [
        "1",
        "2",
        "3",
    ]
global name2index
name2index = {}
for i in range(len(index2name)):
    name2index[index2name[i]] = i
print(name2index)

def precess_image(path):
    import os, cv2
    import glob
    from PIL import Image
    folder_path = path
    # 遍历文件夹中的文件
    id = 1
    for file_path in glob.glob(os.path.join(folder_path, '*')):
        # 检查文件类型
        if os.path.isfile(file_path):
            # 检查文件扩展名
            file_name, file_ext = os.path.splitext(file_path)
            if file_ext.lower() == '.json':
                # 删除 JSON 文件
                os.remove(file_path)
            else:
                # 转换图像为 JPG 格式
                img = cv2.imread(file_path)
                os.remove(file_path)
                # cv2.imwrite(new_file_path, img)
                # 重新编号为 6 位数的 ID 名称
                new_file_name = str(id)  # 去除扩展名
                new_file_name = new_file_name.zfill(6)  # 补零
                new_file_name += '.jpg'
                new_file_path = os.path.join(folder_path, new_file_name)
                # os.rename(file_path, new_file_path)
                cv2.imwrite(new_file_path, img)
                id += 1

def train(dataset_train, dataset_test):
    # 创建数据加载器
    batch_size = 1  # 批处理大小
    shuffle = True  # 是否打乱数据顺序
    num_workers = 4  # 数据加载器的工作进程数

    # 使用训练数据加载器进行迭代:训练代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载预训练的 MobileNet 网络作为主干网络
    # model = torchvision.models.detection.ssd(pretrained=True)

    # 加载预训练的 SSD 检测器，使用 MobileNet 网络作为主干
    backbone = torchvision.models.detection.ssd300_vgg16(
        weights=None, progress=True, num_classes=3, trainable_backbone_layers=5)
    model = backbone.to(device)
    model.out_channels = 1280

    #设置预定义的类别个数3
    # 创建 VOC 数据集对象
    # dataset_root = r'D:\Infant_Multi\Scripts\xy-2024-07013-LaneDetection\image\VOCdevkit'  # VOC 数据集根目录
    # dataset_train = VOCDetection(root=dataset_root, year='2007', image_set='train', transform=transform)
    # dataset_test = VOCDetection(root=dataset_root, year='2007', image_set='test', transform=transform)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                   shuffle=shuffle, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                  shuffle=shuffle, num_workers=num_workers)


    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # 定义优化器和学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    loss_fn = torch.nn.CrossEntropyLoss()

    class_to_id = {
        'aeroplane': 0,
        'bicycle': 1,
        'bird': 2,
        'boat': 3,
        'bottle': 4,
        'bus': 5,
        'car': 6,
        'cat': 7,
        'chair': 8,
        'cow': 9,
        'diningtable': 10,
        'dog': 11,
        'horse': 12,
        'motorbike': 13,
        'person': 14,
        'pottedplant': 15,
        'sheep': 16,
        'sofa': 17,
        'train': 18,
        'tvmonitor': 19
    }

    # 训练循环
    num_epochs = 10  # 训练迭代次数
    step = 0
    model.train()  # 设置模型为训练模式

    for epoch in range(num_epochs):
        for images, targets in dataloader_train:
            images = images.to(device)
            # targets = targets.to(device)
            target = targets2dic(targets)
            outputs = model(images, target)
            img_item_path = os.path.join(dataset_train.root,"VOCdevkit\VOC2007", "JPEGImages", targets["annotation"]["filename"][0])
            # visualization(outputs, img_item_path)
            step += 1
    # 保存模型
    torch.save(model.state_dict(), 'mobilenet_model.pth')

def model_save():
    import torch
    import torch.onnx
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    # 创建模型实例
    model = ssdlite320_mobilenet_v3_large(pretrained=True)
    # 加载训练好的模型权重
    model.load_state_dict(torch.load('mobilenet_model.pth'))
    # 设置模型为评估模式
    model.eval()
    # 创建输入张量
    dummy_input = torch.randn(1, 3, 300, 300)
    # 导出模型为 ONNX 格式
    onnx_path = 'mobilenet_model.onnx'
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

def model_inference():
    import onnx
    import numpy as np
    import onnxruntime
    # 加载 ONNX 模型
    onnx_path = 'mobilenet_model.onnx'
    onnx_model = onnx.load(onnx_path)
    # 创建 ONNX Runtime 的推理会话
    ort_session = onnxruntime.InferenceSession(onnx_path)
    # 准备输入数据
    input_name = ort_session.get_inputs()[0].name
    input_data = np.random.randn(1, 3, 300, 300).astype(np.float32)  # 替换为真实的输入数据
    # 进行推理
    output = ort_session.run(None, {input_name: input_data})

def label_transform(targets):
    target_list = []
    targets = targets
    for item in targets['annotation']['object']:
        tensor_list, dict_item = [], {}
        # 将值转换为张量
        tensor1 = torch.tensor(item['bndbox']['xmin'])
        tensor2 = torch.tensor(item['bndbox']['ymin'])
        tensor3 = torch.tensor(item['bndbox']['xmax'] - item['bndbox']['xmin'])
        tensor4 = torch.tensor(item['bndbox']['ymax'] - item['bndbox']['ymin'])
        # 添加到列表中
        tensor_list.append(tensor1, tensor2, tensor3, tensor4)
        tensor_label = torch.tensor(item['name'])
        # 将列表转换为张量
        tensor_bbox = torch.stack(tensor_list, dim=1)  # 将值转换为张量

    dict_item['bboxes'] = tensor_bbox, dict_item['labels'] = tensor_label,
    target_list.append(dict_item)
    return target_list

def list_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(file)
    return file_list

def save_to_txt(file_list, output_file):
    with open(output_file, 'w') as file:
        for file_name in file_list:
            file.write(file_name.replace('.jpg', '') + '\n')

def file_process():
    # precess_image(r"H:\xy\888")
    # 指定要遍历的文件夹路径
    folder_path = r'D:\Infant_Multi\Scripts\xy-2024-07013-LaneDetection\image\ImageSets'
    # 调用函数获取文件列表
    files = list_files(folder_path)
    # 指定要保存文件名称的txt文件路径
    txt_file_path = r'D:\Infant_Multi\Scripts\xy-2024-07013-LaneDetection\image\VOCdevkit\VOC2007\ImageSets\Main\test.txt'
    # 保存文件名称到txt文件
    save_to_txt(files, txt_file_path)
    exit(0)

def labelme2voc():
    import os
    import json
    import xml.etree.ElementTree as ET
    def convert_json_to_xml(json_file, output_dir):
        with open(json_file, 'r') as f:
            data = json.load(f)
        # 创建根节点和子节点
        root = ET.Element("annotation")
        folder = ET.SubElement(root, "folder")
        filename = ET.SubElement(root, "filename")
        size = ET.SubElement(root, "size")
        width = ET.SubElement(size, "width")
        height = ET.SubElement(size, "height")
        depth = ET.SubElement(size, "depth")
        # 设置节点的文本内容
        folder.text = "images"
        filename.text = os.path.splitext(os.path.basename(json_file))[0] + ".jpg"
        width.text = str(data["imageWidth"])
        height.text = str(data["imageHeight"])
        depth.text = "3"  # 默认设置为RGB图像
        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]
            # 创建对象节点和子节点
            object_elem = ET.SubElement(root, "object")
            name = ET.SubElement(object_elem, "name")
            bndbox = ET.SubElement(object_elem, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            ymin = ET.SubElement(bndbox, "ymin")
            xmax = ET.SubElement(bndbox, "xmax")
            ymax = ET.SubElement(bndbox, "ymax")
            # 设置对象节点的文本内容
            name.text = label
            xmin.text = str(min(points, key=lambda x: x[0])[0])
            ymin.text = str(min(points, key=lambda x: x[1])[1])
            xmax.text = str(max(points, key=lambda x: x[0])[0])
            ymax.text = str(max(points, key=lambda x: x[1])[1])
        # 创建 XML 文档并保存
        xml_tree = ET.ElementTree(root)
        xml_tree.write(os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + ".xml"))
    # 指定包含 JSON 文件的文件夹路径
    json_folder = r'D:\Infant_Multi\Scripts\xy-2024-07013-LaneDetection\image\VOCdevkit\VOC2007\Annotations'
    # 指定输出 XML 文件的文件夹路径
    output_folder = r'D:\Infant_Multi\Scripts\xy-2024-07013-LaneDetection\image\VOCdevkit\VOC2007\Annotations'
    # 遍历文件夹中的所有 JSON 文件并进行转换
    for file_name in os.listdir(json_folder):
        if file_name.endswith('.json'):
            json_file = os.path.join(json_folder, file_name)
            # output_dir = os.path.join(output_folder, os.path.splitext(file_name)[0])
            # os.makedirs(output_dir, exist_ok=True)
            convert_json_to_xml(json_file, output_folder)

def visualization(predictions, img_path):
    # 加载图像
    image = cv2.imread(img_path)
    # 使用OpenCV显示结果, 遍历每个预测框并在图像上绘制
    for boxs, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score.item() > 0.5:  # 设置一个阈值来过滤低置信度的预测
            # box = boxs[0].item()
            image = cv2.rectangle(image, (int(boxs[0]), int(boxs[1])), (int(boxs[2]), int(boxs[3])), (0, 255, 0), 2)
            image = cv2.putText(image, f"{label.item()}: {score.item():.2f}", (int(boxs[0]), int(boxs[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
            # 显示图像
            cv2.imshow('Detection Results', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def id2vocname(class_id):
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    return VOC_CLASSES[class_id - 1]

def targets2dic(targets):
    target = []
    if len(targets["annotation"]["object"]) > 0:
        item_ = {}
        for item in targets["annotation"]["object"]:
            item_["boxes"], item_["labels"] = [], []
            xmin = item["bndbox"]["xmin"][0]
            xmin = float(xmin)
            ymin = item["bndbox"]["ymin"][0]
            ymin = float(ymin)
            W = float(item["bndbox"]["xmax"][0]) #- float(item["bndbox"]["xmin"][0])
            H = float(item["bndbox"]["ymax"][0]) #- float(item["bndbox"]["ymin"][0])
            item_["boxes"].append(xmin)
            item_["boxes"].append(ymin)
            item_["boxes"].append(W)
            item_["boxes"].append(H)
            item_["boxes"] = torch.tensor(item_["boxes"]).type(torch.LongTensor)
            item_["boxes"] = torch.unsqueeze(item_["boxes"], 0).cuda()
            label = int(item["name"][0])
            item_["labels"] = torch.tensor([label]).type(torch.LongTensor)
            item_["labels"] = torch.unsqueeze(item_["labels"], 0).cuda()
        target.append(item_)
    else:
        return target
    return target

def arrange_target(target):
    objects = target["annotation"]["object"]
    box_dics = [obj["bndbox"] for obj in objects]
    box_keys = ["xmin", "ymin", "xmax", "ymax"]
    boxes = []
    for box_dic in box_dics:
        box = [int(float(box_dic[key])) for key in box_keys]
        boxes.append(box)
    boxes = torch.tensor(boxes)
    # 物体名
    labels = [name2index[obj["name"]] for obj in objects]
    labels = torch.tensor(labels)
    dic = {"boxes":boxes, "labels":labels}
    return dic

def show_boxes(image, boxes, names):
    drawn_boxes = draw_bounding_boxes(image, boxes[:1, :], labels=names[:1])
    plt.figure(figsize = (16,16))
    plt.imshow(np.transpose(drawn_boxes, (1, 2, 0)))
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    plt.show()


if __name__ == "__main__":
    # 标记的数据转化为VOC的XML格式
    # labelme2voc()
    # exit()
    print(index2name)
    dataset_root = r'D:\Infant_Multi\Scripts\xy-2024-07013-LaneDetection\image'  # VOC 数据集根目录
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        # transforms.Resize([320, 320]),
    ])

    # 创建 VOC 数据集对象，创建数据加载器
    batch_size = 1  # 批处理大小
    shuffle = True  # 是否打乱数据顺序
    num_workers = 4  # 数据加载器的工作进程数
    dataset_train = VOCDetection(root=dataset_root, year='2007',
                                 image_set='train',
                                 transform=transform,
                                 target_transform=transforms.Lambda(arrange_target),
                                 download=False)
    dataset_test = VOCDetection(root=dataset_root, year='2007',
                                image_set='test',
                                transform=transform,
                                target_transform=transforms.Lambda(arrange_target),
                                download=False)
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
    dataiter = iter(data_loader_train)
    image, target = next(dataiter)
    print(target)
    image = image[0]
    image = (image * 255).to(torch.uint8)  # draw_bounding_boxes到0-255
    boxes = target["boxes"][0]
    labels = target["labels"][0]
    names = [index2name[label.item()] for label in labels]
    show_boxes(image, boxes, names)

    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

    #替换输出分类结果的地方
    in_channels = retrieve_out_channels(model.backbone, (300, 300))  #输入的通道数
    num_anchors = model.anchor_generator.num_anchors_per_location()  #预测锚框的数量
    num_classes = len(index2name) + 1  #分类数:为了将背景也包含在内进行分类而添加1
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)

    # Backbone权重进行冻结
    for p in model.parameters():
        p.requires_grad = False
    # classification_head进行迁移学习
    for p in model.head.classification_head.parameters():
        p.requires_grad = True
    model.cuda()  # 支持GPU

    # 优化算法
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)

    # 训练模式
    model.train()
    epochs = 200
    for epoch in range(epochs):
        for i, (image, target) in enumerate(data_loader_train):
            image = image.cuda()  # 支持GPU
            boxes = target["boxes"][0].cuda()
            labels = target["labels"][0].cuda()
            target = [{"boxes": boxes, "labels": labels}]  # 目标是以字典为要素的列表
            loss_dic = model(image, target)
            loss = sum(loss for loss in loss_dic.values())  # 计算误差的总和
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:  # 每100次显示一次
                print("epoch:", epoch, "iteration:", i, "loss:", loss.item())
    # 保存整个模型
    torch.save(model, 'complete_model.pth')
    # 加载整个模型
    # loaded_model = torch.load('complete_model.pth')
    dataiter = iter(data_loader_test)  # 数据输入迭代器
    image, target = next(dataiter)  # 取出批次数据
    image = image.cuda()  # 数据移送到gpu
    model.eval()
    predictions = model(image)
    image = (image[0] * 255).to(torch.uint8).cpu()  # draw_bounding_boxes函数的输入为0-255
    boxes = predictions[0]["boxes"].cpu()
    labels = predictions[0]["labels"].cpu().detach().numpy()
    labels = np.where(labels >= len(index2name), 0, labels)  # 标签不在范围内时标记为0
    names = [index2name[label.item()] for label in labels]
    # print(names)
    show_boxes(image, boxes, names)
    boxes = []
    names = []
    for i, box in enumerate(predictions[0]["boxes"]):
        score = predictions[0]["scores"][i].cpu().detach().numpy()
        if score > 0.8:  #抽出得分大于0.5的部分
            boxes.append(box.cpu().tolist())
            label = predictions[0]["labels"][i].item()
            if label >= len(index2name):  #标签不在范围的情况下为0
                label = 0
            name = index2name[label]
            names.append(name)
    boxes = torch.tensor(boxes)
    show_boxes(image, boxes, names)
    # train(dataset_train, dataset_test)
    exit(0)