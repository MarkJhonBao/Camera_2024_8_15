import os
import json
import shutil
import xml.etree.ElementTree as ET

def labelme_to_voc(labelme_dir, output_dir):
    # 创建VOC格式的目录结构
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, 'JPEGImages')
    annotation_dir = os.path.join(output_dir, 'Annotations')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)

    # 遍历Labelme标注文件
    for filename in os.listdir(labelme_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(labelme_dir, filename)
            with open(json_path, 'r') as f:
                data = json.load(f)

            # 解析Labelme标注文件
            image_filename = data['imagePath']
            image_path = os.path.join(labelme_dir, image_filename)
            image_basename = os.path.basename(image_path)
            image_dest_path = os.path.join(image_dir, image_basename)
            annotation_filename = os.path.splitext(image_basename)[0] + '.xml'
            annotation_path = os.path.join(annotation_dir, annotation_filename)

            # 复制图像文件到JPEGImages目录
            # os.rename(image_path, image_dest_path)
            shutil.copy(image_path, image_dest_path)

            # 生成VOC格式的XML文件
            root = ET.Element('annotation')
            folder = ET.SubElement(root, 'folder')
            folder.text = 'VOC2007'
            filename = ET.SubElement(root, 'filename')
            filename.text = image_basename

            size = ET.SubElement(root, 'size')
            width = ET.SubElement(size, 'width')
            height = ET.SubElement(size, 'height')
            depth = ET.SubElement(size, 'depth')
            width.text = str(data['imageWidth'])
            height.text = str(data['imageHeight'])
            # depth.text = str(data['imageDepth'])
            depth.text = str(3)

            for shape in data['shapes']:
                object_elem = ET.SubElement(root, 'object')
                name = ET.SubElement(object_elem, 'name')
                name.text = shape['label']
                bndbox = ET.SubElement(object_elem, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                ymin = ET.SubElement(bndbox, 'ymin')
                xmax = ET.SubElement(bndbox, 'xmax')
                ymax = ET.SubElement(bndbox, 'ymax')
                xmin.text = str(int(shape['points'][0][0]))
                ymin.text = str(int(shape['points'][0][1]))
                xmax.text = str(int(shape['points'][1][0]))
                ymax.text = str(int(shape['points'][1][1]))

            tree = ET.ElementTree(root)
            tree.write(annotation_path)
            print(f'Converted {json_path} to VOC format.')

if __name__ == "__main__":
    labelme_dir = r'H:\xy\888\3'
    output_dir = r'D:\Infant_Multi\Scripts\xy-2024-07013-LaneDetection\image'
    labelme_to_voc(labelme_dir, output_dir)


