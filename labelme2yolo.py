# !/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
#-----------------------------------------------------#
   @File : labelme2yolo.py
   @Time : 2020/7/31 9:47 
   @Author : kolomomo 
   @Description :
   将labelme格式标签转换为yolo格式（yolov5测试通过）
#-----------------------------------------------------#  
   @Change Activity: 2020/7/31
   @Change Description :
#-----------------------------------------------------#
"""

import json
import numpy as np
import glob
from pathlib import Path
import shutil
import os

class labelme2yolo():
    def __init__(
            self,
            labelme_image_path ='./data_annotated',
            labelme_json_path = './data_annotated',
            save_data_path = './data_annotated_yolo',
            obj_name='labels.txt',
            dataset_type = 'train',
            image_type = '.jpg',
            ):

        self.labelme_image_path = labelme_image_path
        self.labelme_json_path = labelme_json_path
        self.save_data_path = save_data_path
        self.dataset_type = dataset_type
        self.save_yolo_dataset = dataset_type + '.txt'
        self.save_image_dir = 'obj_'+dataset_type+'_data'

        self.image_type = image_type
        self.width = 0
        self.height = 0
        self.labels = []
        self.labels_out = []

        with open(obj_name, 'r') as f:
            while True:
                n = f.readline().strip()
                if not n:
                    break
                if n[0] != '_':
                    self.labels.append(n)
                    self.labels_out.append(n + '\n')

        self.classes = len(self.labels)

        self.transfer()

    def parse_json2txt(self, json_file, path_txt):
        with open(json_file, 'r') as fp:
            with open(path_txt, 'w+') as ftxt:
                data = json.load(fp)
                self.height = data['imageHeight']
                self.width = data['imageWidth']
                for shape in data['shapes']:
                    label = shape['label']
                    shape_type =  shape['shape_type']

                    if shape_type != "rectangle":
                        print('只支持retangle标注')
                        break

                    if label in self.labels:
                        idx = self.labels.index(label)
                        points = shape['points']
                        x_center, y_center, w, h = self.point_to_box(points)

                        str_annotation = str(idx) + ' ' + str(x_center) + ' ' + \
                                         str(y_center) + ' ' + str(w) + ' ' + \
                                         str(h)+ '\n'
                        ftxt.writelines(str_annotation)
                    else:
                        print('label: {} 已忽略'.format(label))

    def point_to_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        w = max_x - min_x
        h = max_y - min_y
        c_x = min_x + w / 2.
        c_y = min_y + h / 2.
        # 归一化
        c_x_norm = c_x / self.width
        c_y_norm = c_y / self.height
        w_norm = w / self.width
        h_norm = h / self.height

        return (c_x_norm, c_y_norm, w_norm, h_norm)

    def transfer(self):
        labelme_json = glob.glob(str(Path(self.labelme_json_path))+'/*.json')
        save_dir = Path(self.save_data_path)

        if save_dir.exists():
            print('{} 已存在'.format(str(save_dir)))
            # shutil.rmtree(str(save_dir))
        else:
            print('创建目录 {}'.format(str(save_dir)))
            os.mkdir(save_dir)

        obj_image_dir = save_dir / self.save_image_dir

        if obj_image_dir.exists():
            print('{} 已存在'.format(str(obj_image_dir)))
        else:
            print('创建目录 {}'.format(str(obj_image_dir)))
            os.mkdir(obj_image_dir)


        yolo_dataset_txt = Path(self.save_data_path).joinpath(self.save_yolo_dataset)
        yolo_obj_names = Path(self.save_data_path).joinpath('obj.names')
        yolo_obj_data = Path(self.save_data_path).joinpath('obj.data')

        with open(yolo_dataset_txt, 'w+') as ytxt, open(yolo_obj_names, 'w') as obj_n, open(yolo_obj_data, 'w+') as obj_d:

            obj_n.writelines(self.labels_out)
            txt_1 = str(yolo_dataset_txt.relative_to(yolo_dataset_txt.parent.parent)).replace('\\', '/')
            txt_2 = str(yolo_obj_names.relative_to(yolo_obj_names.parent.parent)).replace('\\', '/')
            obj_data_text = 'classes = {}\n{} = {}\nnames = {}\nbackup = backup/'.format(self.classes, self.dataset_type, txt_1, txt_2)
            obj_d.writelines(obj_data_text)

            for json_file in labelme_json:
                print(json_file)
                fn = json_file.split('\\')[-1]
                fn_txt = fn.split('.')[0] + '.txt'
                fn_img = fn.split('.')[0] + self.image_type

                path_txt = obj_image_dir.joinpath(fn_txt)
                path_img = obj_image_dir.joinpath(fn_img)
                origin_path_img =  Path(self.labelme_image_path).joinpath(fn_img)

                if not origin_path_img.exists():
                    print('未找到标签对应图像文件：{}'.format(origin_path_img))
                    continue

                shutil.copyfile(origin_path_img, path_img)

                # 保存数据集路径
                save_path_name = str(path_img.relative_to(path_img.parent.parent.parent)).replace('\\','/')
                ytxt.writelines(save_path_name + '\n')

                self.parse_json2txt(json_file, path_txt)

if __name__ == '__main__':
    labelme2yolo(r'D:\SHARE\labelme\examples\instance_segmentation\all_deblured',
                                r'D:\SHARE\labelme\examples\instance_segmentation\all_deblured',
                                r'D:\SHARE\labelme2yolo\all_deblured',
                                r'D:\SHARE\labelme\examples\instance_segmentation\labels_deblured.txt')