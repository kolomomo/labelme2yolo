# labelme2yolo
labelme标签转换为yolo标签

`python labelme2yolo(...)`

参数：
```
labelme2yolo(labelme_image_path='labelme 图像文件夹目录',
                 labelme_json_path='labelme json文件夹目录',
                 save_data_path='转换yolo格式保存目录',
                 obj_name='提取的类别名称, labelme 里的 labels.txt',
                 dataset_type='train', # train/val/test
                 image_type='.jpg', # 图像格式后缀
                 )
```


