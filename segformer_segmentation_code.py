#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/4/30 17:49
# @Author  : yhy
# @FileName: segformer_segmentation_code.py
# @Software: PyCharm
import os
import pandas as pd
import numpy as np
import mmcv
from mmseg.apis import inference_model, init_model, show_result_pyplot

# 类别映射字典，Cityscapes 的类别
class_mapping = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole',
    6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain',
    10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus',
    16: 'train', 17: 'motorcycle', 18: 'bicycle'
}

# 初始化一个字典，存储所有类别的占比初始值为0
all_categories = {v: 0 for v in class_mapping.values()}

# 语义分割图像 并计算不同类别占比
def get_seg_and_calculate_percentage(img_path, seg_img_path, model, showSegImg):
    img = mmcv.imread(img_path)
    result = inference_model(model, img)
    predict = result.pred_sem_seg.data.cpu().numpy()

    # 如果需要保存分割后的图像 则存储
    if showSegImg == True:
        # 保存合成后的图像
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(seg_img_path, f"{base_filename}_segment.png")
        # opacity 可以设置不同的透明度  范围0到1
        show_result_pyplot(model, img, result, show=False, out_file=save_path, opacity=0.5)

    # 计算每个类别的占比
    unique, counts = np.unique(predict[0], return_counts=True)
    percentages = {class_mapping.get(k, k): v / predict.size for k, v in zip(unique, counts)}
    return {**all_categories, **percentages}

# 对街景图像进行语义分割
def sv_segment(sv_jpg_folder, seg_img_folder, model, showSegImg=False):
    '''
    :param sv_jpg_folder:      街景图像存放文件夹
    :param seg_img_folder:     语义分割结果存放文件夹
    :param model:              加载的模型
    :param showImg:            是否语义分割图片  True保存 False 不保存
    :return:
    '''
    # 判断原始街景图像文件夹是否存在
    assert os.path.exists(sv_jpg_folder), f"file: '{sv_jpg_folder}' dose not exist."
    # 判断语义分割存放结果文件夹是否存在 若不存在则创建
    if not os.path.exists(seg_img_folder):
        os.makedirs(seg_img_folder)  # 创建保存分割结果的目录


    # 图像文件
    jpg_files = [os.path.join(sv_jpg_folder, f) for f in os.listdir(sv_jpg_folder)]  # 获取所有街景图像

    # 初始化一个列表，用于存储所有结果的字典
    results_list = []

    # 输出的csv文件
    csv_file_path = os.path.join(f'seg_result_statistics.csv')  # 设置CSV文件路径

    # 处理每个JPEG文件
    for sv_img in jpg_files:
        img_name = os.path.basename(sv_img)  # 获取图像名
        try:
            # 语义分割街景图像并提取信息  类别-占比
            category_percentage = get_seg_and_calculate_percentage(sv_img, seg_img_folder, model, showSegImg)
            # 存储数据的dataframe
            result_dict = {'image_name': os.path.basename(sv_img),  **category_percentage}
            results_list.append(result_dict)
            print(result_dict)

            # 将数据导出到csv中

        except Exception as e:
            print(f"Error processing file {sv_img}: {e}")  # 打印错误信息

    # 将最终结果输出
    final_result = pd.DataFrame(results_list)
    final_result.to_csv(csv_file_path,  index=False)
    print(f'输出文件存放在--> {csv_file_path} ')

if __name__ == '__main__':
    seg_result_csv = 'seg_result.csv' # 存储语义分割结果的csv文件
    sv_jpg_folder = './sv_img/'       # 存放原始街景图像的文件夹
    seg_img_folder = './sv_img_seg/'  # 存放语义分割后的文件夹
    # Segformer 语义分割网络 （基于cityscapes数据集）
    config_file = 'configs/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'
    checkpoint_file = 'checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
    model = init_model(config_file, checkpoint_file, device='cuda:0')

    # 进行语义分割
    sv_segment(sv_jpg_folder, seg_img_folder, model, showSegImg=True)
