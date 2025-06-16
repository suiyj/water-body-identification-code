import os
import shutil
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F  # 导入 torch.nn.functional
import re
import pandas as pd
# import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, Dataset
import random
import yaml
# from ConvNeXt6Channel_Model import ConvNeXt
from tqdm import tqdm
import geopandas as gpd
from ConvNeXt6Channel_Model import convnext_base, convnext_small, convnext_tiny


def read_img(img_dirs, img_name):
    transform_rgb = transforms.Compose([
        transforms.Resize(size=512),
        transforms.CenterCrop(size=512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_gray = transforms.Compose([
        transforms.Resize(size=512),
        transforms.CenterCrop(size=512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.459], std=[0.226])])

    imgs = []
    for img_dir in img_dirs:
        img_path = os.path.join(img_dir[0], '{}'.format(img_name))
        if img_dir[1] == 0:
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 使用OpenCV加载图像
            image = cv2.resize(image, (512, 512))
            image = Image.fromarray(image)
            image = transform_rgb(image)
            imgs.append(image)
        elif img_dir[1] == 1:  # slope
            image = cv2.imread(img_path, 2)  # 使用OpenCV加载图像
            image[np.isnan(image)] = 0
            image[np.isinf(image)] = 0
            image[image < 0] = 0
            image = cv2.resize(image, (512, 512))
            image = Image.fromarray(image)
            image = transform_gray(image)
            imgs.append(image)
    imgs = torch.cat(imgs, dim=0)
    label = 0
    label = torch.tensor(label, dtype=torch.long)
    return imgs, label

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, img_names_path, transform=None, num_samples=None):
        self.file_paths = root_dir
        self.img_names = yaml.load(open(img_names_path, 'r'), Loader=yaml.FullLoader)
        self.transform = transform
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        imgs = []
        img_name = self.img_names[idx]
        imgsdim, label = read_img(self.file_paths, img_name)

        return imgsdim, label, img_name  # 返回图像和对应的标签
# 数据目录

def pre1():
    file_paths_list = [
        ['/NFS/mfeng/zhiminhu/syj_water/syj_train/img_bing/water_pan_TP_31/', 0],
        ['/NFS/mfeng/zhiminhu/syj_water/syj_train/img_slope/water_pan_TP_31/', 1]
    ]
    train_names_path = '/NFS/mfeng/zhiminhu/syj_water/trian_code/31.yaml'

    image_dataset = CustomImageDataset(file_paths_list, train_names_path, data_transforms2)
    test_loader = DataLoader(dataset=image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    data = pd.DataFrame(columns=['id_pre', 'type_pre'])
    for inputs, labels, image_path in tqdm(test_loader, desc='Processing images', unit='batch'):
        # labels = labels.to(torch.device('cpu'))
        model.eval()
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            # class_names = ['Lake', 'River', 'shadow']
            class_names = ['lake', 'nonwater', 'river']
            for i, path in enumerate(image_path):
                image_path_name = os.path.basename(path)  # 获取文件名
                match = re.search(r'_(\d+)\.tif$', image_path_name)
                # match = re.search(r'reg_(\d+)\.tif', image_path_name)  # 匹配数字部分
                if image_path_name:
                    # last_numeric_value = match.group(1)  # 获取匹配到的数字
                    predicted_class_name = class_names[predicted_class[i].item()]

                    # 添加数据到 DataFrame
                    data = pd.concat(
                        [data, pd.DataFrame({'id_pre': [match.group(1)], 'type_pre': [predicted_class_name]})],
                        ignore_index=True)

                    # print("Predicted Class: ", predicted_class_name)
                    # class_probabilities = probabilities[i]
                    # for j, class_prob in enumerate(class_probabilities):
                    #     print(f"Probability of {class_names[j]}: {class_prob.item()}")

    data.to_excel('/NFS/mfeng/zhiminhu/syj_water/trian_code/result_shp/31_result.xlsx', index=False)

    # # 读取 Shapefile 文件
    # shapefile_path = '/NFS/mfeng/zhiminhu/syj_water/trian_code/result_shp/31shp_true.shp'
    # gdf = gpd.read_file(shapefile_path)
    #
    # # 确保 id_pre 字段和 shapefile 中的 id 字段为同一类型
    # data['id_pre'] = data['id_pre'].astype(int)
    # gdf['id'] = gdf['FID'].astype(int)
    #
    # # 合并 DataFrame 和 GeoDataFrame
    # gdf = gdf.merge(data, left_on='id', right_on='id_pre', how='left')
    #
    # # 创建新的字段并写入类型数据
    # gdf['type_pre'] = gdf['type_pre']
    #
    # # 保存更新后的 Shapefile 文件
    # output_shapefile_path = '/NFS/mfeng/zhiminhu/syj_water/trian_code/result_shp/31shp_true_pre.shp'
    # gdf.to_file(output_shapefile_path)
    #
    # print("Shapefile updated successfully.")


def pre6():
    file_paths_list = [
        # ['/public/home/mfeng/zmhu/XIMA/SHAPE/out_all/', 2],
        ['/NFS/mfeng/zhiminhu/Himalayan/data/region06/06rgb_out/', 0],
        ['/NFS/mfeng/zhiminhu/Himalayan/data/region06/06slope_out/', 3],
        ['/NFS/mfeng/zhiminhu/Himalayan/data/region06/06ndwi_out/', 1]
    ]
    train_names_path = '/NFS/mfeng/zhiminhu/Himalayan/data/region06/region06.yaml'

    image_dataset = CustomImageDataset(file_paths_list, train_names_path, data_transforms2)
    test_loader = DataLoader(dataset=image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    data = pd.DataFrame(columns=['id_pre', 'type_pre'])
    for inputs, labels, image_path in tqdm(test_loader, desc='Processing images', unit='batch'):
        inputs = inputs.to(device)
        # labels = labels.to(torch.device('cpu'))
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            # class_names = ['Lake', 'River', 'shadow']
            class_names = ['water', 'shadow']
            for i, path in enumerate(image_path):
                image_path_name = os.path.basename(path)  # 获取文件名
                match = re.search(r'reg_(\d+)\.tif', image_path_name)  # 匹配数字部分
                if match:
                    last_numeric_value = match.group(1)  # 获取匹配到的数字
                    predicted_class_name = class_names[predicted_class[i].item()]

                    # 添加数据到 DataFrame
                    data = pd.concat(
                        [data, pd.DataFrame({'id_pre': [last_numeric_value], 'type_pre': [predicted_class_name]})],
                        ignore_index=True)

                    # print("Predicted Class: ", predicted_class_name)
                    # class_probabilities = probabilities[i]
                    # for j, class_prob in enumerate(class_probabilities):
                    #     print(f"Probability of {class_names[j]}: {class_prob.item()}")
    data.to_excel('/NFS/mfeng/zhiminhu/Himalayan/data/region06/region06_model16.xlsx', index=False)
if __name__ == '__main__':
    # 数据预处理
    data_transforms2 = {
        '3': transforms.Compose([
            transforms.Resize(size=512),
            transforms.CenterCrop(size=512),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        '1': transforms.Compose([
            transforms.Resize(size=512),
            transforms.CenterCrop(size=512),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ]),
    }

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型
    model = torch.load('/NFS/mfeng/zhiminhu/syj_water/trian_code/result_model/best_model_124.pth')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 检查可用设备
    # model = torch.nn.DataParallel(model)
    model = model.to(device)
    # model = torch.load('/public/home/mfeng/zmhu/XIMA/code/convnext/model7/best_model_12.pth', map_location=torch.device('cpu'))
    # model.to(torch.device('cpu'))
    batch_size = 20
    num_workers = 8
    pre1()
    # pre2()
    # pre3()
    # pre4()
    # pre5()
    # pre6()
