import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
# from torchvision.models import resnet101
# from torchvision.models import resnet50
from ConvNeXt6Channel_Model import convnext_base, convnext_small, convnext_tiny
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import os
import random
import re
import yaml

# 定义超参数
batch_size = 8
learning_rate = 0.001
num_epochs = 64
num_classes = 3
num_workers = 8
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
# 读取 Excel 文件
df = pd.read_excel('/NFS/mfeng/zhiminhu/syj_water/trian_code/water_pan_TP_31_attribute_syj_true.xlsx')
df['FID_1'] = df['FID'].astype(str)
# 将 FID 和 type 创建成一个字典
fid_type_dict = dict(zip(df['FID_1'], df['type']))
# 提取 img_name 中的数字作为 FID
def get_fid_from_img_name(img_name):
    match = re.search(r'_(\d+)\.tif$', img_name)

    if match:
        return match.group(1)  # 返回匹配的结果

    return None
label_mapping = {
    'lake': 0,
    'nonwater': 1,
    'river': 2
}
def assign_label(img_name):
    fid = get_fid_from_img_name(img_name)
    if fid is not None and fid in fid_type_dict:
        label_str = fid_type_dict[fid]
        if label_str in label_mapping:
            return label_mapping[label_str]
    return None
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
            # print("----------------------")

            # print("img_path:")
            # print(img_path)
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 使用OpenCV加载图像
            # print("image:")
            # print(image)
            image = cv2.resize(image, (512, 512))
            image = Image.fromarray(image)
            image = transform_rgb(image)
            imgs.append(image)
        elif img_dir[1] == 1:  # slope
            image = cv2.imread(img_path, 2)  # 使用OpenCV加载图像
            image[np.isnan(image)] = 0
            image[np.isinf(image)] = 0
            image[image < 0] = 0

            # min_value = 0
            # max_value = 90
            # image2 = ((image.astype(np.float32) - min_value) / (max_value - min_value) * 255).astype(np.int16)
            # norm_image = cv2.normalize(image, None, alpha=min_value, beta=max_value, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image = cv2.resize(image, (512, 512))
            image = Image.fromarray(image)
            image = transform_gray(image)
            imgs.append(image)
    imgs = torch.cat(imgs, dim=0)
    label = assign_label(img_name)
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
        img_name = self.img_names[idx]
        imgsdim, label = read_img(self.file_paths, img_name)

        return imgsdim, label, img_name  # 返回图像和对应的标签

# 数据目录
train_file_paths_list = [
    ['/NFS/mfeng/zhiminhu/syj_water/syj_train/img_bing/water_pan_TP_31/', 0],
    ['/NFS/mfeng/zhiminhu/syj_water/syj_train/img_slope/water_pan_TP_31/', 1]
]

train_names_path = '/NFS/mfeng/zhiminhu/syj_water/trian_code/train_true.yaml'
val_names_path = '/NFS/mfeng/zhiminhu/syj_water/trian_code/pre_true.yaml'

# Create an instance of CustomImageDataset with the list of file paths
image_dataset = CustomImageDataset(train_file_paths_list, train_names_path, data_transforms2)
val_dataset = CustomImageDataset(train_file_paths_list, val_names_path, data_transforms2)

num_channels = 5
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'

device_ids = [0,1,2,3,4]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 如果没有可用GPU，则使用CPU
model = convnext_base(num_classes, num_channels)
model = torch.nn.DataParallel(model)
model = model.to(device)

# 定义损失函数和优化器
# 统计每个类别的样本数
# 样本数量
# 读取 train_names_path 中的 YAML 文件
with open(train_names_path, 'r') as file:
    train_data = yaml.load(file, Loader=yaml.FullLoader)

# 创建一个字典用于统计每个类别的样本数
class_count = {'lake': 0, 'nonwater': 0, 'river': 0}

# 遍历每个文件名，提取对应的类别属性，并统计每个类别的样本数
for filename in train_data:
    fid = get_fid_from_img_name(filename)
    if fid in fid_type_dict:
        label = fid_type_dict[fid]
        if label in class_count:
            class_count[label] += 1

# 输出每个类别的样本数量
for label, count in class_count.items():
    if label == 'lake':
        count_lake = count
    if label == 'nonwater':
        count_shadow = count
    if label == 'river':
        count_river = count
# 计算每个类别的权重

total_samples = count_lake + count_shadow + count_river

weight_lake = round(1 / (count_lake / total_samples))
weight_shadow = round(1 / (count_shadow / total_samples))
weight_river = round(1 / (count_river / total_samples))


# 将权重组合成一个权重列表
class_weights = torch.Tensor([weight_lake, weight_shadow, weight_river])
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)


best_acc = 0.0
best_test_acc = 0.0

results_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])

for epoch in range(num_epochs):
    # 创建新的数据加载器，以确保 idx 从零开始
    train_loader = DataLoader(dataset=image_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

    print(len(image_dataset))
    print(len(val_dataset))
    running_loss = 0.0
    running_corrects = 0
    batch_count = 0
    print(epoch)
    model.train()
    for inputs, labels, image_path in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        print(loss.item())

        running_loss += loss.item() * inputs.size(0)
        print(running_loss)
        running_corrects += torch.sum(preds == labels.data)
        batch_count += 1

        if batch_count % 1000 == 0:
            print(
                f'Epoch [{epoch}/{num_epochs}], Batch [{batch_count}/{len(train_loader)}]')

    epoch_loss = running_loss / len(image_dataset)
    epoch_acc = running_corrects.double() / len(image_dataset)

    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 10)
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels, image_path in tqdm(val_loader, desc=f'Testing Epoch {epoch}/{num_epochs}'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)


    test_loss = running_loss / len(val_dataset)
    test_acc = running_corrects.double() / len(val_dataset)
    print('Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))
    # 记录训练和测试结果
    results_df = results_df.append({
        'Epoch': epoch + 1,
        'Train Loss': epoch_loss,
        'Train Acc': epoch_acc.item(),
        'Test Loss': test_loss,
        'Test Acc': test_acc.item()
    }, ignore_index=True)

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        model_path = os.path.join('/NFS/mfeng/zhiminhu/syj_water/trian_code/result_model/', 'best_model_{}.pth'.format(epoch + 1))
        # model_path = os.path.join(r'D:\qgs\data\data3\model2', 'best_model.pth')

        torch.save(model, model_path)
    # 将DataFrame保存为Excel文件
    results_df.to_excel('/NFS/mfeng/zhiminhu/syj_water/trian_code/result_model/training_results.xlsx', index=False)