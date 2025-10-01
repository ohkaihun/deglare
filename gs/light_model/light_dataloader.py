import torch
from torch.utils.data import Dataset
import os
from glob import glob
import numpy as np
from PIL import Image
class lightDataset(Dataset):
    def __init__(self, R, T, gt_path):
        """
        初始化数据集
        :param images: 图像数据 (numpy array 或 torch tensor, 形状为 [num_samples, height, width, channels])
        :param R: 相机旋转四元数 (numpy array 或 torch tensor, 形状为 [num_samples, 4])
        :param T: 相机平移向量 (numpy array 或 torch tensor, 形状为 [num_samples, 3])
        :param targets: 目标参数 (numpy array 或 torch tensor, 形状为 [num_samples, num_parameters])
        """
        super(Dataset, self).__init__()
        self.device=torch.device('cuda:0')
        self.R = torch.tensor(R, dtype=torch.float32)

        self.T = torch.tensor(T, dtype=torch.float32)/100
        self.gt_path = gt_path
        self.centers=torch.tensor(np.load(os.path.join(gt_path,'center_3light.npy')), dtype=torch.float32)

        self.device = torch.device('cuda')
        self.images_lis = sorted(glob(os.path.join(self.gt_path, '*.png')))
        # self.images_np = np.stack(
        #     [np.array(Image.open(im_name).convert("RGBA"))[:, :, 2] for im_name in self.images_lis]) / 255.0
        # self.images = torch.from_numpy(self.images_np.astype(np.float32)) #
        images_list = []

        for im_name in self.images_lis:
            # 打开图像并转换为 RGBA
            img = Image.open(im_name).convert("RGB")

            # 提取第二个通道（绿色通道）
            img_array = np.array(img) / 255.0  # 归一化到 [0, 1]

            # 将处理后的图像添加到列表中
            images_list.append(img_array)

        # 将列表转换为 PyTorch 张量，保持原始尺寸
        self.images = [torch.from_numpy(img.astype(np.float32)) for img in images_list]

        # self.width=self.images.shape[2]
        # self.height=self.images.shape[1]

    def __len__(self):
        """返回数据集的大小"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本
        :param idx: 样本索引
        :return: 图像、位置信息（R, T）、目标参数
        """

        R = self.R[idx]  # 形状为 [4]
        T = self.T[idx]   # 形状为 [3]
        center=self.centers[(idx)*3:(idx+1)*3,:]
        target = self.images[idx] # 形状为 [num_parameters]
        filename = os.path.basename(self.images_lis[idx]).split('.')[0]
        # 将图像从 [H, W, C] 转换为 [C, H, W]（PyTorch 的标准格式）


        return  R, T,center, target,filename
