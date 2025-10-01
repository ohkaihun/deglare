import os
import numpy as np
import cv2
def parse_line(line):
    """
    解析单行数据
    :param line: 文件中的一行字符串
    :return: 包含解析数据的字典
    """
    # 按空格分割字符串
    parts = line.strip().split()

    # 提取相机内参
    fu, u0, v0, ar, s = map(float, parts[0:5])

    # 提取畸变系数
    k1, k2, p1, p2, k3 = map(float, parts[5:10])

    # 提取四元数（标量部分在前）
    quaternion = list(map(float, parts[10:14]))

    # 提取平移向量
    translation = list(map(float, parts[14:17]))

    # 提取图像尺寸
    width, height = map(int, parts[17:19])



    # 返回解析后的数据
    return {
        "camera_intrinsics": {"fu": fu, "u0": u0, "v0": v0, "ar": ar, "s": s},
        "distortion_coefficients": {"k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3},
        "quaternion": quaternion,
        "translation": translation,
        "image_size": {"width": width, "height": height}
    }


def read_file(file_path):
    """
    读取文件并解析数据
    :param file_path: 文件路径
    :return: 包含所有数据的列表
    """
    data = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith('#'):
                continue
            if line.strip():  # 跳过空行
                data.append(parse_line(line))
    return data
def gen_mask(centerx,centery,height, width,itertion,mask_idx):

    Y, X = np.indices((height, width))  # 生成每个像素的坐标
    distance = np.sqrt((X - centerx) ** 2 + (Y - centery) ** 2)  # 计算每个像素到中心的距离
    if itertion<15000:
        # if mask_idx==0:
        threshold_b=(200-(itertion)*2000/150000)
        threshold_r= (200 - (itertion)*2000/150000)
        threshold_g= (200 - (itertion)*2000/150000)
        # else:
        #     threshold_b=(100-(itertion)*900/150000)
        #     threshold_r= (100 - (itertion)*900/150000)
        #     threshold_g= (100 - (itertion)*900/150000)

    else:
        threshold_b=0
        threshold_g=0
        threshold_r=0
    mask_b = ((distance < threshold_b)).astype(np.uint8)  # 生成掩码，距离小于100的部分为1（白色），其他为0（黑色）
    mask_g = ((distance < threshold_g)).astype(np.uint8)  # 生成掩码，距离小于100的部分为1（白色），其他为0（黑色）
    mask_r = ((distance < threshold_r)).astype(np.uint8)  # 生成掩码，距离小于100的部分为1（白色），其他为0（黑色）


    # mask_rgb = np.zeros((height, width, 3), dtype=np.float32)

    # mask_rgb[mask == 1] = [255, 255, 255]  # 将掩码区域设置为白色

    # # 生成掩码
    # mask_b = ((distance <= 20) & (distance > 10)).astype(np.uint8)  # 蓝色通道掩码
    # mask_g = (distance > 10).astype(np.uint8)  # 绿色通道掩码
    # mask_r = (distance > 5).astype(np.uint8)  # 红色通道掩码
    #
    # # 创建三通道掩码
    mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mask_rgb[..., 0] = mask_r * 255  # 蓝色通道
    mask_rgb[..., 1] = mask_g * 255  # 绿色通道
    mask_rgb[..., 2] = mask_b * 255  # 红色通道
    return mask_rgb

def gen_mask_light(centerx,centery,height, width,itertion,mask_idx):

    Y, X = np.indices((height, width))  # 生成每个像素的坐标
    distance = np.sqrt((X - centerx) ** 2 + (Y - centery) ** 2)  # 计算每个像素到中心的距离
    if mask_idx==0:
        threshold_b=20
        threshold_g=8
        threshold_r=5
    elif mask_idx==1:
        threshold_b = 9
        threshold_g = 10
        threshold_r = 8
    else:
        threshold_b=8
        threshold_g=7
        threshold_r=10
    mask_b = ((distance < threshold_b)).astype(np.uint8)  # 生成掩码，距离小于100的部分为1（白色），其他为0（黑色）
    mask_g = ((distance < threshold_g)).astype(np.uint8)  # 生成掩码，距离小于100的部分为1（白色），其他为0（黑色）
    mask_r = ((distance < threshold_r)).astype(np.uint8)  # 生成掩码，距离小于100的部分为1（白色），其他为0（黑色）


    # mask_rgb = np.zeros((height, width, 3), dtype=np.float32)

    # mask_rgb[mask == 1] = [255, 255, 255]  # 将掩码区域设置为白色

    # # 生成掩码
    # mask_b = ((distance <= 20) & (distance > 10)).astype(np.uint8)  # 蓝色通道掩码
    # mask_g = (distance > 10).astype(np.uint8)  # 绿色通道掩码
    # mask_r = (distance > 5).astype(np.uint8)  # 红色通道掩码
    #
    # # 创建三通道掩码
    mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mask_rgb[..., 0] = mask_r * 255  # 蓝色通道
    mask_rgb[..., 1] = mask_g * 255  # 绿色通道
    mask_rgb[..., 2] = mask_b * 255  # 红色通道
    return mask_rgb
# if __name__ == "__main__":
#     # 示例：读取文件
#     file_path = r"C:\Users\Mayn\work\2dgs_water\ray\real_scene\cap_0305_color\opencv_ba.txt"  # 替换为你的文件路径
#     parsed_data = read_file(file_path)
#
#     # 打印解析后的数据
#     for entry in parsed_data:
#         print(entry)