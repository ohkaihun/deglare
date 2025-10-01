import numpy as np
import matplotlib.pyplot as plt
import json
import math
import os
import cv2
from PIL import Image
from tqdm import tqdm
# 读取 transforms.json 文件
with open(r'data/blender_sim/kinghead_bg/transforms.json') as f:
    transforms = json.load(f)

input_dir = r'data/blender_sim/kinghead_bg'
output_dir=r'data/render_sim/kinghead_light'
image_size = (1024, 1280)
light_position = np.array([0, 5.8,3.5])  # 点光源位置
# 球体参数
sphere_center = np.array([0, 5.8,3.5])


# light_position = np.array([0, 0,0])  # 点光源位置
# # 球体参数
# sphere_center = np.array([0, 0,0])

sphere_radius =2
dt = 0.002
K=np.array(transforms['K'])
spectrum=np.array([10,15,500])


lambda_blue = 450e-9
lambda_green = 550e-9
lambda_red = 650e-9
beta_blue = 1e-5
beta_green = beta_blue * (lambda_blue / lambda_green) ** 4
beta_red = beta_blue * (lambda_blue / lambda_red) ** 4
beta=np.array([beta_red,beta_green,beta_blue])
# beta=1e-6
K[1][1]=K[0][0]
g=0.95
# l=-math.log(2/(2+beta))/beta
render_list=[3,5,7,10,13,15,17]
def angle_between_lines(A, B):
    # 计算点积
    dot_product = A[0] * B[0] + A[1] * B[1]+A[2]*B[2]

    # 计算模长
    magnitude_A = math.sqrt(A[0] ** 2 + A[1] ** 2+ A[2] ** 2)
    magnitude_B = math.sqrt(B[0] ** 2 + B[1] ** 2+ B[2] ** 2)

    # 计算余弦值
    cos_theta = dot_product / (magnitude_A * magnitude_B)
    return cos_theta


for index, frame in tqdm(enumerate(transforms['frames']), total=len(transforms['frames']), desc="Processing frames"):
    if index in render_list:
        continue
    os.makedirs(os.path.join(output_dir,'images'),exist_ok=True)
    os.makedirs(os.path.join(output_dir,'light'),exist_ok=True)
    camera_matrix = np.array(frame['transform_matrix'])
    #test
    # camera_matrix[:3,:3]=np.eye(3)
    camera_position = camera_matrix[:3, 3]  # 提取相机位置
    # camera_position=np.array([0, 0,15])

    image = cv2.imread(os.path.join(input_dir,f'cam{index:03d}.png'), cv2.IMREAD_COLOR)  # 以 BGR 格式读取
    image=np.array(image, dtype=np.float64) / 255.0
    image_zeros = np.zeros((*image_size, 3))
    steps_np = np.zeros((*image_size, 1))


    for y in tqdm(range(image_size[0]), desc=f"Processing row {index}", leave=False):
        for x in range(image_size[1]):
            pixel_ndc = np.array([x + 0.5, y + 0.5])  # 屏幕坐标
            pixel_camera = np.array([
                (pixel_ndc[0] - K[0, 2]) / K[0, 0],
                (pixel_ndc[1] - K[1, 2]) / K[1, 1],
                -1  # 假设相机的视线深度为1
            ])
            pixel_camera /= np.linalg.norm(pixel_camera)  # 归一化

            # 使用相机矩阵转换光线方向
            ray_direction = pixel_camera[:3]
            ray_direction = camera_matrix[:3, :3]@ray_direction # 应用旋转
            # pixel_camera /= np.linalg.norm(pixel_camera)  # 归一化

            oc = camera_position - sphere_center
            a = np.dot(ray_direction, ray_direction)
            b = 2 * np.dot(oc, ray_direction)
            c = np.dot(oc, oc) - sphere_radius ** 2
            discriminant = b ** 2 - 4 * a * c

            if discriminant > 0:
                t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                t2 = (-b + np.sqrt(discriminant)) / (2 * a)
                if t1 > t2:
                    t1, t2 = t2, t1

                steps = int((t2 - t1) // dt)
                accumulated_intensity = np.zeros(3)
                steps_np[y,x]=steps
                for i in range(steps):
                    t_current = t1 + i * dt
                    point_on_ray = camera_position + t_current * ray_direction
                    d_point_on_ray = np.linalg.norm(point_on_ray - light_position)
                    cos_theta=angle_between_lines(camera_position-point_on_ray,point_on_ray-sphere_center)
                    p_theta=(1/(4*math.pi))*(1-g**2)/(1 + g**2 - 2 * g * cos_theta)**(3/2)
                    if d_point_on_ray<0.2:
                        lambda_point_on_ray=spectrum*p_theta
                        intensity_point_on_ray = np.exp(-beta * abs(t_current)) / (t_current ** 2)
                        accumulated_intensity += intensity_point_on_ray * (lambda_point_on_ray)
                    else:
                        # lambda_point_on_ray =spectrum / ((d_point_on_ray/0.2) ** 2)*(1-np.exp(-beta* abs(d_point_on_ray/0.2) ))
                        lambda_point_on_ray_extra =spectrum *p_theta* np.exp(-beta* abs(d_point_on_ray/0.2) ) / ((d_point_on_ray/0.2) ** 2)
                        intensity_point_on_ray = np.exp(-beta* abs(t_current)) / (t_current ** 2)
                        accumulated_intensity += intensity_point_on_ray * (lambda_point_on_ray_extra)

                    # intensity_point_on_ray = math.exp(-beta* abs(t_current)) / (t_current ** 2)
                    # accumulated_intensity += intensity_point_on_ray * (lambda_point_on_ray+p_theta)

                image[y, x] += accumulated_intensity
                image_zeros[y, x] += accumulated_intensity



    clip_image = np.clip(image, 0, 1)
    save_image_uint8 = (clip_image * 255).astype(np.uint8)
    output_image = Image.fromarray(save_image_uint8, 'RGB')

    light_clip_image = np.clip(image_zeros, 0, 1)
    light_save_image_uint8 = (light_clip_image * 255).astype(np.uint8)
    light_output_image = Image.fromarray(light_save_image_uint8, 'RGB')

    output_image.save(os.path.join(output_dir,'images',f'cam{index:03d}.png'))
    np.save(os.path.join(output_dir,'images',f'cam{index:03d}.npy'),image)

    light_output_image.save(os.path.join(output_dir,'light',f'cam{index:03d}.png'))
    np.save(os.path.join(output_dir,'light',f'cam{index:03d}.npy'), image_zeros)