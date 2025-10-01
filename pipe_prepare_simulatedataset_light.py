import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2
import os


list_dir=(r"data/curve_fit")
image_dir=os.path.join(list_dir,'images')

profilelist=['r.txt','g.txt','b.txt']
rgb_list=['red','green','blue']
bulb_list=['b','g','r']

def hyper_laplace_1d_with_alpha(x, amplitude, x0, sigma, alpha, offset):
    r = np.abs(x - x0)
    g = offset + amplitude * np.exp(-((r)  ** alpha/sigma))
    return g
def hyper_laplace_2d_with_alpha(xy, amplitude, x0, y0, sigma, alpha, offset):
    x, y = xy
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    g =  amplitude * np.exp(-(abs(r)** alpha/ sigma ) )
    return g

i=0
for img_name in os.listdir(image_dir):
    i+=1
    # if(i==7):
    #     break
    src = cv2.imread(os.path.join(image_dir,img_name))
    height, width = src.shape[:2]
    idx=int(img_name[-6:-4])
    profile_dir = os.path.join(list_dir, 'light_curve_simulate')
    os.makedirs(os.path.join(list_dir,'fit_image',f'{idx:02d}'),exist_ok=True)

    #inital rgb image
    rgb_image = np.zeros((height, width, 3))
    #read centers of light source
    center_path=os.path.join(list_dir,'light_curve_simulate',f'cam{idx:03d}_light_center.txt')
    centers=np.loadtxt(center_path)
    #note that simulated data only have one light source
    for light_idx in range(1):
        centery, centerx =centers #BGR
        #For each light, we need to fit a curve each channel(rgb)
        for rgb_idx in range(3):
            profile_name=img_name.split('.')[0]+f'_{profilelist[rgb_idx]}'
            profile_path=os.path.join(profile_dir,profile_name)
            with open(profile_path, 'r') as file:
                data = np.array([float(line.strip()) for line in file.readlines()])
            #Flip the data from the center to obtain a symmetric light source distribution.
            center_value = data[0]
            flipped_data = np.flip(data[1:])
            intensity_data= np.concatenate((flipped_data, data))


            # We only fit the light source distribution within the truncated region.（intensity_data == 1 is truncated by 8bit-range）
            mask = np.ones_like(intensity_data,dtype=bool)
            #find first index where intensity_data == 1
            first_index = np.where(intensity_data == 1)[0][1] if np.any(intensity_data == 1) else None
            # find last index where intensity_data == 1
            last_index = np.where(intensity_data == 1)[0][-2] if np.any(intensity_data == 1) else None
            # mask[intensity_data<1]=True
            mask[first_index:last_index ]= False
            valid_data = intensity_data[mask]
            x_data = np.linspace(1, intensity_data.size, intensity_data.size)
            valid_x = x_data[mask]

            #fit
            initial_guess = [np.max(intensity_data), (intensity_data.size+1)/2, 1, 0.6, 0.01]  # 初始参数
            bounds = (
                [0 * np.max(intensity_data), -np.inf, 0, 0, 0.009],  # 最小边界
                [1000 * np.max(intensity_data), np.inf, np.inf, 2, 0.1]  # 最大边界
            )
            params, _ = curve_fit(hyper_laplace_1d_with_alpha, valid_x, valid_data, p0=initial_guess, maxfev=300000,
                                  bounds=bounds)
            amplitude, x0, sigma, alpha, offset = params
            fit_data = hyper_laplace_1d_with_alpha(x_data, *params)
            fit_data = np.clip(fit_data, 0, 1)  # 确保强度数据在 0 到 5 之间
            #plot result

            plt.scatter(x_data, intensity_data, label='Data', color=rgb_list[rgb_idx], s=10)
            plt.plot(x_data, fit_data, label='Fitted Hyperlaplacian '+rgb_list[rgb_idx], color=rgb_list[rgb_idx])
            plt.legend()
            plt.title('1D Hyperlaplacian Fit')
            plt.xlabel('X')
            plt.ylabel('Intensity')
            plt.savefig(os.path.join(list_dir,'fit_image',f'{idx:02d}',f"cam{idx}_{bulb_list[light_idx]}_"+rgb_list[rgb_idx]+'channel'))
            plt.clf()
            print(
                f'Channel：{rgb_list[rgb_idx]},Parameters：A={amplitude:.8f}, mean={x0:.8f}, sigma={sigma:.10f}, alpha={alpha:.8f}, offset={offset:.8f}')

            x_image = np.arange(width)
            y_image = np.arange(height)
            x_image, y_image = np.meshgrid(x_image, y_image)

            params_2d = amplitude, centerx, centery, sigma, alpha, offset
            fit_data = hyper_laplace_2d_with_alpha((x_image.ravel(), y_image.ravel()), *params_2d).reshape(
                (height, width))
            rgb_image[..., rgb_idx] = fit_data  # 红色通道

        rgb_image = rgb_image.clip(0, 1)
        save_image_uint8 = (rgb_image * 255).astype(np.uint8)
        output_image = Image.fromarray(save_image_uint8, 'RGB')
        output_image.save(os.path.join(list_dir,'fit_image',img_name))



#thread2
# thread=[
#         2,3,16,  #B   R,G,B
#        1, 2, 2,    #G
#         4,5,4, #R
#
#         1, 3, 17,
#         1, 4, 4,
#         5,4, 3,
#
#         3, 5, 16,
#         2,3, 4,
#         6, 6,5,
#
#         2, 6, 20,
#         2, 5, 5,
#        7,5, 5,
#
#         3, 6, 16,
#         2,5,5,
#         6, 5, 4,
#
#         2, 6,16,
#         3, 4, 4,
#         6, 5, 4,
#         ]