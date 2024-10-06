import os
import shutil
import numpy as np
import rasterio
import cv2
from tqdm import tqdm
from multiprocessing import Pool, Manager
import time

OUTPUT_IMAGE_SUFFIX = '.jpg'
OUTPUT_Mask_SUFFIX = '.png'


# 改进的方法————My
# 用于Landsat8生成mask
# 使用逐层最大最小的方式进行归一化
def get_img_762bands(path):
    # L8:2\3\4\5\6\7\10\p7
    ''' S2:7, 6, 1        L8:6, 5, 1 '''
    img = rasterio.open(path).read((6, 5, 1)).transpose((1, 2, 0))
    max_pixel_value = np.max(img)
    img = np.float32(img)

    return img, max_pixel_value


'''处理单个文件的函数
input_file:输入的文件名称
output_folder:输出tif的目录
output_img_folder:三波段图像目录
output_mask_folder:掩码的目录
'''


def process_images(file_info):
    input_file, input_folder, output_folder, output_img_folder, output_mask_folder, output_tif_Mask_folder = file_info

    input_path = os.path.join(input_folder, input_file)
    try:
        with rasterio.open(input_path) as src:
            ''' S2: 7
                    6
                    5
                    8
                    1

                L8: 6
                    5
                    4
                    8
                    1 
            '''

            band7 = src.read(6)
            band6 = src.read(5)
            band5 = src.read(4)
            pband = src.read(8)
            band2 = src.read(1)
            band3 = src.read(2)
            band4 = src.read(3)
            data = [band7, band6, band5, pband, band2]
            rgbdata = [band2, band3, band4]
            profile = src.profile.copy()

        if all(band is not None for band in data):
            # if (np.array(rgbdata) > 0).all():
            num_zeros_or_less = np.sum(np.array(band2) <= 0)
            if num_zeros_or_less < 50:
                # 检查datafilter中的每一个波段是否都大于pband
                if np.all(band2 > 2 * pband):
                    # 如果所有波段都大于pband，则跳出
                    print("All bands are greater than pband", os.path.basename(input_file))
                else:
                    #             0     1       2      3      4
                    # "B12 > B11 && B12 > (B8+B2) && B12 + B11 > (B8 + B2) * 1.4 && 1.5*pB12 < B12? 1 : 0"
                    #   0     1       0     2  3      0     1      2    3               4       0

                    # 设定大于红外波段最大值的1/2
                    band7_half = np.max(band7) / 4

                    # condition = (data[0] / data[1] >= 1.4) & (data[1] > 0) & (data[0] / data[2] >= 1.4) & (data[2] > 0) & (
                    #         data[0] >= 0.15) & (data[0] / data[3] >= 1.4)

                    # condition = ((data[0] > (data[4] + data[2])) & (data[0] + data[1]) > (
                    #         data[2] + data[4]) * 1.4 & 1.5 * data[3] < data[0])
                    # 使用 NumPy 的逐元素比较运算符和 numpy.logical_and
                    # condition = np.logical_and(
                    #     data[0] > (data[4] + data[2]),
                    #     np.logical_and(
                    #         (data[0] + data[1]) > (data[2] + data[4]) * 1.4, np.logical_and(
                    #             (data[0]) > band2_half, 1.5 * data[3] < data[0]
                    #         )
                    #     )
                    # )
                    # L8条件相对宽松一些
                    data[1] = np.where(data[1] == 0, np.finfo(float).eps, data[1])
                    # condition = np.logical_and(
                    #     data[0] / data[1] >= 0.71,
                    #     np.logical_and(
                    #         (data[0] + data[1]) > (data[2] + data[4]) * 1.8, np.logical_and(
                    #             (data[0]) > 4000,
                    #             np.logical_and(1.5 * data[3] < data[0],
                    #                            np.logical_and((data[0] - data[3]) > 1.2 * data[4],
                    #                                           np.logical_and((band2 + band3 + band4) < 4 * pband,
                    #                                                          1.2 * data[3] < data[1])))
                    #         )
                    #     )
                    # )
                    # (band2 + band3 + band4) *  2< 4 * pband:可以避免云多的影响

                    # 设定大于红外波段最大值的1/2
                    band7_half = np.max(band7) / 3
                    # band7_median = np.median(band7)
                    band7_median = np.percentile(band7, 95)
                    # Part 1
                    part1_condition = np.logical_and(
                        data[0] / data[1] >= 0.75,
                        (data[0] + data[1]) > (data[2] + data[4]) * 2.5
                    )

                    # Part 2
                    part2_condition = np.logical_and(
                        data[0] > 4000,
                        data[0] > min(band7_half, band7_median)
                    )

                    # Part 3
                    part3_condition = np.logical_and(
                        (data[0] - data[3]) > 1.5 * data[4],
                        np.logical_and(
                            (band2 + band3 + band4) < 2 * data[0],
                            2.5 * data[3] < data[0]
                        )
                    )

                    # part all
                    condition_all = np.logical_and(
                        part1_condition,
                        np.logical_and(
                            part2_condition,
                            part3_condition
                        )
                    )

                    num_satisfying_pixels = np.sum(condition_all)

                    # if num_satisfying_pixels > 20:
                    #     # print(num_satisfying_pixels)
                    #     print(num_satisfying_pixels, os.path.basename(input_file))
                    if num_satisfying_pixels > 100:
                        output_path = os.path.join(output_folder, input_file)
                        shutil.copy2(input_path, output_path)

                        img, max_pixel_value = get_img_762bands(input_path)
                        normalized_img = np.zeros_like(img, dtype=np.uint8)
                        for channel in range(img.shape[2]):
                            min_val = np.min(img[:, :, channel])
                            max_val = np.max(img[:, :, channel])
                            normalized_img[:, :, channel] = (
                                    (img[:, :, channel] - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                        output_file = os.path.join(output_img_folder,
                                                   os.path.basename(input_file).replace('.tif', OUTPUT_IMAGE_SUFFIX))
                        cv2.imwrite(output_file, cv2.cvtColor(normalized_img, cv2.COLOR_RGB2BGR))

                        mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                        mask_img[condition_all] = 255
                        mask_output_file = os.path.join(output_mask_folder,
                                                        os.path.basename(input_file).replace('.tif', OUTPUT_Mask_SUFFIX))
                        cv2.imwrite(mask_output_file, mask_img)
                        profile.update({'driver': 'GTiff',
                                        'dtype': rasterio.uint8,
                                        'height': img.shape[0],
                                        'width': img.shape[1],
                                        'count': 1})
                        out_filename_tif = os.path.join(output_tif_Mask_folder, os.path.basename(input_file))
                        if not os.path.exists(out_filename_tif):
                            with rasterio.open(out_filename_tif, 'w', **profile) as dst:
                                dst.write_band(1, condition_all.astype(rasterio.uint8))

        else:
            print(f"无法读取波段数据 for {input_file}")
    except Exception as e:
        print(f"处理 {input_file} 时发生错误: {e}")

# H:\fires\China_Sen2_all\test

'''





"'''
if __name__ == "__main__": # China_Land8_pB7
    # input_folder =           "F:/fires\China_Land8_all/Images/China_Land8_Filter"
    # Global_Lan8_all\Golbal_Land8_pB7
    input_folder =           "G:/fires\Global_Lan8_all\Golbal_Land8_pB7"
    output_tif_folder =      "G:/fires/Clobal_Lan8_all/Images/testGlobal_Lan8_filter"
    output_img_folder =      "G:/fires/Clobal_Lan8_all/Images/testJPGs"
    output_mask_folder =     "G:/fires/Clobal_Lan8_all/Images/testMasks"
    output_tif_Mask_folder = "G:/fires/Clobal_Lan8_all/Images/testTifMasks"

    if not os.path.exists(output_tif_folder):
        os.makedirs(output_tif_folder)
    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)
    if not os.path.exists(output_tif_Mask_folder):
        os.makedirs(output_tif_Mask_folder)
    print(os.path.abspath(output_mask_folder))
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    # Prepare arguments for the multiprocessing pool
    params = {
        'processes': 16  # 设置进程数量
    }
    data = os.listdir(input_folder)

    file_infos = [(file, input_folder, output_tif_folder, output_img_folder, output_mask_folder, output_tif_Mask_folder)
                  for file in
                  input_files]

    # 使用tqdm创建一个进度条
    with tqdm(total=len(data), desc='Processing files') as pbar:
        with Pool(params['processes']) as pool:
            # 使用imap_unordered替代map，因为它可能更快，并且不需要保持输出顺序
            # 注意：imap_unordered在Windows上可能不可用，你可能需要使用map或其他方法
            for _ in pool.imap_unordered(process_images, file_infos):
                pbar.update()  # 更新进度条
