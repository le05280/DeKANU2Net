## Language

- [English](#english)
- [中文](#中文)

---

### English

# DeKANU2Net-paper
Source code of multi-source remote sensing image wildfire monitoring based on improved DeKan-U ² - Net model
### 1-Download_By_GEE

1. **R_TB_Land8_SDown**: Used for reading a table file and downloading corresponding Landsat-8 image data in parallel while performing cropping.

2. **R_TB_Sen2_SDown**: Used for reading a table file and downloading corresponding Sentinel-2 image data in parallel while performing cropping.

   **Parameter Explanation**:

   - `csv_file_path`: Latitude and longitude date table.
   - `folder_path`: Output directory for files, checks for duplicates.

### 2-pS2Mask

1. **Out_3Bands_div_L8**: Used to obtain Landsat-8 fire point masks using the pS2Mask method.

2. **Out_3Bands_div_S2**: Used to obtain Sentinel-2 fire point masks using the pS2Mask method.

   ```
   input_folder = Input path for downloaded files
   output_tif_folder = Output path for filtered tif files
   output_img_folder = Output path for images containing wildfires
   output_mask_folder = Output path for wildfire masks
   output_tif_Mask_folder = Output path for tif images containing wildfire masks
   ```

### 3-DeKANU2Net

The original code is referenced from the following Bilibili videos:

- [https://www.bilibili.com/video/BV1yB4y1z7m](https://www.bilibili.com/video/BV1yB4y1z7m)
- [https://www.bilibili.com/video/BV1Kt4y137iS](https://www.bilibili.com/video/BV1Kt4y137iS)

**DeKANU2Net** is the model's code, where the improved model code is located at `3-DeKANU2Net/src/model_DeKANConv.py`. The original model code is located at `3-DeKANU2Net/src/model_Base.py`. You can modify `3-DeKANU2Net/src/__init__.py` to choose which model to use.

The training parameters are in `3-DeKANU2Net/train.py`, where `--data-path` is the location of the dataset.

This article has uploaded the `pS2MaskL8` dataset. You need to download `DUTS-TE.zip` and `DUTS-TR.zip` from the directory, unzip them, and move them to `3-DeKANU2Net/Datas`.

After the model training is complete, you can use `3-DeKANU2Net/predict.py` to input the validation dataset and view the model's training effect.




---

### 中文

# DeKANU2Net-paper
基于改进DeKan-U²-Net模型的多源遥感影像野火监测源代码
#### 1-Download_By_GEE

1. R_TB_Land8_SDown ：用于读取表格文件，多进程下载对应的Landsat-8影像数据的同时进行裁剪

2. R_TB_Sen2_SDown ：用于读取表格文件，多进程下载对应的Sentinel-2影像数据的同时进行裁剪

   参数解析：

   csv_file_path：经纬度日期表

   folder_path：文件的输出目录，检查是否重复

#### 2-pS2Mask

1. Out_3Bands_div_L8 ：用于使用pS2Mask方法获取Landsat-8火点掩码

2. Out_3Bands_div_S2 ：用于使用pS2Mask方法获取Sentinel-2火点掩码

   ```
   input_folder =    输入下载后的文件路径
   output_tif_folder =   输出过滤的tif文件
   output_img_folder =   输出包含野火的图像
   output_mask_folder =  输出包含野火掩码
   output_tif_Mask_folder = 输出包含野火掩码的tif图像
   ```

#### 3-DeKANU2Net

原始代码参照 up主bilibili

[https://www.bilibili.com/video/BV1yB4y1z7m](https://www.bilibili.com/video/BV1yB4y1z7m)

[https://www.bilibili.com/video/BV1Kt4y137iS](https://www.bilibili.com/video/BV1Kt4y137iS)

DeKANU2Net是模型的代码，其中改进模型代码位于 3-DeKANU2Net/src/model_DeKANConv.py
原始模型代码位于 3-DeKANU2Net/src/model_Base.py
你可以修改3-DeKANU2Net/src/__init__.py
来选择使用哪一种模型

训练参数在 3-DeKANU2Net/train.py 中，--data-path 是数据集的存放位置，

本文上传了 pS2MaskL8 的数据集，需要您从目录中将 DUTS-TE.zip 和 DUTS-TR.zip 下载至本地并解压后移动到 3-DeKANU2Net/Datas 下，

模型训练结束后您可以使用 3-DeKANU2Net/predict.py 将验证数据集输入，查看模型训练效果


