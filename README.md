# DeKANU2Net-paper
Source code of multi-source remote sensing image wildfire monitoring based on improved DeKan-U ² - Net model

环境安装


1. R_TB_Land8_SDown ：用于读取表格文件，多进程下载对应的Landsat-8影像数据的同时进行裁剪

2. R_TB_Sen2_SDown ：用于读取表格文件，多进程下载对应的Sentinel-2影像数据的同时进行裁剪

   参数解析：

   csv_file_path：经纬度日期表

   folder_path：文件的输出目录，检查是否重复

3. Out_3Bands_div_S2 ：用于使用pS2Mask方法获取Landsat-8火点掩码

4. Out_3Bands_div_S2 ：用于使用pS2Mask方法获取Sentinel-2火点掩码

   ```
   input_folder =    输入下载后的文件路径
   output_tif_folder =   输出过滤的tif文件
   output_img_folder =   输出包含野火的图像
   output_mask_folder =  输出包含野火掩码
   output_tif_Mask_folder = 输出包含野火掩码的tif图像
   ```

原始代码参照 up主

DeKANU2Net
是模型的代码，其中改进模型代码位于 DeKANU2Net
/src/model_DeKANConv.py
原始模型代码位于 DeKANU2Net
/src/model_Base.py
你可以修改DeKANU2Net
/src/__init__.py
来选择使用哪一种模型

训练参数在 3-DeKANU2Net
/train.py 中，--data-path 是数据集的存放位置，

本文上传了 pS2MaskL8 的数据集，需要您从目录中将 DUTS-TE.zip 和 DUTS-TR.zip 下载至本地并解压后移动到 3-DeKANU2Net/Datas 下，

模型训练结束后您可以使用 3-DeKANU2Net/predict.py 将验证数据集输入，查看模型训练效果
