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

