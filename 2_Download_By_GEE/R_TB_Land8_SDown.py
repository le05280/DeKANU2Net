import ee
import geemap
from datetime import datetime
import logging
import multiprocessing
import os
import glob
import requests
import shutil
import pandas as pd

print('start')
ee.Initialize()

print('end')

# 创建空列表存储经纬度和时间信息
coordinates_list = []
# 用于存储已经处理的日期和点位的组合，以确保唯一性
processed_combos = []
# 用于存储已导出图像的名称，防止重复导出
exported_images = []
# 用于存储处理结果
results_list = []

# 读取CSV文件E:\Projects\GEE\Cn-all
csv_file_path = 'D:/Pycharm/GEE/with_id/China_all18+.csv'
df = pd.read_csv(csv_file_path)

# 设置文件夹路径   D:\Projects\PycharmProjects\GEE\China-Land8-all
folder_path = "E:/China_Land8_pB7"

# 获取所有文件的路径
file_paths = glob.glob(os.path.join(folder_path, "*"))

# 获取所有文件的名称（不包括后缀）
file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths]

# 去除末尾的最后一个下划线及其之后的部分
file_names_without_suffix = [name[:name.rfind('_')] if '_' in name else name for name in file_names]

# 将文件夹内已存在的文件写入列表中
exported_images = file_names_without_suffix

# 找到最接近给定时间的两个可用时间点
def find_closest_available_times(target_time, coord):
    closest_times = []
    ord = []
    # 使用 ee.Date.parse 创建 ee.Date 对象
    current_date = ee.Date.parse('YYYY/MM/dd', target_time)

    # 判断当前对象
    now_date = current_date.advance(0, 'day')
    if image_exists(coord['longitude'], coord['latitude'], now_date) != 0:
        closest_times.append(now_date.format('YYYY/MM/dd'))
        ord.append(0)
    for i in range(1, 16):
        previous_date = current_date.advance(-i, 'day')
        next_date = current_date.advance(i, 'day')

        if image_exists(coord['longitude'], coord['latitude'], previous_date) != 0:
            closest_times.append(previous_date.format('YYYY/MM/dd'))
            ord.append(-i)
        if image_exists(coord['longitude'], coord['latitude'], next_date) != 0:
            closest_times.append(next_date.format('YYYY/MM/dd'))
            ord.append(i)
        if len(closest_times) >= 3:
            break  # 找到两个最接近的时间点就停止

    return closest_times, len(closest_times), ord


'''
LANDSAT/LC08/C02/T1_L2
该数据集包含由 Landsat 8 OLI/TIRS 传感器产生的数据得出的经过大气校正的地表反射率和地表温度。
这些图像包括5个可见光和近红外(VNIR)波段和2个短波红外(SWIR)波段经过正交校正后的表面反射率处理，
以及一个热红外(TIR)波段经过正交校正后的表面温度处理。它们还包含用于计算 ST 产物的中间带，以及 QA 带。
波段名称为:SR_B*
LANDSAT/LC08/C02/T1:
具有最高可用数据质量的陆地卫星场景被置于第1层，并被认为适合进行时间序列处理分析。第1级包括1级精密地形(L1TP)处理数据，
具有良好的特征辐射测量和跨不同的陆地卫星传感器进行相互校准。第一层场景的地理注册将是一致的，
并在规定的容差范围内[ < = 12 m 平方平均数误差(RMSE)]。所有一级陆地卫星数据可以被认为是一致的和互相校准(不论传感器)在整个收集。请参阅美国地质调查局文件中的更多信息。
'''

def image_exists(lon, lat, date):
    image_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                        .filterBounds(ee.Geometry.Point(lon, lat))
                        .filterDate(date, date.advance(1, 'day')))  # 选择一天的时间范围

    return image_collection.size().getInfo()


# 在 load_closest_image 函数中，使用 description 参数设置图像名称
def load_closest_image(lon, lat, date, time):
    # 将 date 参数传递给 ee.Date.parse 以确保是 ee.Date 对象
    date = ee.Date.parse('YYYY/MM/dd', date)
    # 仅选择需要波段，避免内存超限制\\增加B10地表温度波段
    bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10']
    image_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                        .select(bands)
                        .filterBounds(ee.Geometry.Point(lon, lat))
                        .filterDate(date, date.advance(1, 'day')))  # 选择一天的时间范围

    # 在加载影像时，将时间信息添加到影像的属性中
    image_with_time = image_collection.set('time', time)

    # 获取原始影像的名称
    original_image_name = image_collection.first().get('system:index').getInfo()

    # 使用原始影像的名称作为图层的描述，确保加载时使用原始名称
    return {'eeObject': image_with_time, 'stringValue': original_image_name}


# 导出图像为 GeoTIFF
def export_image(image, name):
    # 检查图像是否已经导出
    if name not in exported_images:
        geometry = image.geometry()
        scale = 10  # 设置导出的分辨率，Sentinel-2 的分辨率通常较高

        ee.batch.Export.image.toDrive(
            image=image.int16(),
            description=name,
            scale=scale,
            folder='R1-500_Sentinel',
            region=geometry,
            fileFormat='GeoTIFF'
        ).start()

        # 将已导出图像的名称添加到列表中
        exported_images.append(name)
    else:
        print('Image already exported:', name)





params = {
    'count': 100,  # How many image chips to export
    'buffer': 9600,  # The buffer distance (m) around each point
    'scale': 20,  # The scale to do stratified sampling
    'seed': 1,  # A randomization seed to use for subsampling.
    'dimensions': '640x640',  # The dimension of each image chip
    'format': "GEO_TIFF",  # The output image format, can be png, jpg, ZIPPED_GEO_TIFF, GEO_TIFF, NPY
    'prefix': 'tile_',  # The filename prefix
    'processes': 36,  # How many processes to used for parallel processing
    'label_out_dir': 'E:/China_Land8_pB7',  # The label output directory. Default to the current working directly
    'val_out_dir': '/val',  # The val output directory. Default to the current working directly
}

from retry import retry


# 添加id作为文件名称
@retry(tries=10, delay=1, backoff=2)
def getResult(point, image, name, id):
    # 检查图像是否已经导出
    if name not in exported_images:
        nNDVI = image.normalizedDifference(['SR_B5', 'SR_B4'])
        image = image.addBands(nNDVI.rename('nndvi'))
        # point = ee.Geometry.Point(point['coordinates'])
        region = point.buffer(params['buffer']).bounds()
        if params['format'] in ['png', 'jpg']:
            url = image.getThumbURL(
                {
                    'region': region,
                    'dimensions': params['dimensions'],
                    'format': params['format'],
                }
            )
        else:
            url = image.getDownloadURL(
                {
                    'region': region,
                    'dimensions': params['dimensions'],
                    'format': params['format'],
                }
            )
        if params['format'] == "GEO_TIFF":
            ext = 'tif'
        else:
            ext = params['format']
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            r.raise_for_status()
        out_dir = os.path.abspath(params['label_out_dir'])
        print(out_dir)
        # basename = str(index).zfill(len(str(params['count'])))
        # 原本pdate为params['prefix']
        filename = f"{out_dir}/{name}_{id}.{ext}"
        print(filename)
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print("Done: ", filename)
        # 将已导出图像的名称添加到列表中  ID与影响名称都重复才作为已导出
        # name = f"{name}_{id}"
        print(name)
        exported_images.append(name)
        return filename
    else:
        print('Image already exported:', name)
        return False


@retry(tries=10, delay=1, backoff=2)
def process_coord(coord):
    # 查找最接近的两个时间点
    closest_times, lens, ord = find_closest_available_times(coord['time'], coord)

    # 如果找到了可用影像，则加载和导出
    if closest_times:
        date_string = coord['time']
        # 将输入日期字符串解析为 datetime 对象
        input_date = datetime.strptime(date_string, "%Y/%m/%d")
        # 格式化为所需的字符串
        output_date_string = input_date.strftime("%Y%m%d")
        # 创建点
        point = ee.Geometry.Point([coord['longitude'], coord['latitude']])
        # 对点做缓冲区（20px---640px）
        region = point.buffer(9600).bounds()

        if lens >= 3:
            result0 = load_closest_image(coord['longitude'], coord['latitude'], closest_times[0], coord['time'])
            result1 = load_closest_image(coord['longitude'], coord['latitude'], closest_times[1], coord['time'])
            result2 = load_closest_image(coord['longitude'], coord['latitude'], closest_times[2], coord['time'])
            # 将点的时间日期添加为文件名称
            original_image_name0 = f"{result0['stringValue']}_{output_date_string}"
            original_image_name1 = f"{result1['stringValue']}_{output_date_string}"
            original_image_name2 = f"{result2['stringValue']}_{output_date_string}"

            sentinel_image0 = result0['eeObject'].mosaic().clip(region)
            sentinel_image1 = result1['eeObject'].mosaic().clip(region)
            sentinel_image2 = result2['eeObject'].mosaic().clip(region)

            # 前中后都包含,顺序:bac,导出bc、02
            if ord[0] > ord[1]:
                nB12 = sentinel_image0.select("SR_B7")
                pB12 = sentinel_image1.select("SR_B7")
                sentinel_image0 = sentinel_image0.addBands(pB12.rename('pB7'))
                sentinel_image2 = sentinel_image2.addBands(nB12.rename('nB7'))
                # 导出图像为 GeoTIFF，同时检查是否已经导出,传入id，作为文件名称命名
                getResult(point, sentinel_image0, original_image_name0, coord['id'])
                getResult(point, sentinel_image2, original_image_name2, coord['id'])

            # 前中后都包含,顺序:bca,导出bc、01
            else:
                nB12 = sentinel_image0.select("SR_B7")
                pB12 = sentinel_image2.select("SR_B7")
                sentinel_image0 = sentinel_image0.addBands(pB12.rename('pB7'))
                sentinel_image1 = sentinel_image1.addBands(nB12.rename('nB7'))
                # 导出图像为 GeoTIFF，同时检查是否已经导出,传入id，作为文件名称命名
                getResult(point, sentinel_image0, original_image_name0, coord['id'])
                getResult(point, sentinel_image1, original_image_name1, coord['id'])

        if lens == 2:
            result0 = load_closest_image(coord['longitude'], coord['latitude'], closest_times[0], coord['time'])
            result1 = load_closest_image(coord['longitude'], coord['latitude'], closest_times[1], coord['time'])

            # 将点的时间日期添加为文件名称
            original_image_name0 = f"{result0['stringValue']}_{output_date_string}"
            original_image_name1 = f"{result1['stringValue']}_{output_date_string}"
            sentinel_image0 = result0['eeObject'].mosaic().clip(region)
            sentinel_image1 = result1['eeObject'].mosaic().clip(region)

            # 前中后都包含,顺序:ba,导出b、0
            if ord[0] > ord[1]:
                pB12 = sentinel_image1.select("SR_B7")
                sentinel_image0 = sentinel_image0.addBands(pB12.rename('pB7'))
                # 导出图像为 GeoTIFF，同时检查是否已经导出,传入id，作为文件名称命名
                getResult(point, sentinel_image0, original_image_name0, coord['id'])

            # 前中后都包含,顺序:bc,导出c、1
            else:
                pB12 = sentinel_image0.select("SR_B7")
                sentinel_image1 = sentinel_image1.addBands(pB12.rename('pB7'))
                # 导出图像为 GeoTIFF，同时检查是否已经导出,传入id，作为文件名称命名
                getResult(point, sentinel_image1, original_image_name1, coord['id'])

    else:
        # 如果一个月内都没有可用影像，则打印提示
        print('No available images for the month around', coord['time'])
        # 将未处理的数据添加到列表中
        results_list.append([coord['longitude'], coord['latitude'], coord['time'], 'No images available', None])


if __name__ == "__main__":
    # 提取经度、纬度和时间
    for index, row in df.iterrows():
        lon = row['Lon']
        lat = row['Lat']
        time = row['Date']
        id = row['id']

        # 检查是否已经处理过相同的日期和点位，判断时仅判断点位与日期。
        combo = f'{lon}_{lat}_{time}'
        if combo not in processed_combos:
            # 将经纬度和时间信息添加到列表中
            coordinates_list.append({
                'longitude': lon,
                'latitude': lat,
                'time': time,
                'id': id
            })

        # 将组合添加到已处理的列表中
        processed_combos.append(combo)
    print(coordinates_list[0])

    logging.basicConfig()
    # 使用 multiprocessing.Pool 创建进程池
    with multiprocessing.Pool(params['processes']) as pool:
        # 使用 pool.map 并行处理 process_coord 函数
        pool.map(process_coord, coordinates_list)
