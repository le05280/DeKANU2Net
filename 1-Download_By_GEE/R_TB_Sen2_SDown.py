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

# Create an empty list to store longitude, latitude, and time information
coordinates_list = []
# Used to store combinations of dates and points that have already been processed to ensure uniqueness
processed_combos = []
# Used to store the names of images that have already been exported to prevent duplicate exports
exported_images = []
# Used to store processing results
results_list = []

# Read the CSV file E:\Projects\GEE\Cn-all  D:\Projects\PycharmProjects\GEE
csv_file_path = 'D:\Projects\PycharmProjects\GEE/China-17-all.csv'
df = pd.read_csv(csv_file_path)

# Set the folder path
folder_path = "G:\China_Sen_17_pB12"

# Get all file paths
file_paths = glob.glob(os.path.join(folder_path, "*"))

# Get all file names (excluding extensions)
file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths]

# Remove the last underscore and everything after it
file_names_without_suffix = [name[:name.rfind('_')] if '_' in name else name for name in file_names]

# Write the existing files in the folder to the list
exported_images = file_names_without_suffix


# Find the two closest available time points to the given time
def find_closest_available_times(target_time, coord):
    closest_times = []
    ord = []
    # Use ee.Date.parse to create an ee.Date object
    current_date = ee.Date.parse('YYYY/MM/dd', target_time)

    # Determine the current object
    now_date = current_date.advance(0, 'day')
    if image_exists(coord['longitude'], coord['latitude'], now_date) != 0:
        closest_times.append(now_date.format('YYYY/MM/dd'))
        ord.append(0)
    for i in range(1, 7):
        previous_date = current_date.advance(-i, 'day')
        next_date = current_date.advance(i, 'day')

        if image_exists(coord['longitude'], coord['latitude'], previous_date) != 0:
            closest_times.append(previous_date.format('YYYY/MM/dd'))
            ord.append(-i)
        if image_exists(coord['longitude'], coord['latitude'], next_date) != 0:
            closest_times.append(next_date.format('YYYY/MM/dd'))
            ord.append(i)
        if len(closest_times) >= 3:
            break  # Stop if two closest time points are found

    return closest_times, len(closest_times), ord


'''
COPERNICUS/S2_SR_HARMONIZED
'''


# Check if Sentinel-2 images exist for the given date
def image_exists(lon, lat, date):
    image_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(ee.Geometry.Point(lon, lat))
                        .filterDate(date, date.advance(1, 'day')))  # Select a one-day time range

    return image_collection.size().getInfo()


# In the load_closest_image function, use the description parameter to set the image name
def load_closest_image(lon, lat, date, time):
    # Pass the date parameter to ee.Date.parse to ensure it is an ee.Date object
    date = ee.Date.parse('YYYY/MM/dd', date)
    # Select only the necessary bands to avoid memory limits
    bands = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12']
    image_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .select(bands)
                        .filterBounds(ee.Geometry.Point(lon, lat))
                        .filterDate(date, date.advance(1, 'day')))  # Select a one-day time range

    # Add time information to the image properties when loading the image
    image_with_time = image_collection.set('time', time)

    # Get the original image name
    original_image_name = image_collection.first().get('system:index').getInfo()

    # Use the original image name as the layer description to ensure the original name is used when loading
    return {'eeObject': image_with_time, 'stringValue': original_image_name}


# Export the image as a GeoTIFF
def export_image(image, name):
    # Check if the image has already been exported
    if name not in exported_images:
        geometry = image.geometry()
        scale = 10  # Set the export resolution, typically higher for Sentinel-2

        ee.batch.Export.image.toDrive(
            image=image.int16(),
            description=name,
            scale=scale,
            folder='R1-500_Sentinel',
            region=geometry,
            fileFormat='GeoTIFF'
        ).start()

        # Add the name of the exported image to the list
        exported_images.append(name)
    else:
        print('Image already exported:', name)


params = {
    'count': 100,  # How many image chips to export
    'buffer': 6400,  # The buffer distance (m) around each point
    'scale': 20,  # The scale to do stratified sampling
    'seed': 1,  # A randomization seed to use for subsampling.
    'dimensions': '640x640',  # The dimension of each image chip
    'format': "GEO_TIFF",  # The output image format, can be png, jpg, ZIPPED_GEO_TIFF, GEO_TIFF, NPY
    'prefix': 'tile_',  # The filename prefix
    'processes': 24,  # How many processes to used for parallel processing
    'label_out_dir': 'G:/China_Sen_17_pB12',  # The label output directory. Default to the current working directly
    'val_out_dir': '/val',  # The val output directory. Default to the current working directly
}

from retry import retry


# Add id as the file name
@retry(tries=10, delay=1, backoff=2)
def getResult(point, image, name, id):
    # Check if the image has already been exported
    if name not in exported_images:
        nNDVI = image.normalizedDifference(['B5', 'B4'])
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
        # Originally pdate was params['prefix']
        filename = f"{out_dir}/{name}_{id}.{ext}"
        print(filename)
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print("Done: ", filename)
        # Add the name of the exported image to the list  ID and image name both duplicate as already exported
        # name = f"{name}_{id}"
        exported_images.append(name)
        return filename
    else:
        print('Image already exported:', name)
        return False


@retry(tries=10, delay=1, backoff=2)
def process_coord(coord):
    # Find the two closest time points, and their length
    closest_times, lens, ord = find_closest_available_times(coord['time'], coord)
    # If available images are found, load and export them
    if closest_times:
        date_string = coord['time']
        # Parse the input date string into a datetime object
        input_date = datetime.strptime(date_string, "%Y/%m/%d")
        # Format it into the desired string
        output_date_string = input_date.strftime("%Y%m%d")
        # Create a point
        point = ee.Geometry.Point([coord['longitude'], coord['latitude']])
        # Buffer the point (20px---640px)
        region = point.buffer(6400).bounds()

        if lens >= 3:
            result0 = load_closest_image(coord['longitude'], coord['latitude'], closest_times[0], coord['time'])
            result1 = load_closest_image(coord['longitude'], coord['latitude'], closest_times[1], coord['time'])
            result2 = load_closest_image(coord['longitude'], coord['latitude'], closest_times[2], coord['time'])
            # Add the point's time date as the file name
            original_image_name0 = f"{result0['stringValue']}_{output_date_string}"
            original_image_name1 = f"{result1['stringValue']}_{output_date_string}"
            original_image_name2 = f"{result2['stringValue']}_{output_date_string}"

            sentinel_image0 = result0['eeObject'].mosaic().clip(region)
            sentinel_image1 = result1['eeObject'].mosaic().clip(region)
            sentinel_image2 = result2['eeObject'].mosaic().clip(region)

            # All three (before, middle, after) are included, order: bac, export bc, 02
            if ord[0] > ord[1]:
                nB12 = sentinel_image0.select("B12")
                pB12 = sentinel_image1.select("B12")
                sentinel_image0 = sentinel_image0.addBands(pB12.rename('pB12'))
                sentinel_image2 = sentinel_image2.addBands(nB12.rename('nB12'))
                # Export the image as a GeoTIFF, check if it has already been exported, pass id as the file name
                getResult(point, sentinel_image0, original_image_name0, coord['id'])
                getResult(point, sentinel_image2, original_image_name2, coord['id'])

            # All three (before, middle, after) are included, order: bca, export bc, 01
            else:
                nB12 = sentinel_image0.select("B12")
                pB12 = sentinel_image2.select("B12")
                sentinel_image0 = sentinel_image0.addBands(pB12.rename('pB12'))
                sentinel_image1 = sentinel_image1.addBands(nB12.rename('nB12'))
                # Export the image as a GeoTIFF, check if it has already been exported, pass id as the file name
                getResult(point, sentinel_image0, original_image_name0, coord['id'])
                getResult(point, sentinel_image1, original_image_name1, coord['id'])

        if lens == 2:
            result0 = load_closest_image(coord['longitude'], coord['latitude'], closest_times[0], coord['time'])
            result1 = load_closest_image(coord['longitude'], coord['latitude'], closest_times[1], coord['time'])

            # Add the point's time date as the file name
            original_image_name0 = f"{result0['stringValue']}_{output_date_string}"
            original_image_name1 = f"{result1['stringValue']}_{output_date_string}"
            sentinel_image0 = result0['eeObject'].mosaic().clip(region)
            sentinel_image1 = result1['eeObject'].mosaic().clip(region)

            # All three (before, middle, after) are included, order: ba, export b, 0
            if ord[0] > ord[1]:
                pB12 = sentinel_image1.select("B12")
                sentinel_image0 = sentinel_image0.addBands(pB12.rename('pB12'))
                # Export the image as a GeoTIFF, check if it has already been exported, pass id as the file name
                getResult(point, sentinel_image0, original_image_name0, coord['id'])

            # All three (before, middle, after) are included, order: bc, export c, 1
            else:
                pB12 = sentinel_image0.select("B12")
                sentinel_image1 = sentinel_image1.addBands(pB12.rename('pB12'))
                # Export the image as a GeoTIFF, check if it has already been exported, pass id as the file name
                getResult(point, sentinel_image1, original_image_name1, coord['id'])

    else:
        # If no available images are found within a month, print a prompt
        print('No available images for the month around', coord['time'])


if __name__ == "__main__":
    # Extract longitude, latitude, and time
    for index, row in df.iterrows():
        lon = row['Lon']
        lat = row['Lat']
        time = row['Date']
        id = row['id']

        # Check if the same date and point have already been processed, only check the point and date.
        combo = f'{lon}_{lat}_{time}'
        if combo not in processed_combos:
            # Add longitude, latitude, and time information to the list
            coordinates_list.append({
                'longitude': lon,
                'latitude': lat,
                'time': time,
                'id': id
            })

        # Add the combination to the processed list
        processed_combos.append(combo)
    print(coordinates_list[0])

    logging.basicConfig()
    # Use multiprocessing.Pool to create a process pool
    with multiprocessing.Pool(params['processes']) as pool:
        # Use pool.map to process the process_coord function in parallel
        pool.map(process_coord, coordinates_list)