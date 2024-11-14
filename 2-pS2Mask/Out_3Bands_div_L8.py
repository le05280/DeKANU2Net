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


# Improved method - My
# Used for generating mask for Landsat8
# Normalize using layer-by-layer max-min method
def get_img_762bands(path):
    # L8: 2\3\4\5\6\7\10\p7
    ''' S2: 7, 6, 1        L8: 6, 5, 1 '''
    img = rasterio.open(path).read((6, 5, 1)).transpose((1, 2, 0))  # Read specific bands and transpose to (height, width, bands)
    max_pixel_value = np.max(img)  # Find the maximum pixel value in the image
    img = np.float32(img)  # Convert image to float32

    return img, max_pixel_value


''' Function to process a single file
input_file: Input file name
output_folder: Output directory for tif files
output_img_folder: Directory for three-band images
output_mask_folder: Directory for mask images
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
            band7 = src.read(6) / 10000  # Read band 6 and normalize
            band6 = src.read(5) / 10000  # Read band 5 and normalize
            band5 = src.read(4) / 10000  # Read band 4 and normalize
            pband = src.read(8) / 10000  # Read band 8 and normalize
            band2 = src.read(1) / 10000  # Read band 1 and normalize
            band3 = src.read(2) / 10000  # Read band 2 and normalize
            band4 = src.read(3) / 10000  # Read band 3 and normalize
            data = [band7, band6, band5, pband, band2]  # Store bands in a list
            rgbdata = [band2, band3, band4]  # Store RGB bands in a list
            profile = src.profile.copy()  # Copy the profile of the source file

        if all(band is not None for band in data):  # Check if all bands are read successfully
            num_zeros_or_less = np.sum(np.array(band2) <= 0)  # Count the number of pixels with values less than or equal to 0 in band2
            if num_zeros_or_less < 50:  # If the number of such pixels is less than 50
                if np.all(band2 > 2 * pband):  # Check if all values in band2 are greater than 2 times pband
                    print("All bands are greater than pband", os.path.basename(input_file))
                else:
                    data[1] = np.where(data[1] == 0, np.finfo(float).eps, data[1])  # Replace 0 values in band6 with a small epsilon value
                    band7_half = np.max(band7) / 3  # Calculate half of the maximum value in band7
                    band7_median = np.percentile(band7, 95)  # Calculate the 95th percentile of band7

                    part1_condition = np.logical_and(
                        data[0] / data[1] >= 0.75,  # Condition for part 1
                        (data[0] + data[1]) > (data[2] + data[4]) * 2.5
                    )

                    part2_condition = np.logical_and(
                        data[0] > 0.5,  # Condition for part 2
                        data[0] > min(band7_half, band7_median)
                    )

                    part3_condition = np.logical_and(
                        (data[0] - data[3]) > 1.5 * data[4],  # Condition for part 3
                        np.logical_and(
                            (band2 + band3 + band4) < 2 * data[0],
                            2.5 * data[3] < data[0]
                        )
                    )

                    condition_all = np.logical_and(
                        part1_condition,
                        np.logical_and(
                            part2_condition,
                            part3_condition
                        )
                    )

                    num_satisfying_pixels = np.sum(condition_all)  # Count the number of pixels satisfying all conditions

                    if num_satisfying_pixels > 200:  # If the number of satisfying pixels is greater than 200
                        output_path = os.path.join(output_folder, input_file)
                        shutil.copy2(input_path, output_path)  # Copy the input file to the output folder

                        img, max_pixel_value = get_img_762bands(input_path)  # Get the image and its maximum pixel value
                        normalized_img = np.zeros_like(img, dtype=np.uint8)  # Initialize a normalized image
                        for channel in range(img.shape[2]):  # Normalize each channel
                            min_val = np.min(img[:, :, channel])
                            max_val = np.max(img[:, :, channel])
                            normalized_img[:, :, channel] = (
                                    (img[:, :, channel] - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                        output_file = os.path.join(output_img_folder,
                                                   os.path.basename(input_file).replace('.tif', OUTPUT_IMAGE_SUFFIX))
                        cv2.imwrite(output_file, cv2.cvtColor(normalized_img, cv2.COLOR_RGB2BGR))  # Save the normalized image

                        mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)  # Initialize a mask image
                        mask_img[condition_all] = 255  # Set the mask pixels to 255 where the conditions are satisfied
                        mask_output_file = os.path.join(output_mask_folder,
                                                        os.path.basename(input_file).replace('.tif', OUTPUT_Mask_SUFFIX))
                        cv2.imwrite(mask_output_file, mask_img)  # Save the mask image
                        profile.update({'driver': 'GTiff',
                                        'dtype': rasterio.uint8,
                                        'height': img.shape[0],
                                        'width': img.shape[1],
                                        'count': 1})  # Update the profile for the mask tif file
                        out_filename_tif = os.path.join(output_tif_Mask_folder, os.path.basename(input_file))
                        if not os.path.exists(out_filename_tif):  # Check if the mask tif file already exists
                            with rasterio.open(out_filename_tif, 'w', **profile) as dst:
                                dst.write_band(1, condition_all.astype(rasterio.uint8))  # Write the mask to the tif file

        else:
            print(f"Unable to read band data for {input_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")


if __name__ == "__main__":
    input_folder = "G:/fires/China_Land8_all/China_Land8_pB7"
    output_tif_folder = "G:/fires/China_Land8_all/Images/pS2China_Lan8_filter"
    output_img_folder = "G:/fires/China_Land8_all/Images/pS2JPGs"
    output_mask_folder = "G:/fires/China_Land8_all/Images/pS2Masks"
    output_tif_Mask_folder = "G:/fires/China_Land8_all/Images/pS2TifMasks"

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

    params = {
        'processes': 16  # Set the number of processes
    }
    data = os.listdir(input_folder)

    file_infos = [(file, input_folder, output_tif_folder, output_img_folder, output_mask_folder, output_tif_Mask_folder)
                  for file in input_files]

    with tqdm(total=len(data), desc='Processing files') as pbar:  # Create a progress bar
        with Pool(params['processes']) as pool:
            for _ in pool.imap_unordered(process_images, file_infos):  # Process files in parallel
                pbar.update()  # Update the progress bar