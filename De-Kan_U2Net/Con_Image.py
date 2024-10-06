import os
from PIL import Image, ImageDraw, ImageFont


# 拼接预测结果图
def concatenate_images(root_folder, output_image_path, spacing=15):
    # common_files_folder = os.path.join(root_folder, 'Common_Files')

    # 获取Common_Files文件夹中的图片文件
    common_image_files = {f for f in os.listdir(root_folder) if
                          f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))}

    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir() and f != root_folder]

    # 获取根文件夹中的图片文件
    root_image_files = [f for f in os.listdir(root_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    root_images = [Image.open(os.path.join(root_folder, img)) for img in root_image_files if img in common_image_files]

    # 获取每个子文件夹中的图片文件
    images_by_folder = []
    max_rows = len(root_images)
    for folder in subfolders:
        image_files = [f for f in os.listdir(folder) if
                       f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and f in common_image_files]
        images_by_folder.append([Image.open(os.path.join(folder, img)) for img in image_files])
        max_rows = max(max_rows, len(image_files))

    # 确保至少有一个子文件夹和一个图片
    if not images_by_folder or not max_rows:
        raise ValueError("No matching images found in the subfolders.")

    # 获取第一个图片的尺寸
    image_width, image_height = root_images[0].size if root_images else images_by_folder[0][0].size

    # 计算新的图像尺寸
    new_image_width = (len(subfolders) + 1) * image_width + (len(subfolders) + 2) * spacing
    new_image_height = max_rows * image_height + (max_rows + 1) * spacing + image_height

    # 创建一个新的空白图像
    new_image = Image.new('RGB', (new_image_width, new_image_height), (255, 255, 255))
    draw = ImageDraw.Draw(new_image)

    # 设置字体
    font_path = "arial.ttf"  # 请确保这个路径下有你想要的字体文件
    font_size = 55  # 设置你想要的字体大小
    font = ImageFont.truetype(font_path, font_size)

    # 将根文件夹的图片粘贴到新的图像中
    for j, img in enumerate(root_images):
        new_image.paste(img, (spacing, (j + 1) * (image_height + spacing)))

    # 将图片粘贴到新的图像中，并添加列标题
    for i, folder in enumerate(subfolders):
        folder_name = os.path.basename(folder)
        draw.text(((i + 1) * (image_width + spacing) + spacing, spacing), folder_name, fill="black", font=font)

        for j, img in enumerate(images_by_folder[i]):
            new_image.paste(img, ((i + 1) * (image_width + spacing) + spacing, (j + 1) * (image_height + spacing)))

    # 保存新的图像
    new_image.save(output_image_path)
    print(f"Image saved to {output_image_path}")


# 使用示例
root_folder = 'A:\Projects\All_Image_Com'
output_image_path = 'A:\Projects\Com_image.png'
concatenate_images(root_folder, output_image_path)