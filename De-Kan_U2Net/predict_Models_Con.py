import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
from PIL import Image

from src import u2net_full


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    models_folder_path = "A:/Projects/A_Models/Original"
    img_folder_path = "A:/Projects/test_Images/L8/predict/Kumar-Roy/Images"  # 修改为你的图片文件夹路径
    threshold = 0.5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(480),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    all_predictions = []

    for model_folder in os.listdir(models_folder_path):
        model_folder_path = os.path.join(models_folder_path, model_folder)
        if not os.path.isdir(model_folder_path):
            continue
        clas_model = model_folder.split('save_weights_')[-1]
        weights_path = os.path.join(model_folder_path, "model_best.pth")

        model = u2net_full()
        weights = torch.load(weights_path, map_location='cpu')

        if "model" in weights:
            model.load_state_dict(weights["model"], strict=False)
        else:
            model.load_state_dict(weights)
        model.to(device)
        model.eval()

        # 创建预测结果文件夹
        predicts_folder = os.path.join(img_folder_path, clas_model + '_predicts')
        if not os.path.exists(predicts_folder):
            os.makedirs(predicts_folder)

        # 遍历文件夹中的所有图片
        for img_name in os.listdir(img_folder_path):
            img_path = os.path.join(img_folder_path, img_name)
            if not os.path.isfile(img_path) or not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue

            origin_img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            h, w = origin_img.shape[:2]
            img = data_transform(origin_img)
            img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

            with torch.no_grad():
                # init model
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

                t_start = time_synchronized()
                pred = model(img)
                t_end = time_synchronized()
                print("inference time: {}".format(t_end - t_start))
                pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]

                pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                pred_mask = np.where(pred > threshold, 1, 0)
                origin_img = np.array(origin_img, dtype=np.uint8)
                seg_img = origin_img * pred_mask[..., None]

                # 确保 seg_img 的数据类型为 uint8
                seg_img = seg_img.astype(np.uint8)

                # 保存预测结果
                pred_img_path = os.path.join(predicts_folder, os.path.basename(img_path))
                cv2.imwrite(pred_img_path, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))

                # 将预测结果添加到列表中
                all_predictions.append(Image.fromarray(seg_img))

    # 拼接所有预测结果
    num_models = len(os.listdir(models_folder_path))
    num_images = len(os.listdir(img_folder_path))
    images_per_model = len(all_predictions) // num_models

    total_width = max(img.width for img in all_predictions) * num_images
    total_height = max(img.height for img in all_predictions) * num_models

    new_im = Image.new('RGB', (total_width, total_height))

    y_offset = 0
    for i in range(0, len(all_predictions), num_images):
        x_offset = 0
        for j in range(i, i + num_images):
            img = all_predictions[j]
            new_im.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += max(img.height for img in all_predictions[i:i + num_images])

    new_im.save('comparison.jpg')
    new_im.show()


if __name__ == '__main__':
    main()
