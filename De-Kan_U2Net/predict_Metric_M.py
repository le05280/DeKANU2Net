import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
import pandas as pd

from src import u2net_full


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def main():
    # Original          Kan_sa
    models_folder_path = "A:/Projects/best_Models/Original"
    img_folder_path = "A:/Projects/test_Images/Images"
    mask_folder_path = "A:/Projects/test_Images/Masks"
    threshold = 0.5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(640),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

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

        predicts_folder = os.path.join(img_folder_path, clas_model + '_predicts')
        if not os.path.exists(predicts_folder):
            os.makedirs(predicts_folder)

        results = []

        for img_name in os.listdir(img_folder_path):
            img_path = os.path.join(img_folder_path, img_name)
            mask_path = os.path.join(mask_folder_path, img_name)
            if not os.path.isfile(img_path) or not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue

            origin_img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            mask_img = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
            mask_img = np.where(mask_img > 0, 1, 0)

            h, w = origin_img.shape[:2]
            img = data_transform(origin_img)
            img = torch.unsqueeze(img, 0).to(device)

            with torch.no_grad():
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

                t_start = time_synchronized()
                pred = model(img)
                t_end = time_synchronized()
                print("inference time: {}".format(t_end - t_start))
                pred = torch.squeeze(pred).to("cpu").numpy()

                pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                pred_mask = np.where(pred > threshold, 1, 0)
                origin_img = np.array(origin_img, dtype=np.uint8)
                seg_img = origin_img * pred_mask[..., None]
                plt.imshow(seg_img)
                plt.show()

                pred_img_path = os.path.join(predicts_folder, os.path.basename(img_path))
                cv2.imwrite(pred_img_path, cv2.cvtColor(seg_img.astype(np.uint8), cv2.COLOR_RGB2BGR))

                eval = Evaluator
                eval = Evaluator(num_class=2)
                eval.add_batch(mask_img, pred_mask)
                precision = eval.Precision()
                recall = eval.Recall()
                f1 = eval.F1()
                oa = eval.OA()
                iou = eval.Intersection_over_Union()
                dice = eval.Dice()
                fwiu = eval.Frequency_Weighted_Intersection_over_Union()

                results.append({
                    'image_name': img_name,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'oa': oa,
                    'iou': iou,
                    'dice': dice,
                    'fwiu': fwiu
                })

            df = pd.DataFrame(results)
            csvname = clas_model + '.csv'
            pred_csv_path = os.path.join(predicts_folder, csvname)
            df.to_csv(pred_csv_path, index=False)

if __name__ == '__main__':
    main()