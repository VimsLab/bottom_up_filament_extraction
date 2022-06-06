from cProfile import label
import cv2
import os
import numpy as np
import glob
from skimage.color import label2rgb
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
gt_folder = '/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/eval_worm_with_ppp_metrics/gt_worms'
predicted_worms = '/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/eval_worm_with_ppp_metrics/predicted_worms'

gt_images = os.listdir(gt_folder)
predicted_images = os.listdir(predicted_worms)


thresh = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,  0.5, 0.55, 0.6, 0.65,  0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

tp = [0] * 18
fp = [0] * 18
fn = [0] * 18

metrics = np.zeros( (len(predicted_images), len(gt_images)))
contours = []
blob = []

for i, name_pred in enumerate(predicted_images):
    image_id_bbbc = name_pred.split('_')[0]

    for ii, name_gt_image in enumerate(gt_images):
        if image_id_bbbc not in name_gt_image:
            continue
        gt_img = cv2.imread(os.path.join(gt_folder, name_gt_image),0)
        pred_img = cv2.imread(os.path.join(predicted_worms, name_pred),0)
        gt_img = 1 * (gt_img>0.5)
        pred_img = 1 * (pred_img > 0.5)
        inter = np.sum ((gt_img * pred_img > 0)) 
        overlap = np.sum (((gt_img + pred_img) > 0))
        iou = inter / overlap
        metrics[i, ii] = iou


        _, gt_img_tmp = cv2.threshold((gt_img*255.).astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
        cnt  = cv2.findContours(gt_img_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = np.zeros(gt_img_tmp.shape, dtype=np.uint8)
        outline = cv2.drawContours(outline, cnt[1], -1, (255, 255, 255))
        outline = (outline > 0 ) * 1.0
        contours.append(np.sum(outline))
        blob.append(np.sum(gt_img))
        # if iou>0.9:
        #     canvas = 1. * (gt_img>0)
        #     canvas_2 = 2. * (pred_img>0)
        #     canvas = canvas + canvas_2
        #     lable_canvas = label(canvas)

        #     mask_pred_numpy = np.zeros(lable_canvas.shape)
        #     label_show = label2rgb(lable_canvas, bg_label=0)
        #     print(inter)
        #     print(overlap)
        #     print(iou)
        #     plt.imshow(label_show)
        #     plt.show()


for i, each_tresh in enumerate(thresh):
    metrics_binary = metrics>each_tresh
    gr_bi = np.max(metrics_binary, 1)
    pr_bi = np.max(metrics_binary, 0)

    tp_curr = np.sum(gr_bi)
    fn_curr = len(gr_bi) - tp_curr
    
    fp_curr = len(pr_bi) - np.sum(pr_bi)

    tp[i] = tp[i] + tp_curr
    fn[i] = fn[i] + fn_curr
    fp[i] = fp[i] + fp_curr

print(tp)
print(fn)
print(fp)

import pdb; pdb.set_trace()

tp = np.asarray(tp)
fp = np.asarray(fp)
fn = np.asarray(fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(precision)
print(recall)

import pdb; pdb.set_trace()


