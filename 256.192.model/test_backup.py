import os
os.environ['DISPLAY']
import sys
import argparse
import time
import matplotlib.pyplot as plt

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import gaussian_filter
from sklearn.cluster import MeanShift, estimate_bandwidth

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import cv2
import json
import numpy as np
import imageio

from test_config import cfg

from utils.osutils import *
from utils.imutils import *
from utils.transforms import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from utils.imutils import im_to_numpy, im_to_torch
from networks import network
from dataloader.mscocoMulti import MscocoMulti
from dataloader.mscocoMulti_double_only import MscocoMulti_double_only
from dataloader.mscoco_backup_7_17 import mscoco_backup_7_17
from tqdm import tqdm

from utils.postprocess import resize_back_output_shape
from utils.postprocess import compute_heatmaps, get_keypoints, compute_end_point_heatmaps
from utils.postprocess import refine_next
from utils.postprocess import group_skels_by_tag
from utils.postprocess import split_and_refine_mid_offsets
from utils.postprocess import group_skel_by_offsets_and_tags
from utils.postprocess import convert_to_coco

from utils.postprocess import compute_key_points_belongs

from utils.color_map import GenColorMap

from utils.preprocess import visualize_offset, visualize_keypoint, visualize_skel,visualize_skel_by_offset_and_tag
# from utils.preprocess import draw_mask
#TO DO:
#Fix Line 75; Only get one image here, but there are two images in total
#            input_var = torch.autograd.Variable(inputs.cuda()[:1])
#
def normalize_include_neg_val(tag):

    # Normalised [0,255]
    normalised = 1.*(tag - np.min(tag))/np.ptp(tag).astype(np.float32)

    return normalised

def draw_mask_color(im, mask, color):
    mask = mask>0.02

    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32)

    # import pdb; pdb.set_trace()

    r[mask] = color[0]
    g[mask] = color[1]
    b[mask] = color[2]

    combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32)*0.5
    # combined = cv2.merge([r, g, b])

    return combined.astype(np.float32)

def draw_mask(im, mask):
    # import pdb; pdb.set_trace()
    mask = mask > 0.02

    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32)

    r[mask] = 1.0
    g[mask] = 0.0
    b[mask] = 0.0

    combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32) * 0.5
    # combined = im.astype(np.float32)
    return combined.astype(np.float32)


def main(args):
    show_img = True
    # show_img = False
    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained = True)
    model = torch.nn.DataParallel(model).cuda().to(device)

    # model =model.cuda()

    # img_dir = os.path.join(cur_dir,'/home/yiliu/work/fiberPJ/data/fiber_labeled_data/instance_labeled_file/')

    test_loader = torch.utils.data.DataLoader(
        MscocoMulti_double_only(cfg, train=False),
        batch_size=args.batch*args.num_gpus, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # load trainning weights
    # args.checkpoint = '//data/stromules/yiliu_code/cpn_weights/' + args.checkpoint
    checkpoint_file = os.path.join(args.checkpoint, args.test+'.pth.tar')
    print(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    # print("info : '{}'").format(checkpoint['info'])

    model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    # change to evaluation mode
    model.eval()

    print('testing...')
    full_result = []
    image_ids_for_eval = []
    with torch.no_grad():

        # for i, (inputs, targets, end_point_label, control_point_label, binary_targets, meta) in tqdm(enumerate(test_loader)):
        for i, (inputs, meta) in tqdm(enumerate(test_loader)):
            if i == 1:
                continue
                # break
            # print(i)
            # #########################################################3
            # img_path = '/home/yliu/work/data/microtubule_data/binary/0_man.png'

            # image = imageio.imread(img_path)
            # image = np.asarray(image)
            # h, w = image.shape[:2]
            
            # crop_height = h % 32
            # crop_width = w % 32
            # image = image[int(crop_height / 2) : int(h - crop_height /2),int(crop_width / 2) : int(w - crop_width /2)]
            # # image = image[130:130 + 256,135:135 + 256]
            # if len(image.shape) == 2:
            #         image = np.tile(image,(3,1,1)).transpose(1,2,0)
            #         image = np.asarray(image)
            # img = im_to_torch(image)
            # img = color_normalize(img, cfg.pixel_means)
            # inputs = img.unsqueeze(0)
            # # # ##################################################################
            input_var = torch.autograd.Variable(inputs.to(device))
            # import pdb; pdb.set_trace()
            # #############debug
            # targets = torch.autograd.Variable(targets.to(device))
            # mask_target = targets[:, 0, :, :].to(device).unsqueeze(1)
            # control_point_target = targets[:, 1, :, :].to(device).unsqueeze(1)
            # end_point_target = targets[:, 2, :, :].to(device).unsqueeze(1)
            # long_offset_target = targets[:, 3:5, :, :].to(device)
            # next_offset_target = targets[:, 5:7, :, :].to(device)
            # prev_offset_target = targets[:, 7:9, :, :].to(device)
            # control_short_offset_target = targets[:, 9:11, :, :].to(device)
            # end_short_offset_target = targets[:, 11:13, :, :].to(device)
            # end_point_label = end_point_label.to(device)
            # control_point_label = control_point_label.to(device)

            # binary_targets = [i.to(device) for i in binary_targets]

            # ground_truth = [mask_target, control_point_target, end_point_target,
            #                 long_offset_target, next_offset_target, prev_offset_target,
            #                 control_short_offset_target, end_short_offset_target,
            #                 end_point_label, control_point_label,binary_targets
            #                 ]
            #################
            input_ = input_var.data.cpu().numpy()[0]
            input_image = input_.transpose(1,2,0)
            image_show = np.transpose(input_,(1,2,0))
                    #-----------------------------------------------------------------
            #

            outputs = model(input_var)
            mask_pred, control_point_pred, end_point_pred, long_offset_pred,\
                next_offset_pred, prev_offset_pred, control_short_offset_pred, end_short_offset_pred, \
                end_point_tags, control_point_tags, binary_targets, refine_pred, dirrectional_mask= outputs

            # for ddi in range(6):
            #     dirrectional_mask_one = dirrectional_mask[0,ddi,:,:].cpu().detach().numpy()
            #     dirrectional_mask_one = (dirrectional_mask_one>0.8) * 1.0
            #     if show_img:
            #         combined_show = draw_mask(image_show, dirrectional_mask_one)
            #         cv2.imshow('dirrectional_mask pred',combined_show)
            #         cv2.waitKey(0)
            # import pdb;pdb.set_trace()

            _,_,_,binary_target_last = binary_targets
            # mask ######################################################################3:
            mask_pred_numpy = mask_pred[0,0,:,:].cpu().detach().numpy()
            # mask_pred_numpy = input_var.data.cpu().numpy()[0]
            # mask_pred_numpy = mask_pred_numpy[0,:,:]

            
            mask_pred_numpy = (mask_pred_numpy > 0.8) *1.0
            restriction_map = np.tile(mask_pred_numpy, (3,1,1)).transpose(1,2,0)
            

            canvas = np.zeros_like(mask_pred_numpy)
            mask_pred_numpy = (mask_pred_numpy>0.8) * 1.0
            if show_img:
                combined_show = draw_mask(image_show, mask_pred_numpy)
                cv2.imshow('mask pred',combined_show)
                cv2.waitKey(0)
            
            
            # control point mask#############################################################

            # # control_point_pred = binary_target_last[:, 0:1, :, :]
            # control_point_pred_numpy = control_point_pred.cpu().detach().numpy()[0,0,:,:]

            # # end point mask ################################################################
            # # end_point_pred = binary_target_last[:,1:2,:,:]
            # end_point_pred_numpy = end_point_pred.cpu().detach().numpy()[0,0,:,:]
            # end_point_pred_numpy = (end_point_pred_numpy>0.1) * 1.0
            # if show_img:
            #     combined_show = draw_mask(image_show, end_point_pred_numpy)
            #     cv2.imshow('end point mask pred',combined_show)
            #     cv2.waitKey(0)

            # control_point_tag ###############################################################
            control_point_tags = refine_pred[:,0:1,:,:]
            tags_map = refine_pred[0,0:3,:,:]
            tags_map2 = refine_pred[0,3:6,:,:]
            tags_map3 = refine_pred[0,6:9,:,:]
 
            
            cluster_embeddings = tags_map.cpu().detach().numpy()
            cluster_embeddings = cluster_embeddings.transpose((1,2,0)).reshape((-1,3))
            restriction_map_flat = restriction_map.reshape((-1,3))
            cluster_embeddings_pixels = cluster_embeddings[np.where(restriction_map_flat[:,0] == 1)]
            # bandwidth = estimate_bandwidth(cluster_embeddings_pixels, quantile=0.3, n_samples=300)
            # ms = MeanShift(bandwidth=bandwidth)
            # ms.fit(cluster_embeddings_pixels)
            # labels = ms.labels_
            # labels_unique = np.unique(labels)
            # n_clusters_ = len(labels_unique)

            tags_map = tags_map.permute((1,2,0)).cpu().detach().numpy()
            tags_map2 = tags_map2.permute((1,2,0)).cpu().detach().numpy()
            tags_map3 = tags_map3.permute((1,2,0)).cpu().detach().numpy()


            def smooth_emb(emb, radius):
                from scipy import ndimage
                from skimage.morphology import disk
                emb = emb.copy()
                w = disk(radius)/np.sum(disk(radius))
                for i in range(emb.shape[-1]):
                    emb[:, :, i] = ndimage.convolve(emb[:, :, i], w, mode='reflect')
                emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
                # import pdb; pdb.set_trace()
                return emb

            tags_map=smooth_emb(tags_map, 5)
            tags_map2=smooth_emb(tags_map2, 5)
            tags_map3=smooth_emb(tags_map3, 5)

            tags_map[np.where((1-restriction_map)>0)] = 0
            tags_map2[np.where((1-restriction_map)>0)] = 0
            tags_map3[np.where((1-restriction_map)>0)] = 0

            visual_tags_map= normalize_include_neg_val(tags_map)
            visual_tags_map2= normalize_include_neg_val(tags_map2)
            visual_tags_map3= normalize_include_neg_val(tags_map3)
            cv2.imshow('visual_tags_map',visual_tags_map)
            cv2.imshow('visual_tags_map2',visual_tags_map2)
            cv2.imshow('visual_tags_map3',visual_tags_map3)
            cv2.waitKey(0)
            continue
            import pdb; pdb.set_trace()
            control_point_tags_numpy = control_point_tags.cpu().detach().numpy()[0,0,:,:]

            # end point tag ######################################################################
            end_point_tags = refine_pred[:,1:2,:,:]
            end_point_tags_numpy = end_point_tags.cpu().detach().numpy()[0,0,:,:]

            # short offset control ###############################################################
            short_offset = control_short_offset_pred.cpu().detach().numpy()[0]
            short_offset_h = control_short_offset_pred[:,0,:,:].cpu().detach().numpy()[0]
            short_offset_v = control_short_offset_pred[:,1,:,:].cpu().detach().numpy()[0]

            if show_img:
                canvas = np.zeros_like(short_offset_h)
                canvas = visualize_offset(canvas, short_offset_h, short_offset_v)
                combined_show = draw_mask(image_show, canvas)
                cv2.imshow('short offsets control',combined_show)
                cv2.waitKey(0)

            #short offset end ######################################################################

            short_offset_end = end_short_offset_pred.cpu().detach().numpy()[0]
            short_offset_h_end = end_short_offset_pred[:,0,:,:].cpu().detach().numpy()[0]
            short_offset_v_end = end_short_offset_pred[:,1,:,:].cpu().detach().numpy()[0]
            if show_img:
                canvas = np.zeros_like(short_offset_h_end)
                canvas = visualize_offset(canvas, short_offset_h_end, short_offset_v_end)
                combined_show = draw_mask(image_show, canvas)
                cv2.imshow('short offsets end',combined_show)
                cv2.waitKey(0)

            ##### offset next #######################################################################

            next_offset = next_offset_pred.cpu().detach().numpy()[0]
            next_offset_h = next_offset_pred[:,0,:,:].cpu().detach().numpy()[0]
            next_offset_v = next_offset_pred[:,1,:,:].cpu().detach().numpy()[0]
            next_offset_tmp = next_offset
            offset_refine = split_and_refine_mid_offsets(next_offset, short_offset)

            if show_img:
                canvas = np.zeros_like(next_offset_h)
                canvas = visualize_offset(canvas, next_offset_h, next_offset_v)
                combined_show = draw_mask(image_show, canvas)
                cv2.imshow('offset next',combined_show)
                cv2.waitKey(0)



            # canvas = np.zeros_like(next_offset_h)
            # canvas = visualize_offset(canvas, offset_refine[0,:,:], offset_refine[1,:,:])
            # combined_show = draw_mask(image_show, canvas)
            # cv2.imshow('offset next refine',combined_show)
            # cv2.waitKey(0)

            ##### offset prev #########################################################################
            prev_offset = prev_offset_pred.cpu().detach().numpy()[0]
            prev_offset_h = prev_offset_pred[:,0,:,:].cpu().detach().numpy()[0]
            prev_offset_v = prev_offset_pred[:,1,:,:].cpu().detach().numpy()[0]

            if show_img:
                canvas = np.zeros_like(prev_offset_h)
                canvas = visualize_offset(canvas, prev_offset_h, prev_offset_v)
                combined_show = draw_mask(image_show, canvas)
                cv2.imshow('offset prev',combined_show)
                cv2.waitKey(0)

            ##### get keypoints control
            offset_refine_next = split_and_refine_mid_offsets(next_offset, short_offset)
            offset_refine_prev = split_and_refine_mid_offsets(prev_offset, short_offset)
            # heatmap_control = compute_heatmaps(control_point_pred_numpy, short_offset)
            # heatmap_control = compute_heatmaps((mask_pred_numpy>0.0) * 1.0, short_offset)
            # offsets_for_control_point_next = [short_offset, offset_refine_next, offset_refine_prev]
            offsets_for_control_point_prev = [short_offset, offset_refine_prev]
            offsets_for_control_point_next = [short_offset, offset_refine_next]


            heatmap_control_prev = compute_end_point_heatmaps((mask_pred_numpy>0.2) * 1.0, offsets_for_control_point_prev)
            heatmap_control_next = compute_end_point_heatmaps((mask_pred_numpy>0.2) * 1.0, offsets_for_control_point_next)
            heatmap_control_prev = gaussian_filter(heatmap_control_prev, sigma=5)
            heatmap_control_next = gaussian_filter(heatmap_control_next, sigma=5)
            heatmap_control = heatmap_control_prev + heatmap_control_next

            heatmap_control = compute_heatmaps((mask_pred_numpy>0.2) * 1.0, short_offset)
            heatmap_control = gaussian_filter(heatmap_control, sigma=5)
            kp_control = get_keypoints(heatmap_control, control_point_tags_numpy, cfg.PEAK_THRESH * 0.5 )

            combined_show=np.zeros_like(heatmap_control)
            if show_img:
                canvas = np.zeros_like(heatmap_control)
                canvas = visualize_keypoint(canvas, kp_control)
                combined_show = draw_mask(image_show, canvas)
                cv2.imshow('kp_control_keypoint', combined_show)
                cv2.waitKey(0)
            input_image_debug = combined_show

            control_point_mask = (mask_pred_numpy>0.2) * 1.0
            control_point_tags_masked = control_point_tags_numpy * control_point_mask

            normalised_tags = normalize_include_neg_val(control_point_tags_masked)

            if show_img:

                cmap = plt.get_cmap('jet')
                rgba_img = cmap(normalised_tags)
                cv2.imshow('rgba_img_control',rgba_img)
                cv2.waitKey(0)

            control_point_mask = (mask_pred_numpy>0.2) * 1.0
            end_point_tags_masked = end_point_tags_numpy * control_point_mask

            normalised_tags = normalize_include_neg_val(end_point_tags_masked)

            if show_img:
                cmap = plt.get_cmap('jet')
                rgba_img = cmap(normalised_tags)
                cv2.imshow('rgba_img_end',rgba_img)
                cv2.waitKey(0)
            # skels = group_skels_by_tag(kp_control, 0.5)
            # image_skel = visualize_skel(input_image, skels )
            # cv2.imshow('skel', image_skel)

            # import pdb;pdb.set_trace()
            ##### get keypoints end ####################################################################################
            offset_refine_next = split_and_refine_mid_offsets(next_offset, short_offset)
            offset_refine_prev = split_and_refine_mid_offsets(prev_offset, short_offset)
            # offsets_for_endpoint_point_prev = [short_offset_end]#, offset_refine_prev]
            # offsets_for_endpoint_point_next = [short_offset_end]#, offset_refine_next]

            # heatmap_end_prev = compute_end_point_heatmaps((end_point_pred_numpy>0.3) * 1.0, offsets_for_endpoint_point_prev)
            heatmap_end = compute_end_point_heatmaps((end_point_pred_numpy>0.1) * 1.0, [short_offset_end])
            heatmap_end = gaussian_filter(heatmap_end, sigma=5)

            # cv2.imshow('heatmap_end', heatmap_end * 255.)
            # cv2.waitKey(0)
            kp_end = get_keypoints(heatmap_end, end_point_tags_numpy, cfg.PEAK_THRESH * 1)

            if show_img:
                canvas = np.zeros_like(heatmap_end)
                canvas = visualize_keypoint(canvas, kp_end)
                combined_show = draw_mask(image_show, canvas)
                cv2.imshow('end_point_keypoint', combined_show)
                cv2.waitKey(0)
            #######################################################################
            # grouped_skeltons = group_skel_by_offsets_and_tags(kp_end, kp_control, offset_refine_next, offset_refine_prev)
            offset_refine_short = split_and_refine_mid_offsets(short_offset, short_offset)
            seg_mask = (mask_pred_numpy> 0.2) * 1


            grouped_skeltons, keypoints_valid = group_skel_by_offsets_and_tags(kp_end, kp_control, next_offset, prev_offset, offset_refine_short, input_image_debug)
            kp_control, kp_end = compute_key_points_belongs(keypoints_valid, kp_end, seg_mask, offset_refine_short)


            ################
            if show_img:
                visualize_grouped_skeltons, canvas_instance = visualize_skel_by_offset_and_tag(image_show, grouped_skeltons, kp_control, kp_end)


            if show_img:
                canvas = np.zeros_like(offset_refine_short[0,:,:])
                canvas = visualize_offset(canvas, offset_refine_short[0,:,:], offset_refine_short[1,:,:])
                combined_show = draw_mask(image_show, canvas)
                cv2.imshow('offset short refine',combined_show)
                cv2.waitKey(0)


            # masks = get_instance_masks(grouped_skeltons, mask_pred_numpy, offset_refine_next, offset_refine_prev, True)

            if show_img:
                cv2.imshow('new skel', visualize_grouped_skeltons)
                cv2.waitKey(0)
                cv2.imshow('new skel', canvas_instance)
                cv2.waitKey(0)

            sing_image_result, one_image_id = convert_to_coco(image_show, grouped_skeltons, kp_control, kp_end, meta)
            full_result.extend(sing_image_result)
            image_ids_for_eval.append(one_image_id)
            ###########################################################################################################

    result_path = args.result
    if not isdir(result_path):
        mkdir_p(result_path)
    result_file = os.path.join(result_path, 'result.json')
    with open(result_file,'w') as wf:
        json.dump(full_result, wf)

    # evaluate on COCO
    eval_gt = COCO(cfg.ori_gt_path)
    eval_dt = eval_gt.loadRes(result_file)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='segm')
    # cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')
    cocoEval.params.imgIds = image_ids_for_eval
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # evaluate on COCO
    eval_gt = COCO(cfg.ori_gt_path)
    eval_dt = eval_gt.loadRes(result_file)
    # cocoEval = COCOeval(eval_gt, eval_dt, iouType='segm')
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')
    cocoEval.params.imgIds = image_ids_for_eval
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Test')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint)')
    parser.add_argument('-f', '--flip', default=True, type=bool,
                        help='flip input image during test (default: True)')
    parser.add_argument('-b', '--batch', default=1, type=int,
                        help='test batch size (default: 32)')
    parser.add_argument('-t', '--test', default='CPN256x192', type=str,
                        help='using which checkpoint to be tested (default: CPN256x192')
    parser.add_argument('-r', '--result', default='result', type=str,
                        help='path to save save result (default: result)')
    main(parser.parse_args())
