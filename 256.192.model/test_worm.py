# from cProfile import label
from email.policy import default
from genericpath import exists
from sqlite3 import connect
from skimage.measure import label
import os
os.environ['DISPLAY']
import sys
import argparse
import time
import matplotlib.pyplot as plt

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import gaussian_filter
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.filters import threshold_multiotsu
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import grey_dilation, binary_dilation
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops, label
from collections import defaultdict
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import cv2
import json
import numpy as np
import imageio
import collections

from test_config_worm import cfg
from scipy import ndimage, misc
from skimage.color import label2rgb
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

from dataloader.mscocoMulti import MscocoMulti
from dataloader.mscocoMulti_double_only_worm import MscocoMulti_double_only_worm
# from dataloader.mscocoMulti_double_only_worm_bi import MscocoMulti_double_only_worm
from networks import network_worm
from networks import network

from dataloader.mscoco_backup_7_17 import mscoco_backup_7_17
from tqdm import tqdm
from utils.preprocess import visualize_offset, visualize_points, visualize_label_map
from utils.postprocess import resize_back_output_shape
from utils.postprocess import compute_heatmaps, get_keypoints, compute_end_point_heatmaps
from utils.postprocess import refine_next
from utils.postprocess import group_skels_by_tag
from utils.postprocess import split_and_refine_mid_offsets
from utils.postprocess import group_skel_by_offsets_and_tags
from utils.postprocess import convert_to_coco
from utils.postprocess import mask_to_pologons
from utils.postprocess import compute_key_points_belongs

from utils.color_map import GenColorMap

from utils.preprocess import visualize_offset, visualize_keypoint, visualize_skel,visualize_skel_by_offset_and_tag

# from utils.preprocess import draw_mask
#TO DO:
#Fix Line 75; Only get one image here, but there are two images in total
#            input_var = torch.autograd.Variable(inputs.cuda()[:1])
#
def normalize_min_max(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if mi is None:
        mi = np.min(x)
    if ma is None:
        ma = np.max(x)
    if dtype is not None:
        x   = x.astype(dtype, copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x, 0, 1)

    return x

class DisjointSet(object):

    def __init__(self):
        self.leader = {} # maps a member to the group's leader
        self.group = {} # maps a group leader to the group (which is a set)

    def add_single(self, a):
        leadera = self.leader.get(a)
        if leadera is None:
            self.leader[a] = a
            self.group[a] = set([a])    
                   
    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])

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

    oirien_folder = cfg.orient_folder 

    result_folder = cfg.result_folder
    intermeidate_output_folder = os.path.join(result_folder, "intermeidate_output")
    result_for_six_output_folder = os.path.join(result_folder, "result_for_six_output")
    result_for_each_instance= os.path.join(result_folder, "result_for_each_instance")
    final_result_folder = os.path.join(result_folder, "final_result")
    json_result_folder = os.path.join(result_folder, "json")

    os.makedirs(intermeidate_output_folder, exist_ok=True)
    os.makedirs(result_for_six_output_folder, exist_ok=True)
    os.makedirs(final_result_folder, exist_ok=True)
    os.makedirs(json_result_folder, exist_ok=True)
    os.makedirs(result_for_each_instance, exist_ok=True)


    result_file = os.path.join(result_folder, 'result_worm_nointer_noco.json')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network_worm.__dict__[cfg.model](cfg.output_shape, cfg.num_class, cfg.inter, pretrained = True)
    # model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, cfg.inter, pretrained = True)
    model = torch.nn.DataParallel(model).cuda().to(device)

    # model =model.cuda()

    # img_dir = os.path.join(cur_dir,'/home/yiliu/work/fiberPJ/data/fiber_labeled_data/instance_labeled_file/')

    test_loader = torch.utils.data.DataLoader(
        MscocoMulti_double_only_worm(cfg, train=False),
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
    results = []
    if True:
        with torch.no_grad():

            # for i, (inputs, targets, end_point_label, control_point_label, binary_targets, meta) in tqdm(enumerate(test_loader)):
            for i, (inputs, meta) in tqdm(enumerate(test_loader)):
                # if i == 1:
                #     continue
                    # break
                # print(i)
                #########################################################3
                # img_path = '/home/yliu/work/data/microtubule_data/binary/0_man.png'

                # image = imageio.imread(img_path)
                # image = np.asarray(image)
                # image = cv2.dilate(image, np.ones((3,3)))
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
                print(meta['w1'])
                print(meta['bi'])
                image_id_in_BBBC010 = meta['bi'][0].split('/')[-1].split('_')[0]
                image_w1 = cv2.imread(meta['w2'][0], cv2.COLOR_BGR2GRAY)
                hard_mask = 1.0 * (cv2.imread(meta['bi'][0], cv2.COLOR_BGR2GRAY) > 0 )
                hard_mask = hard_mask[:,:,0]

                image_w1 = normalize_min_max(image_w1, 0,3072)
                image_show = np.transpose(input_,(1,2,0))
                image_show = image_w1


                if len(image_show.shape) == 2:
                    image_show = np.tile(image_show,(3,1,1)).transpose(1,2,0)
                    image_show = np.asarray(image_show)
                        #-----------------------------------------------------------------
                #

                outputs = model(input_var)
                # mask_pred, refine_pred= outputs
                mask_pred, refine_pred, directional_pred, intersection_areas = outputs

                # for ddi in range(6):
                #     dirrectional_mask_one = directional_pred[0,ddi,:,:].cpu().detach().numpy()
                #     dirrectional_mask_one = (dirrectional_mask_one>0.8) * 1.0
                #     if show_img:
                #         combined_show = draw_mask(image_show, dirrectional_mask_one)
                #         cv2.imshow('dirrectional_mask pred',combined_show)
                #         cv2.waitKey(0)
                # mask ######################################################################3:
                mask_pred_numpy = mask_pred[0,0,:,:].cpu().detach().numpy()
                mask_pred_numpy_2 = mask_pred[0,1,:,:].cpu().detach().numpy()
                intersection_areas_numpy = intersection_areas[0,0,:,:].cpu().detach().numpy()
                intersection_areas_numpy2 = intersection_areas[0,1,:,:].cpu().detach().numpy() # isalnd

                ##############################
                # intersection_areas_numpy3_touching = intersection_areas[0,2,:,:].cpu().detach().numpy() # isalnd

                # intersection_areas_numpy4_endpoints = intersection_areas[0,3,:,:].cpu().detach().numpy() # isalnd

                # intersection_areas_numpy3_touching = (intersection_areas_numpy3_touching > 0.2) *1.0
                # intersection_areas_numpy4_endpoints = (intersection_areas_numpy4_endpoints > 0.5) *1.0

                # combined_show_intersection_areas_numpy3_touching = draw_mask(image_show, intersection_areas_numpy3_touching)

                # combined_show_intersection_areas_numpy4_endpoints = draw_mask(image_show, intersection_areas_numpy4_endpoints)

                # cv2.imwrite(os.path.join(demo_folder,str(i) +"combined_show_intersection_areas_numpy3_touching.png" ), combined_show_intersection_areas_numpy3_touching * 255.)
                # cv2.imwrite(os.path.join(demo_folder,str(i) +"combined_show_intersection_areas_numpy4_endpoints.png" ), combined_show_intersection_areas_numpy4_endpoints*255.)

                ####################################
                
                # intersection_areas_numpy3 = intersection_areas[0,2,:,:].cpu().detach().numpy()
                # mask_pred_numpy = input_var.data.cpu().numpy()[0]
                # mask_pred_numpy = mask_pred_numpy[0,:,:]

                
                mask_pred_numpy = (mask_pred_numpy > 0.2) *1.0
                cv2.imwrite(os.path.join("./intermeidate_output_folder/worm_bi",str(i) +"mask_pred_numpy.png" ), mask_pred_numpy * 255.)


                mask_pred_numpy_2 = (mask_pred_numpy_2 > 0.5) *1.0
                outline_name = meta['bi'][0].split('/')[-1]
                mask_pred_numpy_2 = cv2.imread(os.path.join("/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/training_data_worm_test/outlines", outline_name + '_outline_final.png'))
                mask_pred_numpy_2 = 1.0 * (mask_pred_numpy_2[:,:,0] > 0)
                mask_pred_numpy_2 = cv2.dilate(mask_pred_numpy_2, np.ones((2,2))) # outlines

                ###
                # #use four
                # mask_pred_numpy_2 = intersection_areas_numpy3_touching
                # mask_pred_numpy_2 = cv2.dilate(mask_pred_numpy_2, np.ones((3,3))) # outlines
                # mask_pred_numpy_2 = ((mask_pred_numpy_2 + intersection_areas_numpy4_endpoints) > 0) * 1.0
                # ########

                intersection_areas_numpy = (intersection_areas_numpy > 0.5) *1.0
                intersection_areas_numpy2 = (intersection_areas_numpy2 > 0.5) *1.0
                # intersection_areas_numpy3 = (intersection_areas_numpy3 > 0.7) *1.0
                restriction_map = np.tile(mask_pred_numpy, (3,1,1)).transpose(1,2,0)
                # restriction_map = np.tile(intersection_areas_numpy, (3,1,1)).transpose(1,2,0)
                
                disect_mask = ((cv2.dilate(intersection_areas_numpy,  np.ones((5,5)) ) * mask_pred_numpy - mask_pred_numpy_2 - cv2.dilate(intersection_areas_numpy2,  np.ones((3,3)) ) )>0)*1.

                canvas = np.zeros_like(mask_pred_numpy)
                mask_pred_numpy = (mask_pred_numpy>0.8) * 1.0
                intersection_areas_numpy = (intersection_areas_numpy>0.8) * 1.0

                def smooth_emb(emb, radius):
                    from scipy import ndimage
                    from skimage.morphology import disk
                    emb = emb.copy()
                    w = disk(radius)/np.sum(disk(radius))
                    # for i in range(emb.shape[-1]):
                    emb = ndimage.median_filter(emb, size=radius)
                    # emb = ndimage.maximum_filter(emb, size=9)
                    # emb[:, :] = ndimage.convolve(emb[:, :], w, mode='reflect')
                    # emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
                    # import pdb; pdb.set_trace()
                    return emb
                

                tags_map_single = refine_pred[0,0,:,:].cpu().detach().numpy()
                
                use_hard_mask = True
                if use_hard_mask:
                    tags_map_single = tags_map_single * hard_mask 
                    tags_map_single = smooth_emb(tags_map_single , 5)
                    tags_map_single[np.where(hard_mask==0)] = -100
                    visual_tags_map_single= normalize_include_neg_val((tags_map_single * hard_mask ))

                else:
                    tags_map_single = tags_map_single * mask_pred_numpy 
                    tags_map_single[np.where(mask_pred_numpy==0)] = -100
                    visual_tags_map_single= normalize_include_neg_val((tags_map_single * mask_pred_numpy ))

                # histogram, bin_edges = np.histogram(visual_tags_map_single[np.where(mask_pred_numpy>0)], bins=250, range=(0.000001, np.max(visual_tags_map_single)))  
                # # histogram, bin_edges = np.histogram(tags_map_single[np.where(mask_pred_numpy>0)], bins=250, range=( np.min(tags_map_single[np.where(mask_pred_numpy>0)]), np.max(tags_map_single)))  
                # plt.plot(bin_edges[0:-1], histogram)
                # plt.title("Grayscale Histogram")
                # plt.xlabel("grayscale value")
                # plt.ylabel("pixels")
                # plt.xlim(np.min(visual_tags_map_single[np.where(mask_pred_numpy>0)]), np.max(visual_tags_map_single))
                # plt.show()

                cmap = plt.get_cmap('jet')
                rgba_img = cmap(visual_tags_map_single)
                plt.imsave(os.path.join(intermeidate_output_folder,str(i) +"embedding_map.png" ), rgba_img)

                # rgba_img = rgba2rgb(rgba_img) * 255.
                # rgba_img = 255. * rgba_img[:,:,:3]
                # cv2.imwrite(os.path.join(demo_folder,str(i) +"embedding_map_inter.png" ), 255. * rgba_img[:,:,:3].astype('uint8'))
                r = rgba_img[:,:,0]
                g = rgba_img[:,:,1]
                b = rgba_img[:,:,2]

                r[intersection_areas_numpy==0] = 0
                r[(mask_pred_numpy - intersection_areas_numpy) == 1] = 1         
                g[intersection_areas_numpy==0] = 0
                g[(mask_pred_numpy - intersection_areas_numpy) == 1] = 1         
                b[intersection_areas_numpy==0] = 0
                b[(mask_pred_numpy - intersection_areas_numpy) == 1] = 1         

                rgba_img[:,:,0] = r 
                rgba_img[:,:,1] = g 
                rgba_img[:,:,2] = b 

                plt.imsave(os.path.join(intermeidate_output_folder,str(i) +"embedding_map_inter.png" ), rgba_img)

                # cv2.imshow('rgba_visual_tags_map',rgba_img)
                # cv2.waitKey(0)
                # if show_img:
                #     combined_show = draw_mask(image_show, mask_pred_numpy)
                #     cv2.imshow('mask pred',combined_show)
                #     cv2.waitKey(0)
                # if show_img:
                #     combined_show = draw_mask(image_show, intersection_areas_numpy)
                #     cv2.imshow('mask intersection_areas',combined_show)
                #     cv2.waitKey(0)
                # if show_img:
                #     combined_show = draw_mask(image_show, intersection_areas_numpy2)
                #     cv2.imshow('mask intersection_areas_2',combined_show)
                #     cv2.waitKey(0)
                # if show_img:
                #     combined_show = draw_mask(image_show, mask_pred_numpy_2)
                #     cv2.imshow('mask mask_pred_numpy_2',combined_show)
                #     cv2.waitKey(0)
                # if show_img:
                #     combined_show = draw_mask(image_show, disect_mask)
                #     cv2.imshow('mask disect_mask',combined_show)
                #     cv2.waitKey(0)
                intersection_labels = label(intersection_areas_numpy2)


                combined_show_mask_pred = draw_mask(image_show, mask_pred_numpy)

                combined_show_intersection_areas = draw_mask(image_show, intersection_areas_numpy)

                combined_show_intersection_areas_2 = draw_mask(image_show, intersection_areas_numpy2)

                combined_show_mask_pred_numpy_2 = draw_mask(image_show, mask_pred_numpy_2)

                combined_show_disect_mask = draw_mask(image_show, disect_mask)

                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i) +"combined_show_mask_pred.png" ), combined_show_mask_pred * 255.)
                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i) +"combined_show_intersection_areas.png" ), combined_show_intersection_areas * 255.)
                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i) +"combined_show_intersection_areas_2.png" ), combined_show_intersection_areas_2 * 255.)
                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i) +"combined_show_mask_pred_numpy_2.png" ), combined_show_mask_pred_numpy_2 * 255.)
                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i) +"combined_show_disect_mask.png" ), combined_show_disect_mask*255.)
                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i) +"input_image.png" ), image_show * 255.)

                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i) +"binary_mask.png" ), hard_mask* 255.)

                from skimage.morphology import square as mor_square
                from skimage.morphology import dilation as im_dilation

                def mask_from_seeds(embedding, seeds, mask, similarity_thres=0.7):
                    seeds = label(seeds)
                    
                    # thresholds = threshold_multiotsu(tags_map_single_chane_inter, classes = 4)
                    # regions = np.digitize(tags_map_single_chane_inter, bins=thresholds)

                    cluster_embeddings_pixels = embedding[np.where(seeds > 0)].reshape(-1,1)
                    # thresholds = threshold_multiotsu(cluster_embeddings_pixels, classes = 4)
                    # regions = np.digitize(embedding, bins=thresholds) + 1
                    # seeds[np.where(seeds > 0)] = regions[np.where(seeds > 0)]
                    import pdb; pdb.set_trace()
                    bandwidth = estimate_bandwidth(cluster_embeddings_pixels, quantile=0.15, n_samples=5000)
                    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                    ms.fit(cluster_embeddings_pixels)
                    labels = ms.labels_
                    cluster_centers = ms.cluster_centers_

                    labels_unique = np.unique(labels)
                    n_clusters_ = len(labels_unique)

                    seeds[np.where(seeds > 0)] = (labels + 1).reshape(-1)

                    print("number of estimated clusters : %d" % n_clusters_)     
                    # import pdb; pdb.set_trace()

                    
                    props = regionprops(seeds)

                    mean = {}
                    sum_total = {}
                    sum_count = {}
                    for p in props:
                        row, col = p.coords[:, 0], p.coords[:, 1]
                        emb_mean = np.mean(embedding[row, col], axis=0)
                        mean[p.label] = emb_mean
                        sum_total[p.label] = np.sum(embedding[row, col], axis=0)
                        sum_count[p.label] = len(row)
                    # import pdb; pdb.set_trace()
                    count = 0
                    while True:
                        
                        dilated = im_dilation(seeds, mor_square(5)) 

                        front_r, front_c = np.nonzero((seeds != dilated) * (mask>0))
                        # import pdb; pdb.set_trace()
                        # similarity = [np.dot(embedding[r, c], mean[dilated[r, c]])
                        #             for r, c in zip(front_r, front_c)]
                        # similarity = [np.sqrt((embedding[r, c] - mean[dilated[r, c]])**2)
                        #             for r, c in zip(front_r, front_c)]
                        similarity = [np.abs(embedding[r, c] - mean[dilated[r, c]])
                                    for r, c in zip(front_r, front_c)]
                        # import pdb; pdb.set_trace()
                        # bg = seeds[front_r, front_c] == 0
                        # add_ind = np.logical_and([s > similarity_thres for s in similarity], bg)
                        add_ind = np.array([s < similarity_thres for s in similarity])

                        if np.all(add_ind == False):
                            break
                        # import pdb; pdb.set_trace()
                        seeds[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

                        visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), seeds)
                        cv2.imshow('new_seeds_visual',visualize_img)
                        cv2.waitKey(0)

                        
                        count += 1
                        print(count)
                        # if count >=6:
                        #     break
                        # new_seeds_visual = label2rgb(seeds, image=mask_pred_numpy, bg_label=0)
                        # cv2.imshow("new_seeds_visual", new_seeds_visual)
                        # cv2.waitKey(0)
                    return seeds

                def mask_from_seeds_v3(embedding, island, mask, segments, ds, similarity_thres=0.75):

                    segments_props = regionprops(segments)
                    island_maping = defaultdict(list)
                    not_visited = set(np.unique(segments))
                    #TODO: change 10 ** 4
                    connection_area_label_record = np.zeros(segments.shape)
                    dilated_island_record = np.zeros(segments.shape)
                    for eachisland in np.unique(island):
                        if eachisland == 0:
                            continue
                        current_island = (island == eachisland) * 1.0
                        connection_areas = (cv2.dilate(current_island.astype('uint8'), np.ones((11,11))) * mask - cv2.dilate(current_island.astype('uint8'), np.ones((3,3))) > 0 ) * 1.0
                        
                        # connection_areas = (cv2.dilate(current_island.astype('uint8'), np.ones((11,11))) * mask - cv2.dilate(current_island.astype('uint8'), np.ones((1,1))) > 0 ) * 1.0
                        connection_areas_label = (connection_areas>0) * segments
                        connection_areas_props = regionprops(connection_areas_label)

                        dilated_island_record[connection_areas>0] = eachisland
                        connection_area_label_record += connection_areas_label
                        


                        mean_vals = {}
                        for p in connection_areas_props:
                            if p.label == 0:
                                continue
                            row, col = p.coords[:, 0], p.coords[:, 1]
                            emb_mean = np.mean(embedding[row, col], axis=0)
                            mean_vals[p.label] = emb_mean
                        
                        if len(mean_vals) == 0:
                            continue
                        mean_vals_numpy = np.fromiter(mean_vals.values(), dtype=float)
                        keys_numpy = np.fromiter(mean_vals.keys(), dtype=float)
                        cost_matrix = np.ones((len(mean_vals_numpy), len(mean_vals_numpy))) * 500.


                        #####################################################
                        # for ind, emb_val in enumerate(mean_vals_numpy):
                        #     for ind_2, emb_val_2 in enumerate(mean_vals_numpy):
                        #         if ind == ind_2:
                        #             continue
                        #         cost_matrix[ind][ind_2] = np.abs(emb_val - emb_val_2)
                        # row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        #####################################################

                        for ind, emb_val in enumerate(mean_vals_numpy):
                            for ind_2, emb_val_2 in enumerate(mean_vals_numpy[ind+1:]):
                                cost_matrix[ind][ind + 1 + ind_2] = np.abs(emb_val - emb_val_2)

                        cost_matrix[cost_matrix > similarity_thres] = 1000
                        assignments = np.ones(len(mean_vals_numpy)) * -1
                        while(np.min(cost_matrix) != 1000):
                            pairs = np.where(cost_matrix == np.min(cost_matrix))
                            for pair_a, pair_b in zip(pairs[0], pairs[1]):
                                assignments[pair_a] = pair_b
                                assignments[pair_b] = pair_a
                                cost_matrix[pair_a,:] = 1000
                                cost_matrix[:,pair_b] = 1000
                                cost_matrix[pair_b,:] = 1000
                                cost_matrix[:,pair_a] = 1000

                        # row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        # import pdb; pdb.set_trace()

                        col_ind = assignments.astype(int)
                        for connector_id, to_be_connected in enumerate(col_ind):
                            if to_be_connected == -1: 
                                continue
                            # within the island
                            # connect_label a 
                            connect_label_a = keys_numpy[connector_id]
                            # connect_label b 
                            connect_label_b = keys_numpy[to_be_connected]

                            #fetch segments
                            segments_a = segments[connection_areas_label==connect_label_a]
                            segments_b = segments[connection_areas_label==connect_label_b]

                            # fetch segments label
                            try:
                                segments_a_label = np.bincount(np.delete(segments_a,np.where(segments_a==0))).argmax()
                                segments_b_label = np.bincount(np.delete(segments_b,np.where(segments_b==0))).argmax()
                            except:
                                continue
                            # add island to this group
                            # segments[seeds == eachisland] = island_label
                            # connecting
                            island_maping[segments_a_label].append(eachisland)
                            island_maping[segments_b_label].append(eachisland)
                            ds.add(segments_a_label, segments_b_label)

                            
                            # mask_connecting_segments =  (((segments==segments_a_label) * 1.0) + (seeds == eachisland) * 1.0 + ((segments==segments_b_label) * 1.0) > 0) * 1.0
                            # visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), (segments + seeds) * mask_connecting_segments)
                            # cv2.imshow('connecting_segments',visualize_img)
                            # visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), segments)

                            # cv2.imshow('segments',visualize_img)
                            # visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), current_island)
                            # cv2.imshow('current_island',visualize_img)
                            # cv2.waitKey(0)                    
                    for non_visited_id in not_visited:
                        if non_visited_id == 0:
                            continue
                        ds.add_single(non_visited_id)
                    visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), connection_area_label_record)
                    cv2.imwrite(os.path.join(intermeidate_output_folder,str(i)  + "connection_area_label_record.png" ), visualize_img)
                    visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), dilated_island_record)
                    cv2.imwrite(os.path.join(intermeidate_output_folder,str(i)  + "dilated_island_record.png" ), visualize_img)
                    return island_maping, ds
                        
                        
                    # thresholds = threshold_multiotsu(tags_map_single_chane_inter, classes = 4)
                    # regions = np.digitize(tags_map_single_chane_inter, bins=thresholds)

                    cluster_embeddings_pixels = embedding[np.where(seeds > 0)].reshape(-1,1)
                    # thresholds = threshold_multiotsu(cluster_embeddings_pixels, classes = 4)
                    # regions = np.digitize(embedding, bins=thresholds) + 1
                    # seeds[np.where(seeds > 0)] = regions[np.where(seeds > 0)]
                    import pdb; pdb.set_trace()
                    bandwidth = estimate_bandwidth(cluster_embeddings_pixels, quantile=0.15, n_samples=5000)
                    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                    ms.fit(cluster_embeddings_pixels)
                    labels = ms.labels_
                    cluster_centers = ms.cluster_centers_

                    labels_unique = np.unique(labels)
                    n_clusters_ = len(labels_unique)

                    seeds[np.where(seeds > 0)] = (labels + 1).reshape(-1)

                    print("number of estimated clusters : %d" % n_clusters_)     
                    # import pdb; pdb.set_trace()

                    
                    props = regionprops(seeds)

                    mean = {}
                    sum_total = {}
                    sum_count = {}
                    for p in props:
                        row, col = p.coords[:, 0], p.coords[:, 1]
                        emb_mean = np.mean(embedding[row, col], axis=0)
                        mean[p.label] = emb_mean
                        sum_total[p.label] = np.sum(embedding[row, col], axis=0)
                        sum_count[p.label] = len(row)
                    # import pdb; pdb.set_trace()
                    count = 0
                    while True:
                        
                        dilated = im_dilation(seeds, mor_square(5)) 

                        front_r, front_c = np.nonzero((seeds != dilated) * (mask>0))
                        # import pdb; pdb.set_trace()
                        # similarity = [np.dot(embedding[r, c], mean[dilated[r, c]])
                        #             for r, c in zip(front_r, front_c)]
                        # similarity = [np.sqrt((embedding[r, c] - mean[dilated[r, c]])**2)
                        #             for r, c in zip(front_r, front_c)]
                        similarity = [np.abs(embedding[r, c] - mean[dilated[r, c]])
                                    for r, c in zip(front_r, front_c)]
                        # import pdb; pdb.set_trace()
                        # bg = seeds[front_r, front_c] == 0
                        # add_ind = np.logical_and([s > similarity_thres for s in similarity], bg)
                        add_ind = np.array([s < similarity_thres for s in similarity])

                        if np.all(add_ind == False):
                            break
                        # import pdb; pdb.set_trace()
                        seeds[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

                        visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), seeds)
                        cv2.imshow('new_seeds_visual',visualize_img)
                        cv2.waitKey(0)

                        
                        count += 1
                        print(count)
                        # if count >=6:
                        #     break
                        # new_seeds_visual = label2rgb(seeds, image=mask_pred_numpy, bg_label=0)
                        # cv2.imshow("new_seeds_visual", new_seeds_visual)
                        # cv2.waitKey(0)
                    return seeds
                
                def mask_from_seeds_v2(embedding, seeds, mask, similarity_thres=0.7):
                    seeds = label(seeds)
                    props = regionprops(seeds)

                    mean = {}
                    for p in props:
                        row, col = p.coords[:, 0], p.coords[:, 1]
                        emb_mean = np.mean(embedding[row, col], axis=0)
                        mean[p.label] = emb_mean
                    import pdb; pdb.set_trace()
                    count = 0
                    while True:
                        
                        dilated = im_dilation(seeds, mor_square(9)) 

                        front_r, front_c = np.nonzero(seeds != dilated)
                        import pdb; pdb.set_trace()
                        # similarity = [np.dot(embedding[r, c], mean[dilated[r, c]])
                        #             for r, c in zip(front_r, front_c)]
                        similarity = [np.sqrt(embedding[r, c] - mean[dilated[r, c]])
                                    for r, c in zip(front_r, front_c)]
                        import pdb; pdb.set_trace()
                        # bg = seeds[front_r, front_c] == 0
                        # add_ind = np.logical_and([s > similarity_thres for s in similarity], bg)
                        add_ind = np.array([s < similarity_thres for s in similarity])

                        if np.all(add_ind == False):
                            break

                        seeds[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]
                        new_seeds_visual = label2rgb(seeds, image=mask_pred_numpy, bg_label=0)
                        count += 1
                        print(count)
                        cv2.imshow("new_seeds_visual", new_seeds_visual)
                        cv2.waitKey(0)
                    return seeds

                


                intersection_labels_dilation = grey_dilation(intersection_labels, footprint=np.ones((11,11))) * mask_pred_numpy
                intersection_labels_dilation_real = grey_dilation(intersection_labels, footprint=np.ones((3,3))) * mask_pred_numpy
                # intersection_labels_dilation_sub = (intersection_labels_dilation>0)*1. - (intersection_labels_dilation_real>0) * 1.
                from copy import deepcopy
                intersection_labels_dilation_sub = deepcopy(intersection_labels_dilation) - intersection_labels_dilation_real

                struct2 = ndimage.generate_binary_structure(3, 3)
                # mask_pred_numpy_2 : outlines

                #getsegmentslabels
                segments = (mask_pred_numpy - binary_dilation(intersection_areas_numpy2, structure=np.ones((3,3))) - mask_pred_numpy_2 ) > 0 
                segments = (hard_mask - binary_dilation(intersection_areas_numpy2, structure=np.ones((3,3))) - mask_pred_numpy_2 ) > 0 
                # segments = (mask_pred_numpy - binary_dilation(intersection_areas_numpy2, structure=np.ones((3,3)))) > 0 
                # segments = (mask_pred_numpy - binary_dilation(intersection_areas_numpy, structure=np.ones((5,5)))) > 0 
                segments = remove_small_objects(segments, 10)
                # segments = (mask_pred_numpy - intersection_areas_numpy2 - mask_pred_numpy_2 ) > 0 
                segments_labels = label(segments)

                connection_areas = (intersection_labels_dilation_sub>0) * segments

                #########################################

                # image_intersection_labels = label2rgb(intersection_labels, image=mask_pred_numpy, bg_label=0)
                # image_intersection_labels_dilation= label2rgb(intersection_labels_dilation, image=mask_pred_numpy, bg_label=0)
                # image_labels_dilation_real = label2rgb(intersection_labels_dilation_sub, image=mask_pred_numpy, bg_label=0)
                # intersection_labels_dilation_sub_visual = label2rgb(intersection_labels_dilation_sub, image=mask_pred_numpy, bg_label=0)
                # segments_labels_visual = label2rgb(segments_labels, image=mask_pred_numpy, bg_label=0)
                # connection_areas_visual = label2rgb(connection_areas, image=mask_pred_numpy, bg_label=0)
                
                # cv2.imshow("image_intersection_labels", image_intersection_labels)
                # cv2.imshow("image_dilation", image_intersection_labels_dilation)
                # # cv2.imshow("image_real", image_labels_dilation_real)
                # cv2.imshow("inte_sub", intersection_labels_dilation_sub_visual)
                # cv2.imshow("segments_labels", segments_labels_visual)
                # cv2.imshow("connection_areas_visual", connection_areas_visual)
                image_intersection_labels = visualize_label_map(image_show*255., intersection_labels)
                image_intersection_labels_dilation = visualize_label_map(image_show*255., intersection_labels_dilation)
                intersection_labels_dilation_sub_visual = visualize_label_map(image_show*255., intersection_labels_dilation_sub)

                segments_labels_visual = visualize_label_map(image_show*255., segments_labels)
                connection_areas_visual = visualize_label_map(image_show*255., connection_areas)

                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i)  + "image_intersection_labels.png" ), image_intersection_labels)
                
                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i)  + "inte_sub.png" ), intersection_labels_dilation_sub_visual)
                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i)  + "segments_labels.png" ), segments_labels_visual)
                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i)  + "connection_areas_visual.png" ), connection_areas_visual)

    ###################################################################

                def connect_inters(inter_parts, tags_map_island):
                    class UF:
                        def __init__(self, N):
                            self.parents = list(range(N))
                        def union(self, child, parent):
                            self.parents[self.find(child)] = self.find(parent)
                        def find(self, x):
                            if x != self.parents[x]:
                                self.parents[x] = self.find(self.parents[x])
                            return self.parents[x]
                    
                    
                            
                    props = regionprops(inter_parts)
                    mean_vals = {}
                    for p in props:
                        row, col = p.coords[:, 0], p.coords[:, 1]
                        emb_mean = np.mean(tags_map_island[row, col], axis=0)
                        mean_vals[p.label] = emb_mean

                    uf = UF((len(props)))

                    import pdb; pdb.set_trace()
                    visited = set()
                    mapping = {}
                    for index, (key, value) in enumerate(mean_vals.items()):
                        mapping[key] = index

                    for index, (key, value) in enumerate(mean_vals.items()):                       
                        for index_2, (key_2, value_2) in enumerate(mean_vals.items()):
                            if key == key_2:
                                continue                      
                            print(np.sqrt((value - value_2)**2))  
                            if np.sqrt((value - value_2)**2) < 1:
                                uf.union(mapping[key], mapping[key_2])

                    connections = collections.defaultdict(list)

                    for curr_key, correponding_index in mapping.items():
                        connections[uf.find(correponding_index)].append(curr_key)

                    import pdb; pdb.set_trace()
                    final_inter_map = np.zeros(inter_parts.shape)
                    for new_id, lst in enumerate(connections.values()):
                        for part_id in lst:
                            final_inter_map[np.where(inter_parts==part_id)]= new_id + 1

                    return final_inter_map
                    print(connections)

                        


                tags_map_final = refine_pred[0,0,:,:].cpu().detach().numpy()
                # tags_map_single_chanel = tags_map_single_chanel * mask_pred_numpy
                # tags_map_final = tags_map_final * mask_pred_numpy
                if use_hard_mask:
                    tags_map_final = tags_map_final * hard_mask
                else:
                    tags_map_final = tags_map_final * mask_pred_numpy
                tags_map_with_island = tags_map_final + 30 * intersection_labels_dilation 

                seeds = label(connection_areas)
                seeds = (seeds * 500 + intersection_labels_dilation * connection_areas).astype(np.uint64)
                props = regionprops(seeds)

                # new_seeds = mask_from_seeds(tags_map_final, disect_mask>0, mask_pred_numpy, similarity_thres=0.5)


                visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), segments_labels)
                # cv2.imshow('segments_labels',visualize_img)
                # cv2.waitKey(0)    
                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i) + "segments_labels.png" ), visualize_img)
                    
                ds = DisjointSet()

                
                # islands_labels = label(intersection_areas_numpy2)
                
                # island =  binary_dilation(intersection_areas_numpy2, structure=np.ones((7,7)))
                island =  binary_dilation(intersection_areas_numpy2, structure=np.ones((3,3)))
                island =  binary_dilation(intersection_areas_numpy2, structure=np.ones((1,1)))
                island = remove_small_objects(island, 20)
                # island = remove_small_objects(island, 10)
                islands_labels = label(island)
                
                island_mapping, grouped_ds = mask_from_seeds_v3(tags_map_final, islands_labels, mask_pred_numpy, segments_labels, ds, similarity_thres=0.5)

                groups = grouped_ds.group
                result_map = np.zeros(segments.shape)
                raveled_segments = segments_labels.ravel()
                raveled_island = islands_labels.ravel()
                new_label_id = 1
                old_to_new_mapping = {}

                instance_map_recorder = []
                for key, group_set in groups.items():
                    group_as_array = np.array(list(group_set))
                    result_map[np.isin(segments_labels, group_as_array)] = new_label_id
                    old_to_new_mapping[key] = new_label_id
                    new_label_id += 1

                    canvas = np.zeros(segments.shape)
                    canvas[np.isin(segments_labels, group_as_array)] = 255.
                    canvas = im_dilation(canvas, mor_square(6)) * hard_mask  
                    instance_map_recorder.append(canvas)

                result_map_dilated = im_dilation(result_map, mor_square(5)) * mask_pred_numpy

                # add island
                for old_id in old_to_new_mapping.keys():
                    new_id = old_to_new_mapping[old_id]
                    group_members = np.array(list(groups[old_id]))
                    island_groups = []
                    for each_members in group_members:
                        island_groups.extend(island_mapping[each_members])
                    island_groups = np.array(island_groups)

                    result_map_dilated[np.isin(islands_labels, island_groups)] = new_id

                    temp = instance_map_recorder[new_id - 1]
                    temp[np.isin(islands_labels, island_groups)] = 255.
                    instance_map_recorder[new_id - 1] = temp


                visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), result_map_dilated)
                result_map_visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), result_map)
                # cv2.imshow('result_map_dilated',visualize_img)
                # cv2.waitKey(0)       
                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i) + "result_map_dilated.png" ), visualize_img)
                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i) + "result_map.png" ), result_map_visualize_img)
                res_visual_final = np.zeros(segments.shape)
                
                for page_id, page in enumerate(instance_map_recorder):
                    one_instance = {}
                    
                    # visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), page)
                    # cv2.imshow('results',visualize_img)
                    # cv2.waitKey(0)   
                    # visualize_img = visualize_label_map(np.tile(hard_mask*255., (3,1,1)).transpose(1,2,0), page)
                    # cv2.imshow('hardresults',visualize_img)
                    # cv2.waitKey(0) 
                    if np.sum(page>0) < 500:
                        continue 
                    page = im_dilation(page, mor_square(5)) * mask_pred_numpy
                    seg_poly = mask_to_pologons(page>0)
                    res_visual_final[page>0] = page_id + 1
                    if len(seg_poly) == 0:
                        continue
                    # import pdb; pdb.set_trace()
                    mask_ind = np.where(page>0)

                    each_instance_output_path = os.path.join(result_for_each_instance, image_id_in_BBBC010 + '_' + str(page_id) + '.png')
                    cv2.imwrite(each_instance_output_path, (page>0 ) * 255.)
                    print(len(mask_ind[0]))
                    # bbox = [int(min(end_point_instance[0][0][0],end_point_instance[1][0][0])),\
                    #     int(min(end_point_instance[0][0][1],end_point_instance[1][0][1])),\
                    #         int(abs(end_point_instance[0][0][0]- end_point_instance[1][0][0])),\
                    #             int(abs(end_point_instance[0][0][1] - end_point_instance[1][0][1]))]
                    try:
                        bbox = [int(min(mask_ind[1])),\
                            int(min(mask_ind[0])),\
                                int(abs(max(mask_ind[1])- min(mask_ind[1]))),\
                                    int(abs(max(mask_ind[0])- min(mask_ind[0])))]
                    except:
                        import pdb; pdb.set_trace()
                    one_instance['image_id'] = int(meta['image_id'].detach().cpu()) 
                    one_instance['bbox'] = bbox
                    one_instance['category_id'] = 1        
                    one_instance['segmentation'] = seg_poly       
                    one_instance['score'] = 99 
                    results.append(one_instance)

                result_map_visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), res_visual_final)
                cv2.imwrite(os.path.join(intermeidate_output_folder,str(i) + "_result_map.png" ), result_map_visualize_img)

        ########################################################################################            
        with open(result_file,'w') as wf:
            json.dump(results, wf)
            print('done') 

            ###########################################################################################################


    eval_gt = COCO(cfg.gt_path)
    print(cfg.gt_path)
    eval_dt = eval_gt.loadRes(result_file)
    import pdb; pdb.set_trace()
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='segm')
    
    # cocoEval.params.imgIds = [6]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()      

    cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')
    # cocoEval.params.imgIds = [6]
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
