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
from scipy.ndimage.morphology import grey_dilation, binary_dilation
from skimage.morphology import erosion
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

from test_config import cfg
from scipy import ndimage, misc
from skimage.color import label2rgb
from skimage.morphology import remove_small_objects
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

def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def rgba2rgb( rgba, background=(0,0,0) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 1.

    R, G, B = background
    import pdb; pdb.set_trace()
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

# from utils.preprocess import draw_mask
#TO DO:
#Fix Line 75; Only get one image here, but there are two images in total
#            input_var = torch.autograd.Variable(inputs.cuda()[:1])
#

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
    show_img = False
    # create model
    demo_folder = "./demo_folder/synthetic/"
    demo_folder = "./demo_folder/microtubule_dsc_mask/"
    segments_map_folder = "./demo_folder/microtubule_dsc_mask/segments_map"
    segments_result_map_di = "./demo_folder/microtubule_dsc_mask/segments_result_map_di"
    result_path = "./demo_folder/results/"
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(segments_map_folder, exist_ok=True)
    os.makedirs(segments_result_map_di, exist_ok=True)

    result_file = os.path.join(result_path, 'result_5.json')
    result_file = os.path.join(result_path, 'microtubule_dsc.json')
    # demo_folder = "./demo_folder/wroms/"
    demo_folder_connection_areas = "./demo_folder/synthetic_connection_areas/"
    os.makedirs(demo_folder, exist_ok=True)
    
    os.makedirs(demo_folder_connection_areas, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, cfg.inter, pretrained = True)
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
    results = []
    if True:
        with torch.no_grad():

            # for i, (inputs, targets, end_point_label, control_point_label, binary_targets, meta) in tqdm(enumerate(test_loader)):
            for i, (inputs, meta) in tqdm(enumerate(test_loader)):
                # if "17" not in meta["img_path"][0]:
                #     continue
                    # break
                # print(i)
                #########################################################3
                # img_path = '/home/yliu/work/data/microtubule_data/binary/0_man.png'
                # img_path = '/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/worm_bi/'+str(i)+'mask_pred_numpy.png'
                mt_files = os.listdir( '/home/yliu/work/codeToBiomix/data/dataset_mt_25/dscpredict/')
                img_path = '/home/yliu/work/codeToBiomix/data/dataset_mt_25/mask/'+str(i+1) +'_mask.png'
                img_path = os.path.join('/home/yliu/work/codeToBiomix/data/dataset_mt_25/dscpredict/',mt_files[i])
                
                image = imageio.imread(img_path)
                # image = image[:,:,0]

                image = np.asarray(image)
                image = cv2.erode(image, np.ones((2,2)))
                h, w = image.shape[:2]
                
                crop_height = h % 32
                crop_width = w % 32
                image = image[int(crop_height / 2) : int(h - crop_height /2),int(crop_width / 2) : int(w - crop_width /2)]
                # image = image[130:130 + 256,135:135 + 256]
                if len(image.shape) == 2:
                        image = np.tile(image,(3,1,1)).transpose(1,2,0)
                        image = np.asarray(image)
                img = im_to_torch(image)
                img = color_normalize(img, cfg.pixel_means)
                inputs = img.unsqueeze(0)
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
                hard_mask = 1.0 * (input_image[:,:,0] > 0)
                # cv2.imshow("input_image", input_image[:,:,0] * 255.)
                # cv2.waitKey(0)
                image_show = np.transpose(input_,(1,2,0))
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
                # intersection_areas_numpy3 = intersection_areas[0,2,:,:].cpu().detach().numpy()
                # mask_pred_numpy = input_var.data.cpu().numpy()[0]
                # mask_pred_numpy = mask_pred_numpy[0,:,:]

                
                mask_pred_numpy = (mask_pred_numpy > 0.7) *1.0
                mask_pred_numpy_2 = (mask_pred_numpy_2 > 0.7) *1.0
                mask_pred_numpy_2 = cv2.dilate(mask_pred_numpy_2, np.ones((3,3))) # outlines
                intersection_areas_numpy = (intersection_areas_numpy > 0.7) *1.0
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
                tags_map_single = tags_map_single * mask_pred_numpy 
                tags_map_single = smooth_emb(tags_map_single , 5)
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
                
                visual_tags_map_single[intersection_areas_numpy==0] = 0
                visual_tags_map_single[(mask_pred_numpy - intersection_areas_numpy) == 1] = 1
                rgba_img = cmap(visual_tags_map_single)

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

                plt.imsave(os.path.join(demo_folder,str(i) +"embedding_map_inter.png" ), rgba_img)
                
                #TODO: visualize island tag map!!!
                # import pdb; pdb.set_trace()
                if show_img:

                    
                    cv2.imshow('rgba_visual_tags_map',rgba_img)
                    cv2.waitKey(0)

                
                if show_img:
                    combined_show_mask_pred = draw_mask(image_show, mask_pred_numpy)
                    cv2.imshow('mask pred',combined_show_mask_pred)
                    cv2.waitKey(0)
                if show_img:
                    combined_show_intersection_areas = draw_mask(image_show, intersection_areas_numpy)
                    cv2.imshow('mask intersection_areas',combined_show_intersection_areas)
                    cv2.waitKey(0)
                if show_img:
                    combined_show_intersection_areas_2 = draw_mask(image_show, intersection_areas_numpy2)
                    cv2.imshow('mask intersection_areas_2',combined_show_intersection_areas_2)
                    cv2.waitKey(0)
                if show_img:
                    combined_show_mask_pred_numpy_2 = draw_mask(image_show, mask_pred_numpy_2)
                    cv2.imshow('mask mask_pred_numpy_2',combined_show_mask_pred_numpy_2)
                    cv2.waitKey(0)
                if show_img:
                    combined_show_disect_mask = draw_mask(image_show, disect_mask)
                    cv2.imshow('mask disect_mask',combined_show_disect_mask)
                    cv2.waitKey(0)

                combined_show_mask_pred = draw_mask(image_show, mask_pred_numpy)

                combined_show_intersection_areas = draw_mask(image_show, intersection_areas_numpy)

                combined_show_intersection_areas_2 = draw_mask(image_show, intersection_areas_numpy2)

                combined_show_mask_pred_numpy_2 = draw_mask(image_show, mask_pred_numpy_2)

                combined_show_disect_mask = draw_mask(image_show, disect_mask)

                cv2.imwrite(os.path.join(demo_folder,str(i) +"combined_show_mask_pred.png" ), combined_show_mask_pred * 255.)
                cv2.imwrite(os.path.join(demo_folder,str(i) +"combined_show_intersection_areas.png" ), combined_show_intersection_areas * 255.)
                cv2.imwrite(os.path.join(demo_folder,str(i) +"combined_show_intersection_areas_2.png" ), combined_show_intersection_areas_2 * 255.)
                cv2.imwrite(os.path.join(demo_folder,str(i) +"combined_show_mask_pred_numpy_2.png" ), combined_show_mask_pred_numpy_2 * 255.)
                cv2.imwrite(os.path.join(demo_folder,str(i) +"combined_show_disect_mask.png" ), combined_show_disect_mask*255.)

                intersection_labels = label(intersection_areas_numpy2)

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

                def mask_from_seeds_v3(embedding, island, mask, segments, ds, similarity_thres=1.0):

                    segments_props = regionprops(segments)

                    # initialize island mapping, each junction, we find which segments invlove with this junction.
                    island_maping = defaultdict(list)
                    
                    # disjoint set non-visited segments
                    not_visited = set(np.unique(segments))
                    #TODO: change 10 ** 4

                    #Loop throught each junction
                    for eachisland in np.unique(island):
                        if eachisland == 0:
                            continue
                        current_island = (island == eachisland) * 1.0
                        # get connection areas 
                        # 1. dilate junction

                        mask_for_phi = ~(mask>0)
                        phi = 1.0 * np.ones(mask.shape)
                        phi  = np.ma.MaskedArray(phi, mask_for_phi)

                        connection_areas = (cv2.dilate(current_island.astype('uint8'), np.ones((9,9))) * mask - cv2.dilate(current_island.astype('uint8'), np.ones((3,3))) > 0 ) * 1.0
                        # 2. find pixels overlap with segments
                        connection_areas_label = (connection_areas>0) * segments
                        
                        # 3. label each connector
                        connection_areas_props = regionprops(connection_areas_label)
                        if len(connection_areas_props) <= 0:
                            continue

                        visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), connection_areas_label)
                        # cv2.imshow('connection_areas',visualize_img)
                        # cv2.waitKey(0)

                        cv2.imwrite(os.path.join(demo_folder_connection_areas,str(i) + "_" + str(eachisland) + "_connection_areas.png" ), visualize_img)

                        mean_vals = {}
                        for p in connection_areas_props:
                            if p.label == 0:
                                continue
                            row, col = p.coords[:, 0], p.coords[:, 1]
                            emb_mean = np.mean(embedding[row, col], axis=0)
                            mean_vals[p.label] = emb_mean
                        
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
                        # print(cost_matrix)
                        cost_matrix[cost_matrix > similarity_thres] = 1000
                        assignments = np.ones(len(mean_vals_numpy)) * -1
                        # import pdb; pdb.set_trace()
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
                        col_ind = assignments.astype(int)
                        # import pdb; pdb.set_trace()
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
                            # cv2.waitKey(0)                    
                    for non_visited_id in not_visited:
                        if non_visited_id == 0:
                            continue
                        ds.add_single(non_visited_id)

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
                # segments = (mask_pred_numpy - binary_dilation(intersection_areas_numpy2, structure=np.ones((3,3))) - mask_pred_numpy_2 ) > 0 
                segments = (mask_pred_numpy - intersection_areas_numpy2 - mask_pred_numpy_2 ) > 0 
                segments = remove_small_objects(segments, 16)
                
                cv2.imwrite(os.path.join(segments_map_folder,mt_files[i][:-4] + "_segments.png" ), 255 * (segments> 0))
                segments_labels = label(segments)

                connection_areas = (intersection_labels_dilation_sub>0) * segments

                #########################################

                image_intersection_labels = label2rgb(intersection_labels, image=mask_pred_numpy, bg_label=0)
                image_intersection_labels_dilation= label2rgb(intersection_labels_dilation, image=mask_pred_numpy, bg_label=0)
                image_labels_dilation_real = label2rgb(intersection_labels_dilation_sub, image=mask_pred_numpy, bg_label=0)
                intersection_labels_dilation_sub_visual = label2rgb(intersection_labels_dilation_sub, image=mask_pred_numpy, bg_label=0)
                segments_labels_visual = label2rgb(segments_labels, image=mask_pred_numpy, bg_label=0)
                connection_areas_visual = label2rgb(connection_areas, image=mask_pred_numpy, bg_label=0)
                

                # TO visualize the connector in paper
                connector_visual = cv2.dilate(intersection_areas_numpy2, np.ones((15,15), np.uint8)) * mask_pred_numpy
                visual_connector = segments * connector_visual
                visual_connector_labels = label(visual_connector)

                visual_connector_labels = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), visual_connector_labels)
                cv2.imwrite(os.path.join(demo_folder,str(i) + "visual_connector_labels.png" ), visual_connector_labels)
                # cv2.imshow("visual_connector", visual_connector * 255.)
                # cv2.waitKey(0)
                # cv2.imshow("image_dilation", image_intersection_labels_dilation)
                # # cv2.imshow("image_real", image_labels_dilation_real)
                # cv2.imshow("inte_sub", intersection_labels_dilation_sub_visual)
                # cv2.imshow("segments_labels", segments_labels_visual)
                # cv2.imshow("connection_areas_visual", connection_areas_visual)
                visualize_image_intersection_labels = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), intersection_labels)
                visual_image_intersection_labels_dilation = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), intersection_labels_dilation)
                visual_intersection_labels_dilation_sub_visual = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), intersection_labels_dilation_sub)
                visual_segments_labels_visual = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), mask_pred_numpy)
                visual_connection_areas_visual = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), mask_pred_numpy)
                cv2.imwrite(os.path.join(demo_folder,str(i) + "image_intersection_labels.png" ), visualize_image_intersection_labels)
                cv2.imwrite(os.path.join(demo_folder,str(i) +"image_dilation.png" ), visual_image_intersection_labels_dilation)
                cv2.imwrite(os.path.join(demo_folder,str(i) +"inte_sub.png" ), visual_intersection_labels_dilation_sub_visual)
                cv2.imwrite(os.path.join(demo_folder,str(i) +"segments_labels.png" ), visual_segments_labels_visual)
                cv2.imwrite(os.path.join(demo_folder,str(i) +"connection_areas_visual.png" ), visual_connection_areas_visual)


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

                    final_inter_map = np.zeros(inter_parts.shape)
                    for new_id, lst in enumerate(connections.values()):
                        for part_id in lst:
                            final_inter_map[np.where(inter_parts==part_id)]= new_id + 1

                    return final_inter_map
                    print(connections)

                        



                tags_map_final = refine_pred[0,0,:,:].cpu().detach().numpy()
                # tags_map_single_chanel = tags_map_single_chanel * mask_pred_numpy
                tags_map_final = tags_map_final * mask_pred_numpy
                tags_map_with_island = tags_map_final + 30 * intersection_labels_dilation 

                seeds = label(connection_areas)
                seeds = (seeds * 500 + intersection_labels_dilation * connection_areas).astype(np.uint64)
                props = regionprops(seeds)

                # new_seeds = mask_from_seeds(tags_map_final, disect_mask>0, mask_pred_numpy, similarity_thres=0.5)


                visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), segments_labels)
                # cv2.imshow('segments_labels',visualize_img)
                # cv2.waitKey(0)    
                cv2.imwrite(os.path.join(demo_folder,str(i) +"segments_labels.png" ), visualize_img)
                

                # intilize disjointset
                ds = DisjointSet()

                # label the junctions, each juntion has an id
                islands = intersection_areas_numpy2
                
                islands = cv2.dilate(intersection_areas_numpy2, np.ones((5,5))) * mask_pred_numpy
                islands = remove_small_objects(islands>0, 10)
                islands_labels = label(islands)


                island_mapping, grouped_ds = mask_from_seeds_v3(tags_map_final, islands_labels, mask_pred_numpy, segments_labels, ds, similarity_thres= 0.5)
                

                ######################################################
                # from sklearn.manifold import TSNE
                # import seaborn as sns
                # visual_tags_map_single= normalize_include_neg_val((tags_map_final ))
                # rgba_img = cmap(visual_tags_map_single) 
                
                # # visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), control_points_map_label.cpu().detach().numpy()[0])
                # # control_points_map_label_numpy = control_points_map_label.cpu().detach().numpy()[0]
                # # cv2.imshow("visualize_img", visualize_img)
                # # cv2.imshow("temp", rgba_img)
                # # cv2.waitKey(0)
                # pixels = tags_map_final[control_points_map_label_numpy > 0]
                
                # connector_labels = control_points_map_label_numpy[control_points_map_label_numpy > 0]
                # connector_id = 1
                # connector_group_id = 1
                # connector_label_map = np.zeros(control_points_map_label_numpy.shape)

                # connector_label_gt_map = np.zeros(control_points_map_label_numpy.shape)
                # control_label_unique = np.unique(control_points_map_label_numpy)


                # ##
                # #put text settings
                # #
                # fontScale = 0.5
                # color = 1 
                # thickness = 1
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # ##
                # for each_control_label in control_label_unique:
                #     if each_control_label  == 0:
                #         continue
                #     connector_label_gt_map[control_points_map_label_numpy == each_control_label] = connector_group_id
                #     connector_group_id += 1

                #     temp = 1 * (control_points_map_label_numpy == each_control_label)
                #     temp_label = label(temp)
                #     for each_temp_label in np.unique(temp_label):
                #         if each_temp_label == 0:
                #             continue
                #         connector_label_map[np.where(temp_label == each_temp_label)] = connector_id

                #         blob = (temp_label == each_temp_label).astype("uint8")
                #         M = cv2.moments(blob )
                #         cX = int(M["m10"] / M["m00"])
                #         cY = int(M["m01"] / M["m00"])

                #         connector_label_map_put_text = cv2.putText(connector_label_map_put_text, str(connector_id + 1), (cX, cY), font, 
                #                     fontScale, color, thickness, cv2.LINE_AA)
                #         connector_id += 1

                # import copy
                # connector_label_map_copy = copy.deepcopy(connector_label_map)
                # connector_label_map_copy[connector_label_map_put_text > 0] = connector_id + 10
                # visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), connector_label_map_copy)
                # cv2.imwrite(os.path.join(demo_folder,str(i) +"connector_label_map_copy.png" ), visualize_img)


                # gt_pixels_lables = connector_label_gt_map[control_points_map_label_numpy > 0]
                # connector_labels_extracted = connector_label_map[control_points_map_label_numpy>0]
                # connector_labels_unique = np.unique(connector_label_map)
                # colormap = plt.cm.gist_ncar
                
                # connector_colors = []
                # connector_markers = []
                # connector_ids = []
                # connector_emb_values = []
                # label_to_image = {}
                # gt_labels_list = []
                # import pdb; pdb.set_trace()


                # CLASS_2_COLOR, COLOR_2_CLASS = GenColorMap(500)
                # m = np.repeat(["o", "s", "D", "*"], 500 /4 )

                # class_2_color_new = []
                # for each_tuple in CLASS_2_COLOR:
                #     new_tuple = tuple( [val/255. for val in each_tuple])
                #     class_2_color_new.append(new_tuple)
                # for idx_connector, each_label in enumerate(connector_labels_unique):
                #     if each_label == 0:
                #         continue
                #     emb_mean_connector = np.mean(pixels[connector_labels_extracted == each_label])
                #     gt_labels = np.mean(gt_pixels_lables[connector_labels_extracted == each_label])
                #     connector_ids.append(idx_connector+1)
                #     connector_emb_values.append(emb_mean_connector)
                #     connector_colors.append(class_2_color_new[int(gt_labels)])
                #     connector_markers.append(m[int(gt_labels)])
                #     gt_labels_list.append(gt_labels)
                #     label_to_image[idx_connector+1] = each_label

                # # fig, ax = plt.subplots()
                # # import matplotlib.cm as cm
                # # colors = cm.rainbow(np.linspace(0, 1, len(gt_labels_list)))

                # # for x,y, c in zip(connector_emb_values, gt_labels_list, colors):
                # #     plt.scatter(x, y, color=c)

                # # scatter = mscatter(connector_emb_values, gt_labels_list, c=connector_ids, m=connector_markers, ax=ax)

                # import pdb; pdb.set_trace()
                # fig, ax = plt.subplots()

                # plt.scatter(connector_emb_values,gt_labels_list, alpha=0.5)
                # plt.grid()
                # plt.yticks(range(0,int(max(gt_labels_list))))
                # plt.xlabel('Value of embedding (1D)')
                # plt.ylabel('Connection label id')
                # start, end = ax.get_xlim()
                # ax.xaxis.set_ticks(np.arange(-2, 4, 0.5))
                
                # for i, txt in enumerate(connector_ids):
                #     if txt in [22, 23, 28, 29]:
                #         print('ha')
                #         ax.annotate(txt, (connector_emb_values[i] + 0.05, gt_labels_list[i] + 0.15))
                #     elif i % 2 == 1:
                #         ax.annotate(txt, (connector_emb_values[i] + 0.10, gt_labels_list[i] + 0.15))
                #     else:
                #         ax.annotate(txt, (connector_emb_values[i] - 0.25, gt_labels_list[i] + 0.15))
                # plt.xticks(range(0,int(max(connector_emb_values) * 10 )), )


                # import pdb; pdb.set_trace()
                # # concatenated= np.concatenate(pixels, connector_labels)
                # tsne = TSNE()
                # X_embedded = tsne.fit_transform(pixels.reshape(-1,1))
                # sns.set(rc={'figure.figsize':(11.7,8.27)})
                # palette = sns.color_palette("bright", 10)
                # sns.scatterplot(X_embedded[:,0], X_embedded[:,1], legend='full', palette=palette)

                # import pdb; pdb.set_trace()
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
                    canvas = im_dilation(canvas, mor_square(7)) * hard_mask  
                    instance_map_recorder.append(canvas)
                
                
                result_map_dilated = im_dilation(result_map, mor_square(2)) * mask_pred_numpy   
                

                result_map_segments_canvas = np.zeros(segments.shape)
                result_map_segments_for_save = visualize_label_map(np.tile(result_map_segments_canvas*255., (3,1,1)).transpose(1,2,0), result_map_dilated)
                cv2.imwrite(os.path.join(segments_result_map_di,mt_files[i][:-4] + "_labeled_segments.png"),result_map_segments_for_save)


                eroed_island_labesl = erosion(islands_labels, np.ones((3,3))) 
                ## add island
                for old_id in old_to_new_mapping.keys():
                    new_id = old_to_new_mapping[old_id]
                    group_members = np.array(list(groups[old_id]))
                    island_groups = []
                    for each_members in group_members:
                        island_groups.extend(island_mapping[each_members])
                    island_groups = np.array(island_groups)

                    # result_map_dilated[np.isin(islands_labels, island_groups)] = new_id
                    result_map_dilated[np.isin(eroed_island_labesl, island_groups)] = new_id

                    temp = instance_map_recorder[new_id - 1]
                    temp[np.isin(islands_labels, island_groups)] = 255.
                    instance_map_recorder[new_id - 1] = temp

                
                visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), result_map_dilated)
                result_map_visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), result_map)
                # cv2.imshow('result_map_dilated',visualize_img)
                # cv2.waitKey(0)       
                cv2.imwrite(os.path.join(demo_folder,str(i) +"result_map_dilated.png" ), visualize_img)
                cv2.imwrite(os.path.join(demo_folder,str(i) +"result_map.png" ), result_map_visualize_img)

                
                for page in instance_map_recorder:
                    one_instance = {}
                    # visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), page)
                    # cv2.imshow('results',visualize_img)
                    # cv2.waitKey(0)   
                    # visualize_img = visualize_label_map(np.tile(hard_mask*255., (3,1,1)).transpose(1,2,0), page)
                    # cv2.imshow('hardresults',visualize_img)
                    # cv2.waitKey(0)     
                    seg_poly = mask_to_pologons(page>0)
                    # import pdb; pdb.set_trace()
                    mask_ind = np.where(page>0)

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

                continue
    
                import pdb; pdb.set_trace()
                # new_seeds = mask_from_seeds(visual_tags_map_single, intersection_labels_dilation_sub>0, mask_pred_numpy, similarity_thres=0.15)
                new_seeds[np.where(intersection_labels_dilation==0)] = 0
                new_seeds_visual = label2rgb(new_seeds, image=mask_pred_numpy, bg_label=0)

                visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), new_seeds)
                cv2.imshow('control_points_label',visualize_img)
                cv2.waitKey(0)

                cv2.imshow("new_seeds_visual", new_seeds_visual)
                cv2.waitKey(0)            
                # props = regionprops(seeds)
                new_final_con = connect_inters(new_seeds, tags_map_with_island)
                visualize_img = visualize_label_map(np.tile(mask_pred_numpy*255., (3,1,1)).transpose(1,2,0), new_final_con)
                cv2.imshow('new_final_con',visualize_img)
                cv2.waitKey(0)            
                continue
                # for p in props:
                #     row, col = p.coords[:, 0], p.coords[:, 1]
                #     emb_mean = np.mean(embedding[row, col], axis=0)
                #     mean[p.label] = emb_mean
                # import pdb; pdb.set_trace()
                mean_val = {}
                val_label = {}
                import pdb; pdb.set_trace()
                for p in props:
                    # print(p.label)
                    row, col = p.coords[:, 0], p.coords[:, 1]
                    emb_mean = np.mean(tags_map_with_island[row, col])
                    emb_mean_island = np.mean(intersection_labels_dilation[row, col])
                    # print(intersection_labels_dilation[row, col])
                    canvas = 30 * intersection_labels_dilation
                    canvas_2 = np.zeros(intersection_labels_dilation.shape)
                    canvas[row, col] = 0
                    canvas_2[row, col] = 255.
                    canvas_visual = label2rgb(canvas, image=mask_pred_numpy, bg_label=0)
                    canvas_visual[row, col,0] = 255
                    canvas_visual[row, col,1] = 0
                    canvas_visual[row, col,2] = 0
                    # print(emb_mean)
                    # print(emb_mean_island * 30)                
                    # cv2.imshow("seeds_visual", canvas_visual)
                    # cv2.imshow("canvas_2", canvas_2)
                    # cv2.waitKey(0)


                    mean_val[p.label] = emb_mean
                    val_label[emb_mean] = p.label


                vals = np.fromiter(mean_val.values(), dtype=float)
                keyvals = np.fromiter(mean_val.keys(), dtype=int)

                mean_matrix = np.zeros((len(props), len(props))) + 100
                for index, each_val in enumerate(vals):
                    for index_2, second_val in enumerate(vals[index+1:] ):

                        mean_matrix[index, index+1+index_2] = (each_val - second_val) ** 2
                        print((each_val - second_val) ** 2)
                
                connections = np.where(mean_matrix < 0.5)
                recurse = []
                
                for each_connection in range(len(connections[0])):
                    obja = keyvals[connections[0][each_connection]]
                    objb = keyvals[connections[1][each_connection]]
                    mean_val[obja]
                    
                    print(obja)
                    print(objb)
                    print(mean_val[obja])
                    print(mean_val[objb])
                    # if obja in recurse:
                    #     continue
                    # if objb in recurse:
                    #     continue
                    # recurse.append(obja)
                    # recurse.append(objb)

                    # if obja == 0 :
                    #     continue
                    # if objb == 0:
                    #     continue
                    
                    # if obja not in recurse:
                    #     unit = {}
                    #     unit["parent"] = obja
                    #     unit["children"] = {obja}
                    #     recurse[obja] = unit
                    #     converted = obja

                    # elif objb not in recurse:
                    #     unit = {}
                    #     unit["parent"] = obja
                    #     unit["children"] = {obja}
                    #     recurse[objb] = unit
                    # elif obja in recurse:
                    #     converted = recurse[obja]["parent"]
                    # elif objb in recurse:
                    #     to_convert = recurse[objb]["children"]
                    #     recurse[converted] 
                    
                    # for each_to_convert in to_convert:
                    #     seeds[seeds == each_to_convert] = converted

                    # seeds[seeds == objb] = 0
                    # seeds[seeds == obja] = 0
                    seeds_visual = label2rgb(seeds, image=mask_pred_numpy, bg_label=0)
                    cv2.imshow("seeds_visual", seeds_visual)

                    seeds_visual[:,:,0][np.where(seeds == objb)] = 255
                    seeds_visual[:,:,0][np.where(seeds == obja)] = 255
                    seeds_visual[:,:,1][np.where(seeds == objb)] = 0
                    seeds_visual[:,:,1][np.where(seeds == obja)] = 0
                    seeds_visual[:,:,2][np.where(seeds == objb)] = 0
                    seeds_visual[:,:,2][np.where(seeds == obja)] = 0

                    cv2.imshow("seeds_visual_2", seeds_visual)

                    cv2.waitKey(0)

                seeds_visual = label2rgb(seeds, image=mask_pred_numpy, bg_label=0)
                cv2.imshow("seeds_visual", seeds_visual)
                cv2.waitKey(0)
                # import pdb; pdb.set_trace()
                # for island in np.unique(intersection_labels_dilation):
                #     print(island)
                #     tags_map_with_island_one = ((tags_map_final * (intersection_labels_dilation == island)) -  20 * island + min(np.min(tags_map_final), 0))
                #     visual_tags_map_single_chanel= normalize_include_neg_val(tags_map_with_island_one)
                #     cmap = plt.get_cmap('jet')
                #     rgba_img = cmap(visual_tags_map_single_chanel)
                #     cv2.imshow('tags_map_with_island',rgba_img)
                #     cv2.waitKey(0)



                # image_intersection_labels = label2rgb(intersection_labels, image=mask_pred_numpy, bg_label=0)
                # image_intersection_labels_dilation= label2rgb(intersection_labels_dilation, image=mask_pred_numpy, bg_label=0)
                # image_labels_dilation_real = label2rgb(intersection_labels_dilation_real, image=mask_pred_numpy, bg_label=0)
                # intersection_labels_dilation_sub = label2rgb(intersection_labels_dilation_sub, image=mask_pred_numpy, bg_label=0)
                # segments_labels_visual = label2rgb(segments_labels, image=mask_pred_numpy, bg_label=0)
                # connection_areas_visual = label2rgb(connection_areas, image=mask_pred_numpy, bg_label=0)
                # cv2.imshow("image_intersection_labels", image_intersection_labels)
                # cv2.imshow("image_dilation", image_intersection_labels_dilation)
                # cv2.imshow("image_real", image_labels_dilation_real)
                # cv2.imshow("inte_sub", intersection_labels_dilation_sub)
                # cv2.imshow("segments_labels", segments_labels_visual)
                # cv2.imshow("connection_areas_visual", connection_areas_visual)
                # cv2.waitKey(0)
                if show_img:
                    combined_show = draw_mask(image_show, mask_pred_numpy_2)
                    cv2.imshow('mask mask_pred_numpy_2',combined_show)
                    cv2.imshow('mask mask_pred_numpy_2_2',mask_pred_numpy_2 * 255.)
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
                # control_point_tags = refine_pred[:,0:1,:,:]
                tags_map = refine_pred[0,0,:,:].cpu().detach().numpy()
                tags_map2 = ndimage.median_filter(tags_map, size=9) 
                tags_map2 = np.tile(tags_map2,(3,1,1)).transpose(1,2,0)

                tags_map_single_chanel = refine_pred[0,0,:,:].cpu().detach().numpy()
                # tags_map_single_chanel = tags_map_single_chanel * mask_pred_numpy
                tags_map_single_chane_inter = tags_map_single_chanel * mask_pred_numpy

                # norm_ed = normalize_include_neg_val(tags_map_single_chane_inter) * 255.
                # norm_ed = tags_map_single_chane_inter

                # tags_map_single_chane_inter = tags_map_single_chanel * intersection_areas_numpy
                # histogram, bin_edges = np.histogram(norm_ed, bins=250, range=(0.1, np.max(norm_ed)))  
                # plt.plot(bin_edges[0:-1], histogram)
                # plt.title("Grayscale Histogram")
                # plt.xlabel("grayscale value")
                # plt.ylabel("pixels")
                # plt.xlim(0.1, np.max(norm_ed))
                # plt.show()
                
                #####################################################################################33
                # delete_indices = np.where(tags_map_single_chane_inter.ravel() == 0)
                # np_hist, bins = np.histogram(np.delete(tags_map_single_chane_inter.ravel(), delete_indices), bins=255)
                # from scipy.signal import find_peaks
                # import pdb; pdb.set_trace()
                # peaks, _ = find_peaks(np_hist, height=90)
                # thresholds =bins[(peaks[:-2] + peaks[1:-1]) // 2  ]
                # # thresholds = np.array([-3.85478253, -2.8813822 , -1.94691789, -1.28500568, -0.70096548, 
                # #     -0.03905326,  0.50605092,  1.0511551 ,  1.86881137,  2.80327567,              
                # #         3.65986796])  

                # # thresholds = threshold_multiotsu(tags_map_single_chane_inter, classes = 4)
                # import pdb; pdb.set_trace()
                # regions = np.digitize(tags_map_single_chane_inter, bins=thresholds)
                
                # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

                # # Plotting the original image.
                
                # ax[0].imshow(tags_map_single_chane_inter, cmap='gray')
                # ax[0].set_title('Original')
                # ax[0].axis('off')

                # # Plotting the histogram and the two thresholds obtained from
                # # multi-Otsu.
                # delete_indices = np.where(tags_map_single_chane_inter.ravel() == 0)
                # ax[1].hist(np.delete(tags_map_single_chane_inter.ravel(), delete_indices), bins=255)
                # ax[1].set_title('Histogram')
                # for thresh in thresholds:
                #     ax[1].axvline(thresh, color='r')

                # # Plotting the Multi Otsu result.
                # ax[2].imshow(regions, cmap='jet')
                # ax[2].set_title('Multi-Otsu result')
                # ax[2].axis('off')

                # plt.subplots_adjust()

                # plt.show()
                ##############################################################################################

                # histogram, bin_edges = np.histogram(tags_map_single_chane_inter, bins=250, range=(0.1, np.max(tags_map_single_chane_inter)))  
                # plt.plot(bin_edges[0:-1], histogram)
                # plt.title("Grayscale Histogram")
                # plt.xlabel("grayscale value")
                # plt.ylabel("pixels")
                # plt.xlim(0, np.max(tags_map_single_chane_inter))
                # plt.show()

                # norm_ed = normalize_include_neg_val(tags_map_single_chane_inter)
                # histogram, bin_edges = np.histogram(norm_ed, bins=250, range=(np.min(norm_ed)+0.1, np.max(norm_ed)))  
                # plt.plot(bin_edges[0:-1], histogram)
                # plt.title("Grayscale Histogram")
                # plt.xlabel("grayscale value")
                # plt.ylabel("pixels")
                # plt.xlim(np.min(norm_ed)+0.1, np.max(norm_ed))
                # plt.show()


                tags_map = np.tile(tags_map,(3,1,1)).transpose(1,2,0)
                # tags_map2 = tags_map
                tags_map3 = tags_map
                tags_map4 = tags_map
                tags_map5 = tags_map

                # tags_map = refine_pred[0,0:3,:,:]
                # tags_map2 = refine_pred[0,0:3,:,:]
                # tags_map3 = refine_pred[0,0:3,:,:]
                # tags_map4 = refine_pred[0,0:3,:,:]
                # tags_map5 = refine_pred[0,0:3,:,:]

                # tags_map2 = refine_pred[0,3:6,:,:]
                # tags_map3 = refine_pred[0,6:9,:,:]
                # tags_map4 = refine_pred[0,9:12,:,:]
                # tags_map5 = refine_pred[0,12:15,:,:]
    
                
                # cluster_embeddings = tags_map.cpu().detach().numpy()
                # cluster_embeddings = cluster_embeddings.transpose((1,2,0)).reshape((-1,3))
                # restriction_map_flat = restriction_map.reshape((-1,3))
                # cluster_embeddings_pixels = cluster_embeddings[np.where(restriction_map_flat[:,0] == 1)]
                # bandwidth = estimate_bandwidth(cluster_embeddings_pixels, quantile=0.3, n_samples=300)
                # ms = MeanShift(bandwidth=bandwidth)
                # ms.fit(cluster_embeddings_pixels)
                # labels = ms.labels_
                # labels_unique = np.unique(labels)
                # n_clusters_ = len(labels_unique)

                # tags_map = tags_map.permute((1,2,0)).cpu().detach().numpy()
                # tags_map2 = tags_map2.permute((1,2,0)).cpu().detach().numpy()
                # tags_map3 = tags_map3.permute((1,2,0)).cpu().detach().numpy()
                # tags_map4 = tags_map4.permute((1,2,0)).cpu().detach().numpy()
                # tags_map5 = tags_map5.permute((1,2,0)).cpu().detach().numpy()


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

                # tags_map=smooth_emb(tags_map, 5)
                # tags_map2=smooth_emb(tags_map2, 5)
                # tags_map3=smooth_emb(tags_map3, 5)
                # tags_map4=smooth_emb(tags_map4, 5)
                # tags_map5=smooth_emb(tags_map5, 5)

                # tags_map[np.where((1-restriction_map)>0)] = 0
                # tags_map2[np.where((1-restriction_map)>0)] = 0
                # tags_map3[np.where((1-restriction_map)>0)] = 0
                # tags_map4[np.where((1-restriction_map)>0)] = 0
                # tags_map5[np.where((1-restriction_map)>0)] = 0

                visual_tags_map= normalize_include_neg_val(tags_map)
                visual_tags_map2= normalize_include_neg_val(tags_map2)
                visual_tags_map3 = normalize_include_neg_val(tags_map3)
                visual_tags_map3= normalize_include_neg_val(tags_map3)
                visual_tags_map4= normalize_include_neg_val(tags_map4)
                visual_tags_map5= normalize_include_neg_val(tags_map5)
                visual_tags_map_single_chanel= normalize_include_neg_val(tags_map_single_chane_inter)
    
                cmap = plt.get_cmap('jet')
                rgba_img = cmap(visual_tags_map_single_chanel)
                cv2.imshow('rgba_visual_tags_map',rgba_img)
                
                cv2.imshow('visual_tags_map',visual_tags_map)
                cv2.imshow('visual_tags_map2',visual_tags_map2)
                cv2.imshow('visual_tags_map3',visual_tags_map3)
                cv2.imshow('visual_tags_map4',visual_tags_map4)
                cv2.imshow('visual_tags_map5',visual_tags_map5)
                cv2.waitKey(0)
                mask_pred_numpy = cv2.erode(mask_pred_numpy.astype('uint8'), np.ones((3,3)))
                island_labels = label(intersection_areas_numpy>0)

                segments = ((mask_pred_numpy - intersection_areas_numpy * mask_pred_numpy) > 0 )
                segments_labels = label(segments)

                connection_areas = cv2.dilate(segments.astype('uint8'), np.ones((9,9))) * intersection_areas_numpy
                connection_area_map = ndimage.median_filter(connection_areas * tags_map_single_chanel, size=3) 
                import pdb; pdb.set_trace()
                cmap = plt.get_cmap('jet')
                rgba_img = cmap(connection_area_map)
                cv2.imshow('connection_area_map',connection_area_map)
                cv2.imshow('segments',segments * 255.)
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

        
    
        with open(result_file,'w') as wf:
            json.dump(results, wf)
            print('done') 

    eval_gt = COCO(cfg.gt_file)
    print(cfg.gt_file)
    eval_dt = eval_gt.loadRes(result_file)

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

    CLASS_2_COLOR, COLOR_2_CLASS = GenColorMap(1000)



    for idx in tqdm(range(len(eval_dt.anns))):
        if idx == 0: 
            continue
        ann = eval_dt.anns[idx]
        file_path = eval_gt.loadImgs(ann['image_id'])[0]['file_name']
        file_path = os.path.join(cfg.img_dir, file_path)
        print(idx)
        
        # im = cv2.imread(file_path)
        image = imageio.imread(file_path)
        # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # image = np.tile(image,(3,1,1)).transpose(1,2,0)

        image = np.asarray(image)


        im = (255 * (image > 0)).astype('uint8')   

        im_h, im_w = im.shape[:2]

        canvas = np.zeros(im.shape,dtype = np.float32) 
        bbox = ann['bbox']
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        category_id = ann['category_id']
        import cocoapi.PythonAPI.pycocotools.mask as coco_mask_util
        # import pdb; pdb.set_trace()
        from utils.cv2_util import pologons_to_mask
        # def pologons_to_mask(polygons, size):
        #     height, width = size
        #     # formatting for COCO PythonAPI

        #     rles = coco_mask_util.frPyObjects(polygons, height, width)


        #     rle = coco_mask_util.merge(rles)
        #     mask = coco_mask_util.decode(rle)
        #     return 
            
        try:
            mask = pologons_to_mask(ann['segmentation'],im.shape[:2])
        except:
            continue

        
        # canvas = draw_mask(im, mask, [0,255,155])
        # canvas = draw_mask_color(np.tile(im,(3,1,1)).transpose(1,2,0), mask, [0,255,155])
        canvas = visualize_label_map(np.tile(im,(3,1,1)).transpose(1,2,0), mask * 255)
        cv2.imshow("ttt", canvas)
        cv2.waitKey(0)

        ann_gt_ids = eval_gt.getAnnIds(imgIds=ann['image_id'])
        anns = eval_gt.loadAnns(ann_gt_ids)
        canvas = np.zeros(im.shape,dtype = np.float32) 
        canvas = im
        c = 0
        for ann_gt in anns:
            print(c)
            c = c + 1
            mask = pologons_to_mask(ann_gt['segmentation'],im.shape[:2])
            bbox = ann_gt['bbox']
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h            
            canvas = visualize_label_map(np.tile(im,(3,1,1)).transpose(1,2,0), mask * 255)
            canvas = cv2.rectangle(canvas, (x1,y1), (x2,y2), [100,255,155], 2) 
            cv2.imshow('tt', canvas)
            cv2.waitKey(0)      

    # result_path = args.result
    # if not isdir(result_path):
    #     mkdir_p(result_path)
    # result_file = os.path.join(result_path, 'result.json')
    # with open(result_file,'w') as wf:
    #     json.dump(full_result, wf)

    # # evaluate on COCO
    # eval_gt = COCO(cfg.ori_gt_path)
    # eval_dt = eval_gt.loadRes(result_file)
    # cocoEval = COCOeval(eval_gt, eval_dt, iouType='segm')
    # # cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')
    # cocoEval.params.imgIds = image_ids_for_eval
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
    # # evaluate on COCO
    # eval_gt = COCO(cfg.ori_gt_path)
    # eval_dt = eval_gt.loadRes(result_file)
    # # cocoEval = COCOeval(eval_gt, eval_dt, iouType='segm')
    # cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')
    # cocoEval.params.imgIds = image_ids_for_eval
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()

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
