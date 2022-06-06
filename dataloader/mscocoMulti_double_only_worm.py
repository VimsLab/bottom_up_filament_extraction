import os
import numpy as np
import json
import random
import math
import cv2
import skimage
import skimage.transform
#import imageio
import copy
import skfmm
import torch
import torch.utils.data as data

from pycocotools.coco import COCO

from utils.cv2_util import pologons_to_mask
from utils.osutils import *
from utils.imutils import *
from utils.transforms import *
from utils.preprocess import get_keypoint_discs_offset, compute_mid_long_offsets,compute_short_offsets, get_keypoint_discs, compute_mid_offsets
from utils.preprocess import compute_closest_control_point_offset
from utils.preprocess import visualize_offset, visualize_points, visualize_label_map
from utils.preprocess import draw_mask
from skimage.measure import label


import imageio
# from PIL import Image
# from matplotlib import pyplot as plt


# from config import cfg
# import torchvision.datasets as datasets
# import pdb
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

class MscocoMulti_double_only_worm(data.Dataset):
    def __init__(self, cfg, train=True):
        self.img_folder = cfg.img_path
        self.binary_folder = cfg.binary_folder
        self.is_train = train
        self.inp_res = cfg.data_shape
        self.out_res = cfg.output_shape
        self.pixel_means = cfg.pixel_means
        self.num_class = cfg.num_class
        self.cfg = cfg
        self.max_objects_num = 16
        self.disc_radius = cfg.disc_radius
        self.bbox_extend_factor = cfg.bbox_extend_factor
        self.crop_width = cfg.crop_width
        self.reverse_flag = cfg.reverse_flag
        self.debug = True
        self.debug = False

        self.write_image = False
        self.demo_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/training_data_worm_test"

        os.makedirs(self.demo_folder, exist_ok=True)
        self.inter = cfg.inter


        self.fiber_coco = COCO(cfg.gt_path)
        self.ids = [key for key in list(self.fiber_coco.anns.keys()) if len(self.fiber_coco.anns[key]['seq_x_col'])!=0 ] 
        self.img_ids = list(self.fiber_coco.imgs.keys())
        if train:
            self.scale_factor = cfg.scale_factor
            self.rot_factor = cfg.rot_factor
            self.symmetry = cfg.symmetry

            self.anno = []
            

            with open(cfg.gt_path) as anno_file:
                self.anno.extend(json.load(anno_file))

            # for i in range(len(cfg.gt_path)):
            #     path = cfg.gt_path[i]
            #     with open(path) as anno_file:
            #         self.anno.extend(json.load(anno_file))
        else:


            self.anno = []

            with open(cfg.gt_path) as anno_file:
                self.anno.extend(json.load(anno_file))

            # self.gt_image_root = cfg.gt_image_root
            # self.gt_root = cfg.gt_root
            # self.gt_file = cfg.gt_file
            # gt_anno = os.path.join(self.gt_root ,self.gt_file)
            # self.coco_fiber = COCO(gt_anno)
            # self.coco_ids = self.coco_fiber.getImgIds()
            # self.catIds = self.coco_fiber.getCatIds()
            # import pdb;pdb.set_trace()
    def cropAndResizeImage(self, img, crop_width, output_shape):
        height, width = self.output_shape[0], self.output_shape[1]
        curr_height, curr_width = img.shape[0], img.shape[1]
        image = img[crop_width : img.shape[0] - crop_width, crop_width : img.shape[1] - crop_width]

    def offset_to_color_map(self, offset):

        # print(np.max(offset/np.max(np.absolute(offset))))
        offset = offset / np.max(np.absolute(offset))
        # import pdb;pdb.set_trace()

        # color_map = np.zeros((3,) + offset.shape )
        positive = copy.copy(offset)
        negative = copy.copy(offset)

        positive[np.where(positive<0)] = 0
        negative[np.where(negative>0)] = 0
        # import pdb;pdb.set_trace()

        negative = np.absolute(negative)

        # import pdb;pdb.set_trace()
        r = positive.astype(np.float32)
        g = negative.astype(np.float32)
        b = np.zeros(offset.shape, dtype = np.float32)
        color_map = cv2.merge([r,g,b])
        # import pdb;pdb.set_trace()

        return color_map.astype(np.float32)

    def get_direction(self, mask):
        # find outer contour
        cntrs = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        # Retrieve the key parameters of the rotated bounding box

        # get rotated rectangle from outer contour
        rotrect = cv2.minAreaRect(cntrs[0])

        rect = cv2.minAreaRect(cntrs[0])
        center = (int(rect[0][0]),int(rect[0][1])) 
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])
 
     
        if width < height:
            angle = 90 - angle
        else:
            angle = -angle
        # print(angle)
        # cv2.imshow('angle', mask*255.)
        # cv2.waitKey(0)
        return angle

    def __getitem__(self, index):
        # import pdb;pdb.set_trace()

        if self.is_train:
            # a = self.anno[index]
            coco_fiber = self.fiber_coco
            img_id = self.img_ids[index]
            # img_id = coco_fiber.anns[ann_id]['image_id']
            # image_name = a['imgInfo']['file_path']
            # import pdb; pdb.set_trace()
            # img_path =  a['imgInfo']['file_path']

            
            w1_file_name = coco_fiber.loadImgs(img_id)[0]['w1']
            w2_file_name = coco_fiber.loadImgs(img_id)[0]['w2']
            binary_file_name = coco_fiber.loadImgs(img_id)[0]['binary']

            w1_file_path =  os.path.join(self.img_folder, w1_file_name)
            w2_file_path =  os.path.join(self.img_folder, w2_file_name)

            image_w1 = cv2.imread(w1_file_path, cv2.COLOR_BGR2GRAY)
            non_crop_image_shape = image_w1.shape
            

            image_w2 = cv2.imread(w2_file_path, cv2.COLOR_BGR2GRAY)

            # image_w1 = normalize_min_max(image_w1, 0, np.max(image_w1))
            # image_w2 = normalize_min_max(image_w2, 0, np.max(image_w2))
            image_w1 = normalize_min_max(image_w1, 0, 4095)
            image_w2 = normalize_min_max(image_w2, 0, 3072)

            image_w1 = np.asarray(image_w1)
            image_w2 = np.asarray(image_w2)
            image_w1 = np.expand_dims(image_w1, 0)
            image_w2 = np.expand_dims(image_w2, 0)
            image_w1_w2 = np.concatenate((image_w1, image_w2), axis=0).transpose(1,2,0)

            # import pdb; pdb.set_trace()
        # print(image.shape)
            image = np.asarray(image_w1_w2)
            image_visual = image[:,:,1] * 255
            instances_annos_ids = coco_fiber.getAnnIds(img_id)
            instances_annos = coco_fiber.loadAnns(instances_annos_ids)
        # if True:
            # import pdb; pdb.set_trace()
            # instances_annos = coco_fiber.loadAnns(img_id)
            # overlapping_area_pologon = a['unit']['overlapping_area_pologon']



            cropped_image_shape = (image.shape[0], image.shape[1])
            
            self.out_res = (image.shape[0],image.shape[1])

            mask_label = np.zeros(cropped_image_shape, dtype = np.float32)
            h_scale =  self.out_res[0] / cropped_image_shape[0]
            v_scale =  self.out_res[1] / cropped_image_shape[1]

            end_points_map_final = np.zeros(cropped_image_shape, dtype = np.float32)

            end_points_map_label = np.zeros(cropped_image_shape, dtype = np.float32)
            # end_points_map_label = cv2.resize(end_points_map_label, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            # end_points_map_label = np.expand_dims(end_points_map_label, axis = 0)
            # import pdb;pdb.set_trace()


            control_points_map_final = np.zeros(cropped_image_shape, dtype = np.float32)

            control_points_map_label = np.zeros(cropped_image_shape, dtype = np.float32)
            # control_points_map_label = cv2.resize(control_points_map_label, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            # control_points_map_label = np.expand_dims(control_points_map_label, axis = 0)

            # Short offset
            end_points_off_sets_shorts_map_h_final = np.zeros(cropped_image_shape, dtype = np.float32)
            end_points_off_sets_shorts_map_v_final = np.zeros(cropped_image_shape, dtype = np.float32)

            control_points_off_sets_shorts_map_h_final = np.zeros(cropped_image_shape, dtype = np.float32)
            control_points_off_sets_shorts_map_v_final = np.zeros(cropped_image_shape, dtype = np.float32)

            # Mid offset
            off_sets_nexts_map_h_final = np.zeros(cropped_image_shape, dtype = np.float32)
            off_sets_nexts_map_v_final = np.zeros(cropped_image_shape, dtype = np.float32)
            off_sets_prevs_map_h_final = np.zeros(cropped_image_shape, dtype = np.float32)
            off_sets_prevs_map_v_final = np.zeros(cropped_image_shape, dtype = np.float32)

            # Area Offset
            area_offset_map_h_final = np.zeros(cropped_image_shape, dtype = np.float32)
            area_offset_map_v_final = np.zeros(cropped_image_shape, dtype = np.float32)

            #segmentation mask
            segmentation_mask_final = np.zeros(cropped_image_shape, dtype = np.float32)
            segmentation_mask_final_erode = np.zeros(cropped_image_shape, dtype = np.float32)
            segmentation_mask_final_dilate = np.zeros(cropped_image_shape, dtype = np.float32)

            # outline mask
            outline_final = np.zeros(cropped_image_shape, dtype = np.float32)

            #segorientation_mask
            directional_mask = np.zeros(cropped_image_shape + (6,), dtype = np.float32)
            # print(instances_annos)
            
            # connection map
            stack_for_connection_map = {}
            connection_pair = []
            
            #TODO: change hardcoded maximum number of instances
            connection_map = np.zeros((50,50))

            for instance_id , instance in enumerate(instances_annos):

                if self.reverse_flag == True:
                    if instance_id % 2 == 0:
                        continue
                # import pdb; pdb.set_trace()
                end_points = instance['endpoints']
                control_points = instance['control_points']
                seg = instance['segmentation']

                # off_sets_prevs = instance['off_sets_prev']
                # off_sets_nexts = instance['off_sets_next']
                # seg = instance['dilate']


                segmentation_mask = np.zeros((image.shape[0], image.shape[1]), dtype = np.float32)
                # segmentation_mask[skel] = 1
                # segmentation_mask = cv2.dilate(segmentation_mask, np.ones((3,3)))

                segmentation_mask_poly_mask = pologons_to_mask(seg,cropped_image_shape)
                segmentation_mask[np.where(segmentation_mask_poly_mask>0)] = 1

                #slim mask prediction
                segmentation_mask_erode = cv2.erode(segmentation_mask, np.ones((3,3)))
                # segmentation_mask = np.ascontiguousarray(segmentation_mask, dtype=np.uint8)
                
                segmentation_mask_dialte = cv2.dilate(segmentation_mask, np.ones((3,3)))
                
                _, roi = cv2.threshold((segmentation_mask*255.).astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
                cnt  = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                outline = np.zeros(segmentation_mask.shape, dtype=np.uint8)
                outline = cv2.drawContours(outline, cnt[1], -1, (255, 255, 255))
                outline = (outline > 0 ) * 1.0

                # this_id = instance['instance_id']

                # cv2.imshow("segmentation_mask", segmentation_mask * 255.)
                # cv2.waitKey(0)

                # instance_id starts from 0. So we add one to all of them, starts from 1
                this_id = instance_id + 1
                # import pdb; pdb.set_trace()


                ## BUILD CONNECTION MAP A
                # '''
                if not self.inter:
                    for stack_id in stack_for_connection_map.keys():
                        curr_instance = stack_for_connection_map[stack_id]
                        if np.sum(curr_instance * segmentation_mask > 0) > 0:
                            connection_pair.append((this_id, stack_id))
                            connection_map[this_id, stack_id] = 1
                            connection_map[stack_id, this_id] = 1
                    stack_for_connection_map[this_id] = segmentation_mask
                # '''

                orientation_angle = self.get_direction(segmentation_mask)
                if orientation_angle >= 0  and orientation_angle < 30:
                    directional_mask[:,:,0] += segmentation_mask
                elif orientation_angle >= 30  and orientation_angle < 60:
                    directional_mask[:,:,1] += segmentation_mask
                elif orientation_angle >= 60  and orientation_angle < 90:
                    directional_mask[:,:,2] += segmentation_mask
                elif orientation_angle >= 90  and orientation_angle < 120:
                    directional_mask[:,:,3] += segmentation_mask
                elif orientation_angle >= 120  and orientation_angle < 150:
                    directional_mask[:,:,4] += segmentation_mask
                elif orientation_angle >= 150  and orientation_angle < 180:
                    directional_mask[:,:,5] += segmentation_mask
                # import pdb; pdb.set_trace()
                # segmentation_mask = instance['dilate']

                # list object to arrary object with shape (N,2)
                end_points_label = np.reshape(np.asarray(end_points), (-1, 2))

                # h for horizontal, v for vertical
                # off_sets_prevs_h = np.reshape(np.asarray(off_sets_prevs), (-1, 2))[:, 0]
                # off_sets_prevs_v = np.reshape(np.asarray(off_sets_prevs), (-1, 2))[:, 1]


                control_points_label = np.reshape(np.asarray(control_points), (-1, 2))
                # control_points_label = np.vstack((control_points_label[:,1], control_points_label[:,0])).transpose()
                # import pdb; pdb.set_trace()
                # off_sets_nexts_h = np.reshape(np.asarray(off_sets_nexts), (-1, 2))[:, 0]
                # off_sets_nexts_v = np.reshape(np.asarray(off_sets_nexts), (-1, 2))[:, 1]

                #

                end_points_map = np.zeros(cropped_image_shape, dtype = np.float32)
                control_points_map = np.zeros(cropped_image_shape, dtype = np.float32)



            #     off_sets_nexts_map_h = np.zeros(cropped_image_shape, dtype = np.float32)
            #     off_sets_nexts_map_v = np.zeros(cropped_image_shape, dtype = np.float32)
            #     off_sets_prevs_map_h = np.zeros(cropped_image_shape, dtype = np.float32)
            #     off_sets_prevs_map_v = np.zeros(cropped_image_shape, dtype = np.float32)

                for idx, i in enumerate(end_points_label):
                    end_points_map[i[1], i[0]] = 1

            # #     off_sets_nexts_map_h[i[1], i[0]] = start_points_offsets_h[idx] * v_scale
            # #     off_sets_nexts_map_v[i[1], i[0]] = start_points_offsets_v[idx] * h_scale

                for idx, i in enumerate(control_points_label):
                # print(idx)
                    control_points_map[i[1], i[0]] = 1
            #         off_sets_nexts_map_h[i[1], i[0]] = off_sets_nexts_h[idx] #* h_scale
            #         off_sets_nexts_map_v[i[1], i[0]] = off_sets_nexts_v[idx] #* v_scale
            #         off_sets_prevs_map_h[i[1], i[0]] = off_sets_prevs_h[idx] #* h_scale
            #         off_sets_prevs_map_v[i[1], i[0]] = off_sets_prevs_v[idx] #* v_scale

                end_points_discs = get_keypoint_discs(end_points_label, cropped_image_shape, radius = self.disc_radius)
                control_points_discs = get_keypoint_discs(control_points_label, cropped_image_shape, radius = self.disc_radius)

                area_offset, canvas = compute_closest_control_point_offset(control_points_label, segmentation_mask, map_shape = cropped_image_shape)

                control_points_short_offset, canvas = compute_short_offsets(control_points_label, control_points_discs, map_shape = cropped_image_shape, radius =  self.disc_radius)
                end_points_short_offset, canvas = compute_short_offsets(end_points_label, end_points_discs, map_shape = cropped_image_shape, radius =  self.disc_radius)

                # control_points_prevs_offset, canvas = compute_mid_offsets(control_points_label, off_sets_prevs_map_h, off_sets_prevs_map_v, cropped_image_shape, control_points_discs)
                # control_points_nexts_offset, canvas = compute_mid_offsets(control_points_label, off_sets_nexts_map_h, off_sets_nexts_map_v, cropped_image_shape, control_points_discs)

                _, end_points_map = get_keypoint_discs_offset(end_points_label, end_points_map, cropped_image_shape, radius = self.disc_radius)
                _, control_points_map = get_keypoint_discs_offset(control_points_label, control_points_map, cropped_image_shape, radius =  self.disc_radius)


                #Intergrate to final keypoints disc map
                # end_points_map_final = end_points_map_final + end_points_map
                # end_points_map_label = end_points_map_label + end_points_map * this_id
                end_points_map_final = end_points_map_final + end_points_map
                # import pdb;pdb.set_trace()
                # end_point_map_temp = cv2.resize(end_points_map, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
                # end_points_map_label = np.concatenate((end_points_map_label, np.expand_dims(end_point_map_temp, axis = 0)), axis =0)
                end_points_map_label = end_points_map_label + end_points_map * this_id

                _, end_points_map_for_label = get_keypoint_discs_offset(end_points_label, end_points_map, cropped_image_shape, self.disc_radius)

                end_points_map_label[np.where(end_points_map_for_label > 0)] = this_id



                # control_points_map_final = control_points_map_final + control_points_map
                # control_points_map_label = control_points_map_label + control_points_map * this_id
                # control_points_map_final = control_points_map_final + control_points_map

                # control_point_map_temp = cv2.resize(control_points_map, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
                # control_points_map_label = np.concatenate((control_points_map_label, np.expand_dims(control_point_map_temp, axis = 0)), axis =0)
                # control_points_map_label = control_points_map_label + control_points_map * this_id
                _, control_points_map_for_label = get_keypoint_discs_offset(control_points_label, control_points_map, cropped_image_shape, self.disc_radius)

                # control_points_map_label[np.where(control_points_map_for_label > 0)] = this_id
                control_points_map_label[np.where(segmentation_mask > 0)] = this_id


                # short offset:
                end_points_off_sets_shorts_map_h_final = end_points_off_sets_shorts_map_h_final + end_points_short_offset[:,:,0]
                end_points_off_sets_shorts_map_v_final = end_points_off_sets_shorts_map_v_final + end_points_short_offset[:,:,1]

                control_points_off_sets_shorts_map_h_final = control_points_off_sets_shorts_map_h_final + control_points_short_offset[:,:,0]
                control_points_off_sets_shorts_map_v_final = control_points_off_sets_shorts_map_v_final + control_points_short_offset[:,:,1]

                # Mid offset:
                # off_sets_nexts_map_h_final = off_sets_nexts_map_h_final + control_points_nexts_offset[:,:,0]
                # off_sets_nexts_map_v_final = off_sets_nexts_map_v_final + control_points_nexts_offset[:,:,1]

                # off_sets_prevs_map_h_final = off_sets_prevs_map_h_final + control_points_prevs_offset[:,:,0]
                # off_sets_prevs_map_v_final = off_sets_prevs_map_v_final + control_points_prevs_offset[:,:,1]

                #area_offsets
                area_offset_map_h_final = area_offset_map_h_final + area_offset[:,:,0]
                area_offset_map_v_final = area_offset_map_v_final + area_offset[:,:,1]

                #segmentation mask
                segmentation_mask_final = segmentation_mask_final + segmentation_mask
                segmentation_mask_final_erode = segmentation_mask_final_erode + segmentation_mask_erode
                segmentation_mask_final_dilate = segmentation_mask_final_dilate + segmentation_mask_dialte

                outline_final = outline + outline_final
            
            intersection_areas = cv2.dilate((segmentation_mask_final > 1).astype("uint8"), np.ones((31,31))) * ((segmentation_mask_final>0) * 1.)
            intersection_areas_small = cv2.dilate((segmentation_mask_final > 1).astype("uint8"), np.ones((5,5))) * ((segmentation_mask_final>0) * 1.)

            # get touching area
            touching_and_overlapping_area_small = cv2.dilate((segmentation_mask_final_dilate > 1).astype("uint8"), np.ones((3,3))) * ((segmentation_mask_final_dilate>0) * 1.)

            outline_final = (outline_final > 0) * 1

            # get end_points map
            end_points_map_final  = 1.0 *(end_points_map_final > 0)
            #############################
            # write outline
            # binary_file_name = coco_fiber.loadImgs(img_id)[0]['binary']
            # os.makedirs(os.path.join(self.demo_folder,"outlines"),exist_ok=True)
            # cv2.imwrite(os.path.join(os.path.join(self.demo_folder,"outlines"),binary_file_name + "_outline_final.png" ), outline_final * 255.)
            ##############################
            
            outline_final = cv2.dilate(outline_final.astype('uint8'), (np.ones((2,2))))
            outline_final_inter = outline_final * intersection_areas
            # boundaries_line = cv2.dilate((segmentation_mask_final > 1).astype("uint8"), np.ones((3,3))) * ((segmentation_mask_final>0) * 1.)

            # intersection_areas_full = np.stack([intersection_areas, intersection_areas_small])
            intersection_areas_full = np.stack([intersection_areas, intersection_areas_small, touching_and_overlapping_area_small, end_points_map_final])
            
            if self.write_image:
                visualize_img = visualize_label_map(np.tile(image_visual, (3,1,1)).transpose(1,2,0), control_points_map_label)
                cv2.imwrite(os.path.join(self.demo_folder,str(img_id) + "control_points_map_label_full_instance.png" ), visualize_img)
                visualize_img = visualize_label_map(np.tile(image_visual, (3,1,1)).transpose(1,2,0), end_points_map_final)
                cv2.imwrite(os.path.join(self.demo_folder,str(img_id) + "end_points_map_final.png" ), visualize_img)
                visualize_img = visualize_label_map(np.tile(image_visual, (3,1,1)).transpose(1,2,0), touching_and_overlapping_area_small)
                cv2.imwrite(os.path.join(self.demo_folder,str(img_id) + "touching_and_overlapping_area_small.png" ), visualize_img)



            # cv2.imshow("outline_final", outline_final * 255.)
            # cv2.imshow("outline_final_final", segmentation_mask_final * 255.)
            # cv2.waitKey(0)
            # PUSH PULL ONLY INTERSECTION
            #'''
            if self.inter:
                segmentation_mask_final_inter = (cv2.dilate((segmentation_mask_final > 1).astype("uint8"), np.ones((31,31))) - (segmentation_mask_final>1) * 1.) * ((segmentation_mask_final>0) * 1.)
                segmentation_mask_final_inter_globs = (cv2.dilate((segmentation_mask_final > 1).astype("uint8"), np.ones((31,31)))) * ((segmentation_mask_final>0) * 1.)
                # cv2.imshow("segmentation_mask_final_inter", (cv2.dilate((segmentation_mask_final > 1).astype("uint8"), np.ones((11,11))) - (segmentation_mask_final>1) * 1.) * 255.)
                # cv2.waitKey(0)
                mask_for_phi = ~(segmentation_mask_final>0)
                phi = 1.0 * np.ones(segmentation_mask_final.shape)
                phi  = np.ma.MaskedArray(phi, mask_for_phi)
                if np.sum(segmentation_mask_final>1) != 0:
                    phi[np.where(segmentation_mask_final>1)] = 0
                    ok = skfmm.distance(phi, dx=1)
                    distance_values = ok.data
                    segmentation_mask_final_inter_globs = ((((distance_values < 35) * (distance_values > 0) * 1.0 + 1.0 * (segmentation_mask_final>1)) * (segmentation_mask_final > 0) ) > 0)* 1.0 
                inter_blobs_label = label(segmentation_mask_final_inter_globs)

                if self.write_image:
                    visualize_img = visualize_label_map(np.tile(image_visual, (3,1,1)).transpose(1,2,0), inter_blobs_label)
                    cv2.imwrite(os.path.join(self.demo_folder,str(img_id) + "inter_blobs_label.png" ), visualize_img)
                    
                    visualize_img = visualize_label_map(np.tile(image_visual, (3,1,1)).transpose(1,2,0), outline_final)
                    cv2.imwrite(os.path.join(self.demo_folder,str(img_id) + "outline_final.png" ), visualize_img)

                    segment_labels =  label(((segmentation_mask_final > 0).astype("uint8") * 1.0  - intersection_areas_small * 1.0 - outline_final) > 0)
                    visualize_img = visualize_label_map(np.tile(image_visual, (3,1,1)).transpose(1,2,0), segment_labels)
                    cv2.imwrite(os.path.join(self.demo_folder,str(img_id) + "segment_labels.png" ), visualize_img)
                    
                    visualize_img = visualize_label_map(np.tile(image_visual, (3,1,1)).transpose(1,2,0), (segmentation_mask_final > 0).astype("uint8"))
                    cv2.imwrite(os.path.join(self.demo_folder,str(img_id) + "segmentation_mask_final.png" ), visualize_img)
                    visualize_img = visualize_label_map(np.tile(image_visual, (3,1,1)).transpose(1,2,0), (segmentation_mask_final_erode > 0).astype("uint8"))
                    cv2.imwrite(os.path.join(self.demo_folder,str(img_id) + "segmentation_mask_final_erode.png" ), visualize_img)
                    
                    visualize_img = visualize_label_map(np.tile(image_visual, (3,1,1)).transpose(1,2,0), intersection_areas_small)
                    cv2.imwrite(os.path.join(self.demo_folder,str(img_id) + "intersection_areas_small.png" ), visualize_img)

                def normalize_include_neg_val(tag):

                    # Normalised [0,255]
                    normalised = 1.*(tag - np.min(tag))/np.ptp(tag).astype(np.float32)
            
                    return normalised

                ##############################################################################
                # normalized_distance_map = normalize_include_neg_val(ok.data) # skfmm distances are negative value
                # cmap = plt.get_cmap('jet')
                # rgba_img = cmap(normalized_distance_map)
                # cv2.imshow('ttt', rgba_img)
                # cv2.waitKey(0)
                # import pdb; pdb.set_trace()

                
        
                # visualize_img = visualize_label_map(np.tile(image, (3,1,1)).transpose(1,2,0), inter_blobs_label)
                # cv2.imshow('inter_blobs_label',visualize_img)
                # cv2.waitKey(0)
                ##############################################################################

                # control_points_map_label = (inter_blobs_label * 20 + control_points_map_label) * segmentation_mask_final_inter
                control_points_map_label = (inter_blobs_label * 20 + control_points_map_label) * segmentation_mask_final_inter_globs

                if self.write_image:
                    visualize_img = visualize_label_map(np.tile(image_visual, (3,1,1)).transpose(1,2,0), control_points_map_label)
                    cv2.imwrite(os.path.join(self.demo_folder,str(img_id) + "control_points_map_label.png" ), visualize_img)

                connection_map = np.zeros((500,500))
                blob_labels = np.unique(inter_blobs_label)
                
                for each_blob in blob_labels:
                    if each_blob == 0:
                        continue
                    connected_labels = np.unique(control_points_map_label[np.where(inter_blobs_label == each_blob)])
                    connected_labels = np.delete(connected_labels, 0)

                    for obja_idx in range(len(connected_labels)):
                        for objb_idx in range(obja_idx+1,len(connected_labels)):
                            connection_map[int(connected_labels[obja_idx]), int(connected_labels[objb_idx])] = 1
                            connection_map[int(connected_labels[objb_idx]), int(connected_labels[obja_idx])] = 1
            #'''
            # ==================================
            
            # cv2.imshow("segmentation_mask_final_inter", segmentation_mask_final_inter * 255.)
            # cv2.imshow("segmentation_mask_final>0", (segmentation_mask_final>0) * 255.)
            # cv2.imshow("segmentation_mask_final_inter_globs>0", (segmentation_mask_final_inter_globs>0) * 255.)
            # cv2.waitKey(0)
            segmentation_mask_final[segmentation_mask_final > 0] = 1
            outline_final[outline_final > 0] = 1
            # segmentation_mask_final = np.stack([segmentation_mask_final, outline_final])

            #end points
            # end_points_map_final = end_points_map_final * reverse_overlapping_area_mask
            # end_points_map_label = end_points_map_label * reverse_overlapping_area_mask

            end_points_map_final = cv2.resize(end_points_map_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            # end_points_map_label = end_points_map_label[1:, :,:]
            end_points_map_label = end_points_map_label #* reverse_overlapping_area_mask
            end_points_map_label = cv2.resize(end_points_map_label, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )

            #end points short offsets
            # end_points_off_sets_shorts_map_h_final = end_points_off_sets_shorts_map_h_final * reverse_overlapping_area_mask
            # end_points_off_sets_shorts_map_v_final = end_points_off_sets_shorts_map_v_final * reverse_overlapping_area_mask

            end_points_off_sets_shorts_map_h_final = cv2.resize(end_points_off_sets_shorts_map_h_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            end_points_off_sets_shorts_map_v_final = cv2.resize(end_points_off_sets_shorts_map_v_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )

            # Control points
            # control_points_map_final = control_points_map_final * reverse_overlapping_area_mask
            # control_points_map_label = control_points_map_label * reverse_overlapping_area_mask

            control_points_map_final = cv2.resize(control_points_map_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            # control_points_map_label = cv2.resize(control_points_map_label, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            # control_points_map_label = control_points_map_label[1:]
            control_points_map_label = control_points_map_label #* reverse_overlapping_area_mask
            control_points_map_label = cv2.resize(control_points_map_label, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )

            #import pdb;pdb.set_trace()
            #Control points short offsts
            control_points_off_sets_shorts_map_h_final = control_points_off_sets_shorts_map_h_final #* reverse_overlapping_area_mask
            control_points_off_sets_shorts_map_v_final = control_points_off_sets_shorts_map_v_final #* reverse_overlapping_area_mask

            control_points_off_sets_shorts_map_h_final = cv2.resize(control_points_off_sets_shorts_map_h_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            control_points_off_sets_shorts_map_v_final = cv2.resize(control_points_off_sets_shorts_map_v_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )

            #Next points offsets
            off_sets_nexts_map_h_final = off_sets_nexts_map_h_final #* reverse_overlapping_area_mask
            off_sets_nexts_map_v_final = off_sets_nexts_map_v_final #* reverse_overlapping_area_mask


            off_sets_nexts_map_h_final = cv2.resize(off_sets_nexts_map_h_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            off_sets_nexts_map_v_final = cv2.resize(off_sets_nexts_map_v_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )

            #previous point offsets
            off_sets_prevs_map_h_final = off_sets_prevs_map_h_final #* reverse_overlapping_area_mask
            off_sets_prevs_map_v_final = off_sets_prevs_map_v_final #* reverse_overlapping_area_mask


            off_sets_prevs_map_h_final = cv2.resize(off_sets_prevs_map_h_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            off_sets_prevs_map_v_final = cv2.resize(off_sets_prevs_map_v_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )

            #Area offsets map
            area_offset_map_h_final = area_offset_map_h_final #* reverse_overlapping_area_mask
            area_offset_map_v_final = area_offset_map_v_final #* reverse_overlapping_area_mask

            area_offset_map_h_final = cv2.resize(area_offset_map_h_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )
            area_offset_map_v_final = cv2.resize(area_offset_map_v_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_NEAREST )

            #Image
            # image = cv2.resize(image, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_AREA)

            # segmentation mask
            # segmentation_mask_final = cv2.resize(segmentation_mask_final, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_AREA)

            # Change scale
            end_points_off_sets_shorts_map_h_final *= h_scale
            end_points_off_sets_shorts_map_v_final *= v_scale

            control_points_off_sets_shorts_map_h_final *= h_scale
            control_points_off_sets_shorts_map_v_final *= v_scale

            off_sets_nexts_map_h_final *= h_scale
            off_sets_nexts_map_v_final *= v_scale
            off_sets_prevs_map_h_final *= h_scale
            off_sets_prevs_map_v_final *= v_scale

            area_offset_map_h_final *= h_scale
            area_offset_map_v_final *= v_scale

            image_visual = image[:,:,0] * 255
            if self.write_image:
                cv2.imwrite(os.path.join(self.demo_folder,str(img_id) + "w1.png" ), image[:,:,0] * 255.)
                cv2.imwrite(os.path.join(self.demo_folder,str(img_id) + "w2.png" ), image[:,:,1] * 255.)


            if self.debug:

                # show label
                # print (image.shape)
                # print (control_points_map_label.shape)
                cv2.imshow('image_visual a',image[:,:,0])
                cv2.imshow('image_visual b',image[:,:,1] )
                cv2.waitKey(0)

                visualize_img = visualize_label_map(np.tile(image_visual, (3,1,1)).transpose(1,2,0), control_points_map_label)
                cv2.imshow('control_points_label',visualize_img)
                cv2.waitKey(0)

                visualize_img = visualize_label_map(np.tile(segmentation_mask_final, (3,1,1)).transpose(1,2,0)*255., outline_final)
                cv2.imshow('outline',visualize_img)
                cv2.waitKey(0)

                # show control points
                # import pdb; pdb.set_trace()
                canvas = np.zeros_like(control_points_map_final)
                canvas = visualize_points(canvas, control_points_map_final)
                combined_show = draw_mask(image_visual, canvas, [255., 255., 255.])
                cv2.imshow('control points',combined_show)
                cv2.waitKey(0)

                # show end points
                canvas = np.zeros_like(control_points_map_final)
                canvas = visualize_points(canvas, end_points_map_final)
                combined_show = draw_mask(image_visual, canvas, [0., 255., 0.])
                cv2.imshow('end points',combined_show)
                cv2.waitKey(0)

                # show short offsets
                canvas = np.zeros_like(control_points_map_final)
                canvas = visualize_offset(canvas, end_points_off_sets_shorts_map_h_final, end_points_off_sets_shorts_map_v_final)
                combined_show = draw_mask(image_visual, canvas, [0., 255., 0.])
                cv2.imshow('short offsets',combined_show)
                cv2.waitKey(0)



                # show control points
                canvas = np.zeros_like(control_points_map_final)
                canvas = visualize_offset(canvas, control_points_off_sets_shorts_map_h_final, control_points_off_sets_shorts_map_v_final)
                combined_show = draw_mask(image_visual, canvas, [0., 255., 0.])
                cv2.imshow('short offsets',combined_show)
                cv2.waitKey(0)

                # show next
                canvas = np.zeros_like(control_points_map_final)
                canvas = visualize_offset(canvas, off_sets_nexts_map_h_final, off_sets_nexts_map_v_final)
                combined_show = draw_mask(image_visual, canvas, [0., 255., 0.])
                cv2.imshow('off_sets_nexts ',combined_show)
                cv2.waitKey(0)
                # show prev
                canvas = np.zeros_like(control_points_map_final)
                canvas = visualize_offset(canvas, off_sets_prevs_map_h_final, off_sets_prevs_map_v_final)
                combined_show = draw_mask(image_visual, canvas, [0., 255., 0.])
                cv2.imshow('off_sets_prevs',combined_show)
                cv2.waitKey(0)
                # show area offset
                canvas = np.zeros_like(control_points_map_final)
                canvas = visualize_offset(canvas, area_offset_map_h_final, area_offset_map_v_final)
                combined_show = draw_mask(image_visual, canvas, [0., 255., 0.])
                cv2.imshow('area_offset ',combined_show)
                cv2.waitKey(0)
                # show mask

                # canvas = np.zeros_like(off_sets_prevs_map_h_final)
                # canvas = visualize_points(canvas, segmentation_mask_final)
                # combined_show = draw_mask(image, canvas, [0., 255., 0.])
                # cv2.imshow('segmentation_mask_final offsets',combined_show)
                # cv2.waitKey(0)

                # cv2.imwrite('test/combined.png', combined_show)

        else:
            coco_fiber = self.fiber_coco
            img_id = self.img_ids[index]
            # img_id = coco_fiber.anns[ann_id]['image_id']
            # image_name = a['imgInfo']['file_path']
            # import pdb; pdb.set_trace()
            # img_path =  a['imgInfo']['file_path']

            
            w1_file_name = coco_fiber.loadImgs(img_id)[0]['w1']
            w2_file_name = coco_fiber.loadImgs(img_id)[0]['w2']
            binary_file_name = coco_fiber.loadImgs(img_id)[0]['binary']

            w1_file_path =  os.path.join(self.img_folder, w1_file_name)
            w2_file_path =  os.path.join(self.img_folder, w2_file_name)
            binary_path =  os.path.join(self.binary_folder, binary_file_name)

            image_w1 = cv2.imread(w1_file_path, cv2.COLOR_BGR2GRAY)
            non_crop_image_shape = image_w1.shape
            

            image_w2 = cv2.imread(w2_file_path, cv2.COLOR_BGR2GRAY)

            # image_w1 = normalize_min_max(image_w1, 0, np.max(image_w1))
            # image_w2 = normalize_min_max(image_w2, 0, np.max(image_w2))
            image_w1 = normalize_min_max(image_w1, 0, 4095)
            image_w2 = normalize_min_max(image_w2, 0, 3072)

            image_w1 = np.asarray(image_w1)
            image_w2 = np.asarray(image_w2)
            image_w1 = np.expand_dims(image_w1, 0)
            image_w2 = np.expand_dims(image_w2, 0)
            image_w1_w2 = np.concatenate((image_w1, image_w2), axis=0).transpose(1,2,0)

            # import pdb; pdb.set_trace()
        # print(image.shape)
            image = np.asarray(image_w1_w2)
            

            # img_info = self.coco_fiber.loadImgs(index)[0]
            # file_path = os.path.join(self.gt_image_root, img_info['file_name'])
            # image = imageio.imread(file_path)
            # input_shape = image.shape[0:2]
        # print(image.shape)
            # import pdb;pdb.set_trace()
            # image = np.asarray(image)
            # annIds = self.coco_fiber.getAnnIds(imgIds=img_info['id'], catIds=self.catIds)
            # anns = self.coco_fiber.loadAnns(annIds)
            # import pdb;pdb.set_trace()

        if len(image.shape) == 2:
                image = np.tile(image,(3,1,1)).transpose(1,2,0)
                image = np.asarray(image)
        if self.is_train:
            # image, points = self.data_augmentation(image, points, a['operation'])
            img = im_to_torch(image)  # CxHxW


            # Color dithering
            # img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            # img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            # img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        else:
            # image = cv2.resize(image, (self.out_res[1],self.out_res[0]), interpolation = cv2.INTER_AREA)
            img = im_to_torch(image)

        img = color_normalize(img, self.pixel_means)


        if self.is_train:
        # if True:

            control_target15 = generate_heatmap(control_points_map_final, self.cfg.gk15)
            control_target11 = generate_heatmap(control_points_map_final, self.cfg.gk11)
            control_target9 = generate_heatmap(control_points_map_final, self.cfg.gk9)
            control_target7 = generate_heatmap(control_points_map_final, self.cfg.gk7)

            end_target15 = generate_heatmap(end_points_map_final, self.cfg.gk15)
            end_target11 = generate_heatmap(end_points_map_final, self.cfg.gk11)
            end_target9 = generate_heatmap(end_points_map_final, self.cfg.gk9)
            end_target7 = generate_heatmap(end_points_map_final, self.cfg.gk7)

            targets15 = np.stack((control_target15,end_target15))
            targets11 = np.stack((control_target11,end_target11))
            targets9 = np.stack((control_target9,end_target9))
            targets7 = np.stack((control_target7,end_target7))
            segmentation_mask_final = segmentation_mask_final_erode
            ground_truth = [segmentation_mask_final, outline_final, control_points_map_final, end_points_map_final,
                            area_offset_map_h_final, area_offset_map_v_final,
                            off_sets_nexts_map_h_final,off_sets_nexts_map_v_final,
                            off_sets_prevs_map_h_final,off_sets_prevs_map_v_final,
                            control_points_off_sets_shorts_map_h_final, control_points_off_sets_shorts_map_v_final,
                            end_points_off_sets_shorts_map_h_final, end_points_off_sets_shorts_map_v_final
                            ]
         # control_points_map_final and end_points_map_final is not used if use control_points_map_targets and end_pointsmap_targets

            targets = torch.Tensor(ground_truth)

            # binary_targets = [torch.Tensor(targets15), torch.Tensor(targets11), torch.Tensor(targets9), torch.Tensor(targets7)]
            control_points_map_label = torch.Tensor(control_points_map_label)
            intersection_areas_full = torch.Tensor(intersection_areas_full)
            end_points_map_label = torch.Tensor(end_points_map_label)

            directional_mask = (directional_mask>0) * 1.0
            directional_mask = directional_mask.transpose(2,0,1)
            directional_mask = torch.Tensor(directional_mask)
            # import pdb; pdb.set_trace()
            # meta = {'index' : index, 'imgID' : a['imgInfo']['imgID'],
            meta = {'index' : index,
        # 'GT_bbox' : np.array([gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]]),
            }##, 'augmentation_details' : details}
        if self.is_train:

            return img, targets, end_points_map_label, control_points_map_label, intersection_areas_full, directional_mask, connection_map, meta
        else:
            # meta = {'index': index, 'imgID' : img_info['id'], 'img_path' : file_path, 'out_shape':self.out_res, 'input_shape':input_shape}
            meta = {'index' : index,
            'w1':w1_file_path,
            'w2':w2_file_path, 
            'bi':binary_path,
            'image_id':img_id
            }##, 'augmentation_details' : details}
            meta['det_scores'] = float(1)
            return img, meta

    def __len__(self):
        if self.is_train:
            return len(self.img_ids)
        else:
            # import pdb;pdb.set_trace()
            # return len(self.coco_ids)
            return len(self.img_ids)
            # return 8
