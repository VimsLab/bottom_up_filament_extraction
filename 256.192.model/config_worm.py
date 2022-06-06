import os
import os.path
import sys
import numpy as np
import cv2
import torch

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/data/synthetic/curves_pool_t_junction_zigzag_di_5/')
    img_dir = os.path.join(cur_dir,'/home/yliu/work/colab/work/fiberPJ/data/worm_data/BBBC010_v2_images/')
    binary_folder = '/home/yliu/work/colab/work/fiberPJ/data/worm_data/BBBC010_v1_foreground' 
    jason_file_p = 'worm_ann_keypoint_sequence.json'
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/colab/work/fiberPJ/DRIFT_Instace_segmentation_for_filaments/dataset/embedding_curve_coco_train_lstm_1000_512_kernel_5_step_30/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/for_pixel_embedding_512/images/')


    gt_path = os.path.join('/home/yliu/work/colab/work/fiberPJ/data/worm_data' , 'worm_ann_keypoint_sequence.json')
    # gt_path = os.path.join('/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'curve_coco_train_lstm_100_256_kernel_5.json')
    # gt_path = os.path.join('/home/yliu/work/colab/work/fiberPJ/DRIFT_Instace_segmentation_for_filaments/dataset/', 'embedding_curve_coco_train_lstm_1000_256_kernel_5_step_30.json')

    con_mask = True
    inter = True
    model = 'CPN50'
    # model = 'CPN101'
    batch_size = 2

    lr = 5e-3
    lr_gamma = 0.5
    lr_dec_epoch = list(range(50,200,20))

    weight_decay = 1e-5

    num_class = 2
    # img_path = os.path.join(root_dir, 'data', 'COCO2017', 'train2017')
    img_path = img_dir
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    bbox_extend_factor = (0.1, 0.15) # x, y

    # data augmentation setting
    scale_factor=(0.7, 1.35)
    rot_factor=45

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device = torch.device("cpu" )
    # data_shape = (608, 800)
    # output_shape = (608, 800)

    # data_shape = (448, 608)
    # output_shape = (448, 608)
    # data_shape = (256, 512)
    # output_shape = (64, 128)

    # tensorboard_path = './runs/radius_12_k_12_8_6_3'

    # tensorboard_path = './runs/radius_12_only'
    # batch_size = 1
    # data_shape = (576, 768)
    # output_shape = (576, 768)
    # info = 'Training with disc size 12 both keypoint and offset'

    ####################################
    ### CUDA_VISIBLE_DEVICES=0 python test.py --workers=12 -c checkpoint_longshort -t epoch50checkpoint
    # tensorboard_path = './runs/radius_12_in_384_out_192_short_long'
    # batch_size = 4
    # data_shape = (384, 496)
    # output_shape = (192, 256)
    # info = 'Training with data_shape = (384, 496) output_shape = (192, 256) No batch Norm last layer'
###########################################################################
    # tensorboard_path = './runs/radius_12_in_384_out_192_short_long_disc_scale_mid_offset'
    # batch_size = 4
    # data_shape = (384, 496)
    # output_shape = (192, 256)
    # info = 'Training with data_shape = (384, 496) output_shape = (192, 256) No batch Norm last layer'
 #############################################################################
    # tensorboard_path = './runs/radius_double_offset'
    # batch_size = 1
    # data_shape = (384, 512)
    # output_shape = (384, 512)
    # disc_radius = 20
    # info = 'Training with data_shape =(384, 512) output_shape = (384, 512) No batch Norm last layer'
    # ############################################################################
    # tensorboard_path = './runs/radius_double_offset_para_nobn_v2'
    # batch_size = 1
    # data_shape = (384, 512)
    # output_shape = (384, 512)
    # disc_radius = 20
    # info = 'Training with data_shape =(384, 512) output_shape = (384, 512) No batch Norm last layer'
    # #############################################################################
    # tensorboard_path = './runs/radius_double_offset_para_nobn_v2'
    # batch_size = 1
    # data_shape = (384, 512)
    # output_shape = (384, 512)
    # disc_radius = 20
    # info = 'Training with data_shape =(384, 512) output_shape = (384, 512) No batch Norm last layer'
    # # #############################################################################
    # tensorboard_path = './runs/mask_keypoints'

#### Jan 6 ###############
    # batch_size = 1
    # data_shape = (384, 512)
    # output_shape = (384, 512)

    # disc_radius = 20
    # info = 'Training with data_shape =(384, 512) output_shape = (384, 512) No batch Norm last layer'

#### smaller size

    data_shape = (512, 512)
    output_shape = (512, 512)
    
    disc_radius = 7
    info = 'Training with data_shape =(384, 512) output_shape = (384, 512) No batch Norm last layer'
    # tensorboard_path = './runs/radius_12_in_576_out_768_short_long_disc'
    # batch_size = 1
    # data_shape = (576, 768)
    # output_shape = (576, 768)
    # info = 'Training with data_shape = (384, 496) output_shape = (192, 256) No batch Norm last layer'

    crop_width = 0

    # disc_radius = 12

    gaussain_kernel = (7, 7)

    # disc_kernel_points_16 = np.ones((16,16))
    # disc_kernel_points_12 = np.ones((12,12))
    # disc_kernel_points_8 = np.ones((8,8))
    # disc_kernel_points_3 = np.ones((3,3))

    disc_kernel_points_16 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
    disc_kernel_points_12 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    disc_kernel_points_8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    disc_kernel_points_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    gk15 = (15, 15)
    gk11 = (11, 11)
    gk9 = (9, 9)
    gk7 = (7, 7)
    # folders = [x[0] for x in os.walk('/usa/yliu1/colab/data/instance_labeled_file/')]

    #
    # gt_path = []
    # for i in range(2, len(folders),3):
    #     # import pdb;pdb.set_trace()
    #     img_root = folders[i]
    #
    #     anno_root = folders[i + 1] + '/'
    #
    #     target_file = 'ann_keypoint_instance_mask_no_crop.json'
    #     if 'fb_second_100_test_common' in anno_root:
    #         continue
    #     if 'fb_fifth_common' in anno_root:
    #         continue
    #     gt_path.append(os.path.join(anno_root, target_file))
    #
    # gt_path.append(os.path.join('/usa/yliu1/colab/work/fiberPJ/data/fiber_labeled_data/', 'ann_keypoint_instance_mask_no_crop.json'))
# home/yliu/work/codeToBiomix/intersection_detection/dataset/curve_coco_train_lstm_100_256_kernel_3/          
    # gt_path = os.path.join('/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/data/synthetic/', 'curves_pool_t_junction_zigzag_di_5.json')
    # gt_path = os.path.join('/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/data/synthetic/', 'curves_pool_50_step.json')

    # gt_path = os.path.join('/home/yliu/work/colab/work/fiberPJ/DRIFT_Instace_segmentation_for_filaments/dataset/', 'embedding_curve_coco_train_lstm_1000_512_kernel_5_step_30.json')
    # gt_path = os.path.join('/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'for_pixel_embedding_512.json')
    reverse_flag = True
    # gt_path = os.path.join('/usa/yliu1/colab/work/fiberPJ/data/fiber_labeled_data/', 'ann_keypoint_instance_mask_no_crop.json')
    # gt_path = os.path.join('/usa/yliu1/colab/work/fiberPJ/data/fiber_labeled_data/', 'ann_keypoint_double_offset_test.json')
    # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'skel_5_non_coco_fifth.json')
    # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'skel_3_non_coco_fifth.json')
    # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'skel_1_non_coco_fifth.json')

cfg = Config()
add_pypath(cfg.root_dir)
add_pypath(os.path.join(cfg.root_dir, 'cocoapi/PythonAPI'))

