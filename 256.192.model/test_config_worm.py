import os
import os.path
import sys
from unittest import result
import numpy as np

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')

    img_dir = os.path.join(cur_dir,'/home/yliu/work/colab/work/fiberPJ/data/worm_data/BBBC010_v2_images/')
    binary_folder = '/home/yliu/work/colab/work/fiberPJ/data/worm_data/BBBC010_v1_foreground' 
    jason_file_p = 'worm_ann_keypoint_sequence.json'
    result_folder = "good_worm_result"
    result_folder = "worm_direction"
    result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/worm_for_demo/"
    orient_folder = os.path.join("/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/six_output/", "worm")
    
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/colab/work/fiberPJ/DRIFT_Instace_segmentation_for_filaments/dataset/embedding_curve_coco_train_lstm_1000_512_kernel_5_step_30/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/for_pixel_embedding_512/images/')


    # gt_path = os.path.join('/home/yliu/work/colab/work/fiberPJ/data/worm_data' , 'worm_ann_keypoint_sequence_test_rebuttal.json')
    inter = True
    model = 'CPN50' # option 'CPN50', 'CPN101'  

    num_class = 2
    # img_path = os.path.join(root_dir, 'data', 'COCO2017', 'val2017')
    img_path = img_dir
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    bbox_extend_factor = (0.1, 0.15) # x, y

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    # data_shape = (256, 192)
    # output_shape = (64, 48)
    # data_shape = (608, 800)
    # output_shape = (608, 800)
    # data_shape = (448, 608)
    # output_shape = (448, 608)


    # data_shape = (576, 768)
    # output_shape = (576, 768)
##################################################################
    # data_shape = (384, 496)
    # output_shape = (192, 256)

##################################################################
    data_shape = (256, 256)
    output_shape = (256, 256)
    disc_radius = 7
    PEAK_THRESH = 0.001
##################################################################
    disc_kernel_points_16 = np.ones((16,16))
    disc_kernel_points_12 = np.ones((12,12))
    disc_kernel_points_8 = np.ones((8,8))
    disc_kernel_points_3 = np.ones((3,3))
    reverse_flag = True

    gk15 = (15, 15)
    gk11 = (11, 11)
    gk9 = (9, 9)
    gk7 = (7, 7)

    crop_width = 25

    # gt_image_root = '/usa/yliu1/colab/work/fiberPJ/data/fiber_labeled_data/instance_labeled_file/'
    # gt_file = 'fb_coco_style_fifth.json'
    # gt_root = '/usa/yliu1/colab/work/fiberPJ/data/fiber_labeled_data/'
    data_folder_test = '/home/yliu/work/colab/work/fiberPJ/data/worm_data' 
    jason_file_test ='worm_ann_keypoint_sequence_test_one_way.json'

    gt_path = os.path.join( '/home/yliu/work/colab/work/fiberPJ/data/worm_data', 'worm_ann_keypoint_sequence_test_one_way.json')

    # ori_gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'skel_5_non_coco_fifth.json')
    # # ori_gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'ann_keypoint_short_long_offset.json')
    # ori_gt_path = os.path.join('/usa/yliu1/colab/data/instance_labeled_file/fb_fourth_common/anno'
    #         , 'ann_keypoint_double_offset_test.json')
    # ori_gt_path = os.path.join('/usa/yliu1/colab/work/fiberPJ/data/fiber_labeled_data/', 'ann_common.json')
    # ori_gt_path = os.path.join('/usa/yliu1/colab/data/instance_labeled_file/fb_450_train_common/anno', 'ann_common.json')

cfg = Config()
add_pypath(cfg.root_dir)
add_pypath(os.path.join(cfg.root_dir, 'cocoapi/PythonAPI'))
