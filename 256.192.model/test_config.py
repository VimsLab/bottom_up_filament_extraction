import os
import os.path
import sys
import numpy as np

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')
    
    # img_dir = os.path.join(cur_dir,'/usa/yliu1/colab/work/fiberPJ/data/fiber_labeled_data/instance_labeled_file/')
    # img_dir = os.path.join(cur_dir,'/usa/yliu1/colab/data/instance_labeled_file/fb_fourth_common/img')
    img_dir = os.path.join(cur_dir,'/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/data/synthetic/curves_pool_t_junction_zigzag_di_5/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/curve_coco_train_lstm_100_512_kernel_5/images/')
    img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/curve_coco_test_lstm_100_256_kernel_5/images/')
    img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/ave_num5curve_coco_test_lstm_100_256_kernel_11_step_60/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/ave_num5curve_coco_test_lstm_100_256_kernel_7_step_60/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/curve_coco_test_lstm_100_512_kernel_7/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/curve_coco_test_lstm_100_256_kernel_5/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/ave_num5curve_coco_test_lstm_100_256_kernel_11_step_60/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/ave_num5curve_coco_test_lstm_100_256_kernel_7_step_60/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/ave_num5curve_coco_test_lstm_100_256_kernel_11_step_60/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/curve_coco_test_lstm_100_256_kernel_7/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/ave_num5curve_coco_test_lstm_100_256_kernel_3_step_60/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/curve_coco_test_lstm_100_256_kernel_11/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/curve_coco_train_lstm_100_256_kernel_5/images/')


    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/curve_coco_test_lstm_100_512_kernel_5/images/')
    # img_dir = os.path.join(cur_dir,'/home/yliu/work/codeToBiomix/intersection_detection/dataset/curve_coco_test_lstm_100_512_kernel_3/images/')

    orient_folder = os.path.join("/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/six_output/", "ave_num5curve_coco_test_lstm_100_256_kernel_11_step_60")
    # orient_folder = os.path.join("/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/six_output/", "curve_coco_test_lstm_100_512_kernel_7")
    # orient_folder = os.path.join("/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/six_output/", "curve_coco_test_lstm_100_256_kernel_5")
    # orient_folder = os.path.join("/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/six_output/", "ave_num5curve_coco_test_lstm_100_256_kernel_11_step_60")
    # orient_folder = os.path.join("/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/six_output/", "pruberson_images_1024")

    result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/ave_num5curve_coco_test_lstm_100_256_kernel_11_step_60_ca_oa/"
    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/p_ruberson_1024/"
    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/mt_12/"
    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/mt_12_ca/"

    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/ave_num5curve_coco_test_lstm_100_256_kernel_11_step_60/"
    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/ave_num5curve_coco_test_lstm_100_256_kernel_7_step_60/"
    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/ave_num5curve_coco_test_lstm_100_256_kernel_11_step_60/"
    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/curve_coco_test_lstm_100_256_kernel_7/"
    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/ave_num5curve_coco_test_lstm_100_256_kernel_3_step_60/"
    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/ave_num5curve_coco_test_lstm_100_256_kernel_7_step_60/"
    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/curve_coco_test_lstm_100_512_kernel_7/"
    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/curve_coco_test_lstm_100_256_kernel_5_with_ca_thin/"
    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/ave_num5curve_coco_test_lstm_100_256_kernel_11_step_60/"
    
    # result_folder = "/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/microtubules_mt_12/"

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


    use_GT_bbox = False
    if use_GT_bbox:
        gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'COCO_2017_val.json')
    else:
        # if False, make sure you have downloaded the val_dets.json and place it into annotation folder
          # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'skel_5_non_coco_fifth.json')
          # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'testing_multi_v3.json')
        # gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'ann_keypoint_short_long_offset.json')
        # gt_path = os.path.join('/usa/yliu1/colab/data/instance_labeled_file/fb_fourth_common/anno'
        #     , 'ann_keypoint_double_offset_test.json')
        # gt_path = os.path.join('/usa/yliu1/colab/data/instance_labeled_file/fb_450_train_common/anno', 'ann_common.json')
        gt_path = os.path.join('/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/data/synthetic/', 'curves_pool_t_junction_zigzag_di_5.json')
        gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'curve_coco_test_lstm_100_256_kernel_11.json')
        gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'curve_coco_test_lstm_100_256_kernel_5.json')
        
        # gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'curve_coco_test_lstm_100_256_kernel_7.json')

        # gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'ave_num5curve_coco_test_lstm_100_256_kernel_5_step_60.json')
        # gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'ave_num5curve_coco_test_lstm_100_256_kernel_7_step_60.json')
        # gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'ave_num5curve_coco_test_lstm_100_256_kernel_11_step_60.json')
        # gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'ave_num5curve_coco_test_lstm_100_256_kernel_3_step_60.json')
        # gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'curve_coco_train_lstm_100_256_kernel_5.json')


        gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'curve_coco_test_lstm_100_512_kernel_5.json')
        # gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'ave_num5curve_coco_test_lstm_100_256_kernel_7_step_60.json')
        gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'ave_num5curve_coco_test_lstm_100_256_kernel_11_step_60.json')
        # gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'ave_num5curve_coco_test_lstm_100_256_kernel_5_step_60.json')

    # gt_image_root = '/usa/yliu1/colab/work/fiberPJ/data/fiber_labeled_data/instance_labeled_file/'
    # gt_file = 'fb_coco_style_fifth.json'
    # gt_root = '/usa/yliu1/colab/work/fiberPJ/data/fiber_labeled_data/'

    # gt_image_root = '/usa/yliu1/colab/data/instance_labeled_file/fb_second_100_test_common/img'
    # gt_root = '/usa/yliu1/colab/data/instance_labeled_file/fb_second_100_test_common/anno'
    # gt_file = 'ann_common.json'
    # ori_gt_path = os.path.join( '/usa/yliu1/colab/data/instance_labeled_file/fb_second_100_test_common/anno/' ,gt_file)
    # ori_gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'curve_coco_train_lstm_100_512_kernel_5.json')
    # ori_gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'curve_coco_test_lstm_100_256_kernel_5.json')
    # ori_gt_path = os.path.join( '/home/yliu/work/codeToBiomix/intersection_detection/dataset/', 'curve_coco_train_lstm_100_256_kernel_5.json')
    gt_file = gt_path

    # ori_gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'skel_5_non_coco_fifth.json')
    # # ori_gt_path = os.path.join('/home/yiliu/work/fiberPJ/data/fiber_labeled_data/', 'ann_keypoint_short_long_offset.json')
    # ori_gt_path = os.path.join('/usa/yliu1/colab/data/instance_labeled_file/fb_fourth_common/anno'
    #         , 'ann_keypoint_double_offset_test.json')
    # ori_gt_path = os.path.join('/usa/yliu1/colab/work/fiberPJ/data/fiber_labeled_data/', 'ann_common.json')
    # ori_gt_path = os.path.join('/usa/yliu1/colab/data/instance_labeled_file/fb_450_train_common/anno', 'ann_common.json')

cfg = Config()
add_pypath(cfg.root_dir)
add_pypath(os.path.join(cfg.root_dir, 'cocoapi/PythonAPI'))
