import numpy as np
from patchify import patchify, unpatchify
import os
import imageio
import cv2

mt_files_feed_folder = '/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/p_ruberson/segments_result_map_di/'
mt_files = os.listdir( '/home/yliu/work/data/zder_s/')
output_folder = '/home/yliu/work/colab/work/fiberPJ/pytorch-cpn/256.192.model/demo_folder/outputs/p_ruberson/combined/'
os.makedirs(output_folder, exist_ok=True)

patch_size = 512
for i in range(len(mt_files)):

    img_path = os.path.join('/home/yliu/work/data/zder_s/',mt_files[i])
    image = imageio.imread(img_path)
    if len(image.shape) == 2:
        image = np.tile(image,(3,1,1)).transpose(1,2,0)
    print(image.shape)
    patches = patchify(image, (patch_size,patch_size,3), step=512) # patch shape [2,2,3]
    # import pdb; pdb.set_trace()
    for ii in range(patches.shape[0]):
        print(ii)
        for jj in range(patches.shape[1]):
            print(jj)
            feed_image = os.path.join(mt_files_feed_folder, mt_files[i].split(".")[0] + "_" + str(ii) + "_"+str(jj)+"_labeled_segments_with_direction.png")
            page = cv2.imread(feed_image)
            patches[ii,jj, 0, :,:, :] = page

    reconstructed_image = unpatchify(patches, image.shape)
    output_path = os.path.join(output_folder, mt_files[i])
    cv2.imwrite(output_path, reconstructed_image)