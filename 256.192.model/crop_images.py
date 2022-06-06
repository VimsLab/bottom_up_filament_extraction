import numpy as np
from patchify import patchify, unpatchify
import os
import imageio
import cv2

mt_files = os.listdir( '/home/yliu/work/data/zder_s/')
output_folder = '/home/yliu/work/data/zder_s_cropped_1024/'
os.makedirs(output_folder, exist_ok=True)

patch_size = 1024
for i in range(len(mt_files)):

    img_path = os.path.join('/home/yliu/work/data/zder_s/',mt_files[i])
    image = imageio.imread(img_path)
    if len(image.shape) > 2:
        image = image [:,:,0]
    print(image.shape)
    patches = patchify(image, (patch_size,patch_size), step=1024) # patch shape [2,2,3]
    for ii in range(patches.shape[0]):
        for jj in range(patches.shape[1]):
            output_path = os.path.join(output_folder, mt_files[i].split(".")[0] + "_" + str(ii) + "_"+str(jj)+".png")
            page = (patches[ii,jj, :,:] > 0.9 )* 255.
            page = patches[ii,jj, :,:] 
            cv2.imwrite(output_path, page)