from matplotlib.path import Path

import scipy.stats as st
import matplotlib.patches as patches
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import json

from skimage.morphology import skeletonize
from PIL import Image, ImageDraw
from tqdm import tqdm
from get_intersection_and_endpoint import get_skeleton_endpoint, get_skeleton_intersection_and_endpoint, get_skeleton_intersection

import sys
sys.path.insert(0, '/home/yliu/work/colab/work/fiberPJ/data/')
from fiber_tools.common.util.cv2_util import pologons_to_mask, mask_to_pologons
def create_curves(width, height, num_cp, num_points, L, deform ):

    im = np.zeros((width, height))
    # random angle to rotate
    an_rot_st = 0
    an_rot = 180
    angle = np.deg2rad(an_rot_st -random.randint(0, an_rot))
    # angle = np.deg2rad(90)
    rot_mat = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rot_mat = np.asarray(rot_mat)

    #random translation
    trans = random.randint(0, round(width/3))
    trans_mat = [[trans*np.sin(angle)],[trans*np.cos(angle)]]

    #rand om deformations
    d =  np.random.randn(num_cp)*deform
    xp = np.linspace(-L/2,L/2,num_cp)

    pp = interpolate.splrep(xp,d);

    #generate points
    x = np.linspace(-L/2,L/2,num_points)
    y = interpolate.splev(x, pp);

    points = np.stack((x,y))
    pts = np.dot(rot_mat,points)

    # import pdb; pdb.set_trace()
    pts = pts + np.tile(trans_mat,(1, num_points))

    #points to rasterize in an image
    im_pts = np.minimum(np.maximum(0,np.rint(pts+width/2 - 1)),width - 1)
    im_pts = im_pts.astype('int32')

    index_to_del = np.where(im_pts[0] == 0)
    im_pts = np.delete(im_pts, index_to_del, 1)


    index_to_del = np.where(im_pts[0] == im.shape[0] - 1 )
    im_pts = np.delete(im_pts, index_to_del, 1)

    index_to_del = (np.where(im_pts[1] == 0))
    im_pts = np.delete(im_pts, index_to_del, 1)

    index_to_del = (np.where(im_pts[1] == im.shape[1] - 1))
    im_pts = np.delete(im_pts, index_to_del, 1)
    import copy
    im_pts_contour = copy.deepcopy(im_pts)
    im_pts_contour = np.array(im_pts_contour)

    im_pts = tuple(map(tuple,im_pts))
#

    # import pdb; pdb.set_trace()


    im_pts_contour = [a for a in zip(im_pts_contour[0,:],im_pts_contour[1,:])]

    # import pdb; pdb.set_trace()
    def rm_order(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]
    im_pts_contour = rm_order(im_pts_contour)

    im_pts_contour = np.array(im_pts_contour)
    # import pdb; pdb.set_trace()
    im[im_pts] = 1
    im[0 , :] = 0
    im[im.shape[0] - 1, :] = 0
    im[: , im.shape[1] - 1] = 0
    im[: , 0] = 0
    # plt.imshow(im)
    # plt.show()


    return im, im_pts_contour


def main():
    debug = False
    gt_file = os.path.join('../data/synthetic', 'curves_pool_50_step.json')

    # directory_curve_pool = './dataset/test_pool'
    # directory_curve_pool = '../data/synthetic/curves_pool_t_junction_zigzag_di_5'
    directory_curve_pool = '../data/synthetic/curves_pool_50_step'

    if not os.path.exists(directory_curve_pool):
        os.makedirs(directory_curve_pool)

    number_of_images = 20000
    width = 256
    height = 256

    num_cp = 10
    num_points = 10000
    L = 500
    deform = 20

    train_data = []

    for id_img in tqdm(range(number_of_images)):

        single_data = {}
        img_info = {}

        instances = []
        num_of_curves = random.randint(1,4)
        if random.randint(0,8) < 3:
            kernel = np.ones((3,3),np.uint8)
        else:
            kernel = np.ones((5,5),np.uint8)
        input_image = np.zeros((width, height))
        input_image_dilate = np.zeros((width, height))

        prev_skel_im_exsit = False

        canvas = np.zeros((width, height))
        for ii in range(num_of_curves):
            end_points = []
            control_points = []

            off_sets_prev = []
            off_sets_next = []

            off_sets_next_double = []
            off_sets_prev_double = []
            instance_curve = {}

            im, im_pts = create_curves(width, height, num_cp, num_points, L, deform)

            im_pts = np.vstack((im_pts[:,1], im_pts[:,0]))


            im_pts = np.expand_dims(im_pts.transpose(), axis=1)

            # import pdb; pdb.set_trace()
            # cv2.circle(im, (im_pts[1][0],im_pts[0][0]) , 5, 1, 1)
            # cv2.circle(im, (im_pts[1][-1],im_pts[0][-1]) , 5, 1, 1)
            # cv2.imshow('ty', im)
            # cv2.waitKey(0)
            # im = skeletonize(im)
            im = im > 0
            im = np.asarray(im, dtype='uint8')
            current_skel_endpoint = get_skeleton_endpoint(im)

            im_dilate = cv2.dilate(im, kernel)
            instance_curve['skel'] =  mask_to_pologons(im)
            instance_curve['dilate'] = mask_to_pologons(im_dilate)

            input_image_dilate = input_image_dilate + im_dilate
            input_image = input_image + im

            # im = im * 1.

            contours, hierarchy = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

            # find the longest contour.
            longest = 0
            ########################################################################
            canvas_debug = np.zeros(im.shape)
            #######################################################################

            for idx, contour in enumerate(contours):
                #####################################################################
                # canvas_debug = cv2.drawContours(canvas_debug, contours, idx, 1, 1)
                # cv2.imshow('t', canvas_debug)
                # cv2.waitKey(0)
                #######################################################################
                if cv2.arcLength(contour, False) > cv2.arcLength(contours[longest], False):
                    longest = idx
            try:
                longest_contour = contours[longest]
            except:
                import pdb; pdb.set_trace()

            step = 50

            # longest_contour_sampled = longest_contour[0 : int(len(longest_contour)/2): step]
            # import pdb; pdb.set_trace()
            longest_contour_sampled = im_pts[0 : int(len(im_pts)): step]
            longest_contour_sampled = np.concatenate((longest_contour_sampled,im_pts[-1,np.newaxis]))
            longest_contour = im_pts
            if debug:
                canvas = cv2.drawContours(canvas, contours, longest, 1, 1)
                import pdb; pdb.set_trace()
                for i in range(longest_contour_sampled.shape[0]):

                    cv2.circle(canvas, (longest_contour_sampled[i][0][0],longest_contour_sampled[i][0][1]) , 5, 1, 1)

                cv2.circle(canvas, (longest_contour_sampled[0][0][0],longest_contour_sampled[0][0][1]) , 4, 1, 1)
                # cv2.circle(canvas, (longest_contour[-1][0][0],longest_contour[-1][0][1]) , 5, 1, 1)
                cv2.imshow('t', canvas)

            for pt in range(len(longest_contour_sampled)):
                if pt == (len(longest_contour_sampled) - 1):
                    control_points.append(longest_contour_sampled[pt][0][0])
                    control_points.append(longest_contour_sampled[pt][0][1])

                    off_sets_next.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[-1][0][0])
                    off_sets_next.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[-1][0][1])


                    off_sets_prev.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 1][0][0] )
                    off_sets_prev.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt - 1][0][1] )

                    off_sets_next_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[-1][0][0])
                    off_sets_next_double.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[-1][0][1])

                    try:
                        off_sets_prev_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 2][0][0])
                        off_sets_prev_double.append(longest_contour_sampled[pt][0][1]  - longest_contour_sampled[pt - 2][0][1])
                    except:
                        print('except1')
                        off_sets_prev_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 1][0][0])
                        off_sets_prev_double.append(longest_contour_sampled[pt][0][1]  - longest_contour_sampled[pt - 1][0][1])
                elif pt == 0:
                    end_points.append(longest_contour_sampled[pt][0][0])
                    end_points.append(longest_contour_sampled[pt][0][1])
                    control_points.append(longest_contour_sampled[pt][0][0])
                    control_points.append(longest_contour_sampled[pt][0][1])

                    off_sets_next.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt + 1][0][0])
                    off_sets_next.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt + 1][0][1])
                    off_sets_prev.append(0)
                    off_sets_prev.append(0)

                    try:
                        off_sets_next_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt + 2][0][0])
                        off_sets_next_double.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt + 2][0][1])
                    except:
                        print('except2')
                        import pdb; pdb.set_trace()
                        print(len(longest_contour_sampled))
                        off_sets_next_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt + 1][0][0])
                        off_sets_next_double.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt + 1][0][1])

                    off_sets_prev_double.append(0)
                    off_sets_prev_double.append(0)
                else:
                    control_points.append(longest_contour_sampled[pt][0][0])
                    control_points.append(longest_contour_sampled[pt][0][1])
                    off_sets_next.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt + 1][0][0] )
                    off_sets_next.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt + 1][0][1] )
                    off_sets_prev.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 1][0][0] )
                    off_sets_prev.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt - 1][0][1] )

                    if pt == (len(longest_contour_sampled) - 2):
                        off_sets_next_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[-1][0][0])
                        off_sets_next_double.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[-1][0][1])
                        try:
                            off_sets_prev_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 2][0][0] )
                            off_sets_prev_double.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt - 2][0][1] )
                        except:
                            print('except3')
                            off_sets_prev_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 1][0][0] )
                            off_sets_prev_double.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt - 1][0][1] )

                    elif pt == 1:
                        try:
                            off_sets_next_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt + 2][0][0])
                            off_sets_next_double.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt + 2][0][1])
                        except:
                            print('except4')
                            off_sets_next_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt + 1][0][0])
                            off_sets_next_double.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt + 1][0][1])

                        off_sets_prev_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 1][0][0])
                        off_sets_prev_double.append(longest_contour_sampled[pt][0][1]  - longest_contour_sampled[pt - 1][0][1])
                    else:
                        try:
                            off_sets_next_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt + 2][0][0] )
                            off_sets_next_double.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt + 2][0][1] )
                        except:
                            print('except5')
                            off_sets_next_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt + 1][0][0] )
                            off_sets_next_double.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt + 1][0][1] )
                        try:
                            off_sets_prev_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 2][0][0] )
                            off_sets_prev_double.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt - 2][0][1] )
                        except:
                            print('except6')
                            off_sets_prev_double.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 2][0][0] )
                            off_sets_prev_double.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt - 2][0][1] )

            end_points.append(longest_contour_sampled[-1][0][0])
            end_points.append(longest_contour_sampled[-1][0][1])
            # control_points.append(longest_contour_sampled[-1][0][0])
            # control_points.append(longest_contour_sampled[-1][0][1])
            off_sets_next.append(0)
            off_sets_next.append(0)
            off_sets_next_double.append(0)
            off_sets_next_double.append(0)
            # off_sets_prev.append(longest_contour[int(len(longest_contour)/2)][0][0] - longest_contour_sampled[-1][0][0] )
            # off_sets_prev.append(longest_contour[int(len(longest_contour)/2)][0][1] - longest_contour_sampled[-1][0][1] )

            # try:
            #     off_sets_prev_double.append(longest_contour[int(len(longest_contour)/2)][0][0] - longest_contour_sampled[-2][0][0] )
            #     off_sets_prev_double.append(longest_contour[int(len(longest_contour)/2)][0][1] - longest_contour_sampled[-2][0][1] )
            # except:
            #     off_sets_prev_double.append(longest_contour[int(len(longest_contour)/2)][0][0] - longest_contour_sampled[-1][0][0] )
            #     off_sets_prev_double.append(longest_contour[int(len(longest_contour)/2)][0][1] - longest_contour_sampled[-1][0][1] )

            for i in range(len(end_points)):
                end_points[i] = int(end_points[i])
                # start_points_offsets[i] = int(start_points_offsets[i])

            for i in range(len(control_points)):
                control_points[i] = int(control_points[i])
                off_sets_prev[i] = int(off_sets_prev[i])
                off_sets_next[i] = int(off_sets_next[i])
                off_sets_next[i] = int(off_sets_next[i])
                off_sets_next_double[i] = int(off_sets_next_double[i])
                off_sets_prev_double[i] = int(off_sets_prev_double[i])

            instance_curve['id'] = ii + 1
            instance_curve['end_points'] = end_points

            instance_curve['control_points'] = control_points
            instance_curve['off_sets_prev'] = off_sets_prev
            instance_curve['off_sets_next'] = off_sets_next
            instance_curve['off_sets_next_double'] = off_sets_next_double
            instance_curve['off_sets_prev_double'] = off_sets_prev_double
            instances.append(instance_curve)
            ##################

        if debug:
            for pt in range(0, len(control_points), 2):

                curr = (control_points[pt], control_points[pt + 1])
                next_pt = (control_points[pt] - off_sets_next[pt], control_points[pt + 1] - off_sets_next[pt + 1])

                cv2.arrowedLine(canvas, curr, next_pt, 1, 1)

            cv2.imshow('t',canvas)
            cv2.waitKey(0)

        if debug:
            for pt in range(0, len(control_points), 2):

                curr = (control_points[pt], control_points[pt + 1])
                next_pt = (control_points[pt] - off_sets_prev[pt], control_points[pt + 1] - off_sets_prev[pt + 1])

                cv2.arrowedLine(canvas, curr, next_pt, 1, 1)

            cv2.imshow('t5',canvas)
            cv2.waitKey(0)

        # if debug:
        #     for pt in range(0, len(control_points), 2):
        #
        #         curr = (control_points[pt], control_points[pt + 1])
        #         next_pt = (control_points[pt] - off_sets_prev_double[pt], control_points[pt + 1] - off_sets_prev_double[pt + 1])
        #
        #         cv2.arrowedLine(canvas, curr, next_pt, 1, 1)
        #     #
        #     cv2.imshow('t5',canvas)
        #     cv2.waitKey(0)
        #
        # if debug:
        #     for pt in range(0, len(control_points), 2):
        #
        #         curr = (control_points[pt], control_points[pt + 1])
        #         next_pt = (control_points[pt] - off_sets_next_double[pt], control_points[pt + 1] - off_sets_next_double[pt + 1])
        #
        #         cv2.arrowedLine(canvas, curr, next_pt, 1, 1)
        #
        #     cv2.imshow('t5',canvas)
        #     cv2.waitKey(0)




        input_image = input_image > 0
        input_image = input_image * 1.0



        img_info ['file_name'] = str(id_img) + '.png'
        img_info ['file_path'] = directory_curve_pool

        single_data['instances'] = instances

        single_data ['img_info'] = img_info

        input_image = cv2.dilate(input_image, kernel)
        img_debug = Image.fromarray(input_image * 255.)
        img_debug = np.asarray(img_debug)
        # img_debug_draw = ImageDraw.Draw(img_debug)
        # import pdb; pdb.set_trace()
        ##############################################
        # for a in range(0, len(control_points), 2):
        #
        #     img_debug = cv2.circle(img_debug, (control_points[a], control_points[a + 1]), 3, (200,255,200), 10)
        #     print(a)
        # import pdb; pdb.set_trace()
        # cv2.imshow('t5', img_debug)
        # cv2.waitKey(0)
        # for a in instances:
        #     end_points = a['endpoints']
        #     for b in end_points:
        #         # import pdb; pdb.set_trace()
        #         img_debug = cv2.circle(img_debug, (b[0], b[1]), 5, (200,0,0), 1)
        # I = np.asarray(img_debug)
        # cv2.imshow('t', I)
        # cv2.waitKey(0)
        ##################################################

        train_data.append(single_data)
        input_image = Image.fromarray(input_image * 255)
        # print(directory_curve_pool + '/' + str(id_img) + '.png')
        input_image.convert('L').save(directory_curve_pool + '/' + str(id_img) + '.png')

    print('saving transformed annotation...')
    with open(gt_file,'w') as wf:
        json.dump(train_data, wf)
        print('done')


if __name__ == '__main__':
    main()



# n = 4 # Number of possibly sharp edges
# r = .7 # magnitude of the perturbation from the unit circle,
# # should be between 0 and 1
# N = n*3+1 # number of points in the Path
# # There is the initial point and 3 points per cubic bezier curve. Thus, the curve will only pass though n points, which will be the sharp edges, the other 2 modify the shape of the bezier curve

# # angles = np.linspace(0,2*np.pi,N)
# # angles = np.linspace(0,np.pi,N)

# lengths = np.linspace(0, 100, N)
# codes = np.full(N,Path.CURVE4)
# codes[0] = Path.MOVETO

# max_num_deform_points = 4
# random_deform_points_index = []

# for i in range(max_num_deform_points):
#     random_deform_points_index.append(random.randint(0, N))


# noises = st.norm.rvs(loc = 3,scale = 10,size= max_num_deform_points)
# y = np.zeros(np.size(lengths))
# import pdb; pdb.set_trace()

# for i in range(len(random_deform_points_index)):
#     idx = random_deform_points_index[i]
#     y[idx] = y[idx] + noises[i]

# import pdb; pdb.set_trace()
# verts = np.stack((lengths, y), axis=-1)

# # verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r*np.random.random(N)+1-r)[:,None]
# # verts[-1,:] = verts[0,:] # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
# path = Path(verts, codes)

# # fig = plt.figure()
# ax = fig.add_subplot(111)
# patch = patches.PathPatch(path, facecolor='none', lw=2)
# ax.add_patch(patch)

# ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
# ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
# ax.axis('off') # removes the axis to leave only the shape

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats as st
# import random
# from scipy.optimize import curve_fit

# #number of data points
# n = 50

# #function
# def func(data):
#     return 10*np.exp(-0.5*data)

# def fit(data, a, b):
#     return a*np.exp(b*data)

# #define interval
# a = 0
# b = 4

# #generate random data grid
# x = []
# for i in range(0, n):
#     x.append(random.uniform(a, b))
# x.sort()

# #noise-free data points
# yclean = []
# for i in range(0, n):
#     yclean.append(func(x[i]))

# #define mean, standard deviation, sample size for 0 noise and 1 errors
# mu0 = 0
# sigma0 = 0.4
# mu1 = 0.5
# sigma1 = 0.02

# #generate noise
# noise = st.norm.rvs(mu0, sigma0, size = n)
# y = yclean + noise
# yerr = st.norm.rvs(mu1, sigma1, size = n)

# #now x and y is your data
# #define analytic x and y
# xan = np.linspace(a, b, n)
# yan = []
# for i in range(0, n):
#     yan.append(func(xan[i]))

# #now estimate fit parameters
# #initial guesses
# x0 = [1.0, 1.0]
# #popt are list of optimal coefficients, pcov is covariation matrix
# popt, pcov = curve_fit(fit, x, y, x0, yerr)

# fity = []
# for i in range(0, n):
#     fity.append(fit(xan[i], *popt))

# print ('function used to generate is 10 * exp( -0.5 * x )')
# print ('fit function is', popt[0], '* exp(', popt[1], '* x )')

# #plotting data and analytical function
# plt.rc("figure", facecolor="w")
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif',size = 16)
# plt.title("Data", fontsize=20)
# plt.errorbar(x, y, yerr, fmt='o')
# plt.plot(xan, yan, 'r')
# plt.plot(xan, fity, 'g')
# plt.show()
