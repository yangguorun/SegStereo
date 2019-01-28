from __future__ import division
import os
import os.path
import string
import cv2
import numpy as np
import argparse
import sys
sys.path.insert(0, '../python')
sys.path.insert(0, '../tools/disp_tools')
import caffe
from lib import flowlib as fl

parser = argparse.ArgumentParser(description='SegStereo and ResNetCorr Caffe implementation.')
parser.add_argument('--model_weights',    type = str,   help='model weights',   default = './ResNetCorr/ResNetCorr_SRC_pretrain.caffemodel')
parser.add_argument('--model_deploy',     type = str,   help='deploy prototxt', default = './ResNetCorr/ResNetCorr_deploy.prototxt')
parser.add_argument('--data',             type = str,   help='data directory (containing left, right)', default = '../data/KITTI')
parser.add_argument('--result',           type = str,   help='result directory', default = './ResNetCorr/result/KITTI')
parser.add_argument('--gpu',              type = int,   help='GPU ID', default = 0)
parser.add_argument('--colorize',         dest = 'color_disp', action = 'store_true')
parser.add_argument('--no-colorize',      dest = 'color_disp', action = 'store_false')
parser.add_argument('--evaluate',         dest = 'eval_disp',  action = 'store_true')
parser.add_argument('--no-evaluate',      dest = 'eval_disp',  action = 'store_false')
args = parser.parse_args()

if __name__ == '__main__':
    # 1. Indicate the date and result path
    # 1.1 Check the model and prototxt
    if(not os.path.exists(args.model_weights)):
        raise BaseException('Caffe model weights does not exist!')
    if(not os.path.exists(args.model_deploy)):
        raise BaseException('Deploy prototxt does not exist!')
    # 1.2 Indicate the data directories
    left_image_dir  = os.path.join(args.data, 'left')     # left image directory
    right_image_dir = os.path.join(args.data, 'right')    # right image directory
    gt_disp_dir     = os.path.join(args.data, 'gt_disp')        # ground-truth disparity directory
    if(not os.path.exists(left_image_dir)):
        raise BaseException('Data directory does not exist!')
    if(not os.path.exists(right_image_dir)):
        raise BaseException('Data directory does not exist!')
    if(args.eval_disp):
        if (not os.path.exists(gt_disp_dir)):
            raise BaseException('You want to evaluate, but ground-truth disparity directory not exists!')
    # 1.3 Indicate the result directories
    pred_disp_dir   = os.path.join(args.result, 'disp')
    color_disp_dir  = os.path.join(args.result, 'color')
    error_disp_dir  = os.path.join(args.result, 'error')
    if (not os.path.exists(pred_disp_dir)):
        os.makedirs(pred_disp_dir)
    if (args.color_disp):
        if (not os.path.exists(color_disp_dir)):
            os.makedirs(color_disp_dir)
    if (args.eval_disp):
        if (not os.path.exists(error_disp_dir)):
            os.makedirs(error_disp_dir)

    # 2. Predict disparity maps
    # 2.1 Caffe model initialize
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.model_deploy, args.model_weights, caffe.TEST)
    adapted_height  = net.blobs['data'].height
    adapted_width   = net.blobs['data'].width
    mean_ = [103.939, 116.779, 123.68]
    img_list = sorted(os.listdir(left_image_dir))
    # 2.2 Forward image and save disparity maps
    for img_fn in img_list:
        # 2.2.1 Read the images
        left_image_path = os.path.join(left_image_dir, img_fn)
        right_image_path = os.path.join(right_image_dir, img_fn)
        img0 = cv2.imread(left_image_path)
        img1 = cv2.imread(right_image_path)

        # 2.2.2 Resize the input image
        target_height   = img0.shape[0]
        target_width    = img0.shape[1]
        if (img0.shape[0] != adapted_height) or (img0.shape[1] != adapted_width):
            img0 = cv2.resize(img0, (adapted_width, adapted_height))
        if (img1.shape[0] != adapted_height) or (img1.shape[1] != adapted_width):
            img1 = cv2.resize(img1, (adapted_width, adapted_height))
        img_blob = np.array([img0, img1]).transpose(0, 3, 1, 2)   # num, height, width, channel -> num, channel, height, width
        img_blob = img_blob.astype(np.float32)

        # 2.2.3 Substract the mean value and fill the blobs
        for c in range(0, 3):
            img_blob[0][c] = img_blob[0][c] - mean_[c]
            img_blob[1][c] = img_blob[1][c] - mean_[c]
        input_dict = {}
        input_dict['data'] = img_blob

        # 2.2.4 Network forward
        print('Processing ' + img_fn)
        out = net.forward(**input_dict)
        disp_blob = np.squeeze(out[net.outputs[0]])

        # 2.2.5 Write disp to file
        disp_blob = fl.resize_disp(disp_blob, target_width, target_height)
        disp_path = os.path.join(pred_disp_dir, img_fn)
        disp_img = disp_blob * 256.0
        disp_img = disp_img.astype(np.uint16)
        cv2.imwrite(disp_path, disp_img)
        if (args.color_disp):
            color_img   = fl.disp_to_color(disp_blob)
            color_path  = os.path.join(color_disp_dir, img_fn)
            cv2.imwrite(color_path, color_img)

    # 3. Evaluate
    if (args.eval_disp):
        for img_fn in img_list:
            pd_disp_path = os.path.join(pred_disp_dir, img_fn)
            gt_disp_path = os.path.join(gt_disp_dir, img_fn)
            pd_disp = fl.read_disp_png(pd_disp_path)
            gt_disp = fl.read_disp_png(gt_disp_path)
            (single_epe, single_err) = fl.evaluate_kitti_disp(gt_disp, pd_disp)
            error_img = fl.disp_error_to_color(gt_disp, pd_disp)
            error_path = os.path.join(error_disp_dir, img_fn)
            cv2.imwrite(error_path, error_img)
            error_line = 'Disp ' + img_fn + ' EPE = %.4f' + ' Err = %.4f';
            print (error_line % (single_epe, single_err))








