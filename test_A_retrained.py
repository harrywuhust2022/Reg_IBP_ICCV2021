from __future__ import division
import os
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader

from src import utils

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy

import torchvision.transforms.functional as F
from matplotlib import cm as CM

import torch.backends.cudnn as cudnn
import torch

import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import math
from torchvision import datasets, transforms
import glob
import cv2
from tqdm import tqdm
import math
from torchvision import datasets, transforms
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
import torch.utils.data as data_utils
# from utils_certify import *
from scipy.ndimage.interpolation import zoom
from PIL import Image
import scipy.ndimage
from torchvision.transforms import ToPILImage
import os
from matplotlib import pyplot as plt
import numpy as np
import h5py
from certify_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("using cuda: ", format(device))


def bound_propagation(model, initial_bound, epsilon_try):
    low1, upp1 = initial_bound
    low2, upp2 = initial_bound
    low3, upp3 = initial_bound

    for layer, module in model.DME.branch1.named_modules():

        if isinstance(module, nn.Conv2d):
            # torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
            c = (low1 + upp1) / 2
            r = (upp1 - low1) / 2

            c_ = nn.functional.conv2d(c, weight=module.weight, bias=module.bias, stride=module.stride,
                                      padding=module.padding, dilation=module.dilation, groups=module.groups)

            r_ = nn.functional.conv2d(r, weight=abs(module.weight), bias=None, stride=module.stride,
                                      padding=module.padding, dilation=module.dilation, groups=module.groups)

            low1 = c_ - r_
            upp1 = c_ + r_

        elif isinstance(module, nn.MaxPool2d):
            low_ = nn.functional.max_pool2d(low1, kernel_size=module.kernel_size, stride=module.stride,
                                            padding=module.padding, dilation=module.dilation,
                                            ceil_mode=module.ceil_mode, return_indices=module.return_indices)
            upp_ = nn.functional.max_pool2d(upp1, kernel_size=module.kernel_size, stride=module.stride,
                                            padding=module.padding, dilation=module.dilation,
                                            ceil_mode=module.ceil_mode, return_indices=module.return_indices)

            low1, upp1 = low_, upp_

        elif isinstance(module, nn.BatchNorm2d):
            low_ = nn.functional.batch_norm(low1, num_features=module.num_features,
                                            eps=module.eps, momentum=module.momentum, affine=True)

            upp_ = nn.functional.batch_norm(upp1, num_features=module.num_features,
                                            eps=module.eps, momentum=module.momentum, affine=True)

            low1, upp1 = low_, upp_

        elif isinstance(module, nn.ReLU):
            # bounded ReLU 原始的BReLU: min(max(x,0),1)
            # revise: min(max(x,0),epsilon)

            # '''
            low_ = torch.clamp(nn.functional.relu(low1, inplace=True), min=-epsilon_try, max=255)
            upp_ = torch.clamp(nn.functional.relu(upp1, inplace=True), min=-epsilon_try, max=255)
            # '''

            # '''
            # normal relu
            #low_ = nn.functional.relu(low1, inplace=True)
            #upp_ = nn.functional.relu(upp1, inplace=True)
            #X1 = nn.functional.relu(X1, inplace=True)
            # '''

            low1, upp1 = low_, upp_

    for layer, module in model.DME.branch2.named_modules():

        if isinstance(module, nn.Conv2d):
            # torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
            c = (low2 + upp2) / 2
            r = (upp2 - low2) / 2

            c_ = nn.functional.conv2d(c, weight=module.weight, bias=module.bias, stride=module.stride,
                                      padding=module.padding, dilation=module.dilation, groups=module.groups)

            r_ = nn.functional.conv2d(r, weight=abs(module.weight), bias=None, stride=module.stride,
                                      padding=module.padding, dilation=module.dilation, groups=module.groups)

            low2 = c_ - r_
            upp2 = c_ + r_

        elif isinstance(module, nn.MaxPool2d):
            low_ = nn.functional.max_pool2d(low2, kernel_size=module.kernel_size, stride=module.stride,
                                            padding=module.padding, dilation=module.dilation,
                                            ceil_mode=module.ceil_mode, return_indices=module.return_indices)
            upp_ = nn.functional.max_pool2d(upp2, kernel_size=module.kernel_size, stride=module.stride,
                                            padding=module.padding, dilation=module.dilation,
                                            ceil_mode=module.ceil_mode, return_indices=module.return_indices)

            low2, upp2 = low_, upp_

        elif isinstance(module, nn.BatchNorm2d):
            low_ = nn.functional.batch_norm(low2, num_features=module.num_features,
                                            eps=module.eps, momentum=module.momentum, affine=True)

            upp_ = nn.functional.batch_norm(upp2, num_features=module.num_features,
                                            eps=module.eps, momentum=module.momentum, affine=True)

            low2, upp2 = low_, upp_

        elif isinstance(module, nn.ReLU):
            # bounded ReLU 原始的BReLU: min(max(x,0),1)
            # revise: min(max(x,0),epsilon)

            # '''
            low_ = torch.clamp(nn.functional.relu(low2, inplace=True), min=-epsilon_try, max=255)
            upp_ = torch.clamp(nn.functional.relu(upp2, inplace=True), min=-epsilon_try, max=255)
            # '''

            # '''
            # normal relu
            #low_ = nn.functional.relu(low2, inplace=True)
            #upp_ = nn.functional.relu(upp2, inplace=True)
            #X2 = nn.functional.relu(X2, inplace=True)
            # '''

            low2, upp2 = low_, upp_

    for layer, module in model.DME.branch3.named_modules():

        if isinstance(module, nn.Conv2d):
            # torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
            c = (low3 + upp3) / 2
            r = (upp3 - low3) / 2

            c_ = nn.functional.conv2d(c, weight=module.weight, bias=module.bias, stride=module.stride,
                                      padding=module.padding, dilation=module.dilation, groups=module.groups)

            r_ = nn.functional.conv2d(r, weight=abs(module.weight), bias=None, stride=module.stride,
                                      padding=module.padding, dilation=module.dilation, groups=module.groups)

            low3 = c_ - r_
            upp3 = c_ + r_

        elif isinstance(module, nn.MaxPool2d):
            low_ = nn.functional.max_pool2d(low3, kernel_size=module.kernel_size, stride=module.stride,
                                            padding=module.padding, dilation=module.dilation,
                                            ceil_mode=module.ceil_mode, return_indices=module.return_indices)
            upp_ = nn.functional.max_pool2d(upp3, kernel_size=module.kernel_size, stride=module.stride,
                                            padding=module.padding, dilation=module.dilation,
                                            ceil_mode=module.ceil_mode, return_indices=module.return_indices)

            low3, upp3 = low_, upp_

        elif isinstance(module, nn.BatchNorm2d):
            low_ = nn.functional.batch_norm(low3, num_features=module.num_features,
                                            eps=module.eps, momentum=module.momentum, affine=True)

            upp_ = nn.functional.batch_norm(upp3, num_features=module.num_features,
                                            eps=module.eps, momentum=module.momentum, affine=True)

            low3, upp3 = low_, upp_

        elif isinstance(module, nn.ReLU):
            # bounded ReLU 原始的BReLU: min(max(x,0),1)
            # revise: min(max(x,0),epsilon)

            # '''
            low_ = torch.clamp(nn.functional.relu(low3, inplace=True), min=-epsilon_try, max=255)
            upp_ = torch.clamp(nn.functional.relu(upp3, inplace=True), min=-epsilon_try, max=255)
            # '''

            '''
            # normal relu
            low_ = nn.functional.relu(low3, inplace=True)
            upp_ = nn.functional.relu(upp3, inplace=True)
            X3 = nn.functional.relu(X3, inplace=True)
            '''

            low3, upp3 = low_, upp_

    low = torch.cat((low1, low2, low3), 1)
    upp = torch.cat((upp1, upp2, upp3), 1)

    for layer, module in model.DME.fuse.named_modules():

        if isinstance(module, nn.Conv2d):

            c = (low + upp) / 2
            r = (upp - low) / 2

            c_ = nn.functional.conv2d(c, weight=module.weight, bias=module.bias, stride=module.stride,
                                      padding=module.padding, dilation=module.dilation, groups=module.groups)

            r_ = nn.functional.conv2d(r, weight=abs(module.weight), bias=None, stride=module.stride,
                                      padding=module.padding, dilation=module.dilation, groups=module.groups)

            low = c_ - r_
            upp = c_ + r_

        elif isinstance(module, nn.MaxPool2d):
            low_ = nn.functional.max_pool2d(low, kernel_size=module.kernel_size, stride=module.stride,
                                            padding=module.padding, dilation=module.dilation,
                                            ceil_mode=module.ceil_mode, return_indices=module.return_indices)
            upp_ = nn.functional.max_pool2d(upp, kernel_size=module.kernel_size, stride=module.stride,
                                            padding=module.padding, dilation=module.dilation,
                                            ceil_mode=module.ceil_mode, return_indices=module.return_indices)

            low, upp = low_, upp_

        elif isinstance(module, nn.BatchNorm2d):
            low_ = nn.functional.batch_norm(low, num_features=module.num_features,
                                            eps=module.eps, momentum=module.momentum, affine=True)

            upp_ = nn.functional.batch_norm(upp, num_features=module.num_features,
                                            eps=module.eps, momentum=module.momentum, affine=True)

            low, upp = low_, upp_

        elif isinstance(module, nn.ReLU):
            # bounded ReLU 原始的BReLU: min(max(x,0),1)
            # revise: min(max(x,0),epsilon)

            # '''
            low_ = torch.clamp(nn.functional.relu(low, inplace=True), min=-epsilon_try, max=255)
            upp_ = torch.clamp(nn.functional.relu(upp, inplace=True), min=-epsilon_try, max=255)

            # '''

            '''
            # normal relu
            low_ = nn.functional.relu(low, inplace=True)
            upp_ = nn.functional.relu(upp, inplace=True)
            X = nn.functional.relu(X, inplace=True)
            '''

            low, upp = low_, upp_

    return low, upp


def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_cuda:
        v = (torch.from_numpy(x).type(dtype)).to(device)
    if is_training:
        v = Variable(v, requires_grad=True, volatile=False)
    else:
        v = Variable(v, requires_grad=False, volatile=True)
    return v


def L2_bound_propagation(model, x, epsilon):
    first_y1 = model.DME.branch1[0](x)
    first_y2 = model.DME.branch2[0](x)
    first_y3 = model.DME.branch3[0](x)

    weight_1 = torch.norm(model.DME.branch1[0].conv.weight, p=2, dim=(1,2,3))
    weight_2 = torch.norm(model.DME.branch2[0].conv.weight, p=2, dim=(1,2,3))
    weight_3 = torch.norm(model.DME.branch3[0].conv.weight, p=2, dim=(1,2,3))

    weight_1 = weight_1.cpu().detach().numpy()
    weight_1 = weight_1.reshape(16,1,1)
    weight_1 = torch.from_numpy(weight_1).to(device)

    bound_1 = (first_y1 - weight_1 * epsilon, first_y1 + weight_1 * epsilon)

    weight_2 = weight_2.cpu().detach().numpy()
    weight_2 = weight_2.reshape(20,1,1)
    weight_2 = torch.from_numpy(weight_2).to(device)

    bound_2 = (first_y2 - weight_2 * epsilon, first_y2 + weight_2 * epsilon)

    weight_3 = weight_3.cpu().detach().numpy()
    weight_3 = weight_3.reshape(24,1,1)
    weight_3 = torch.from_numpy(weight_3).to(device)

    bound_3 = (first_y3 - weight_3 * epsilon, first_y3 + weight_3 * epsilon)

    low1, upp1 = bound_1
    low2, upp2 = bound_2
    low3, upp3 = bound_3

    # from second layer to start IBP

    for layer, module in model.DME.branch1.named_modules():# layer: 0.conv,  0.relu
        if (layer == '0.conv') or (layer == '0.relu'):
            low1, upp1 = low1, upp1
        else:
            if isinstance(module, nn.Conv2d):

                c = (low1 + upp1) / 2
                r = (upp1 - low1) / 2

                c_ = nn.functional.conv2d(c, weight=module.weight, bias=module.bias, stride=module.stride,
                                          padding=module.padding, dilation=module.dilation, groups=module.groups)

                r_ = nn.functional.conv2d(r, weight=abs(module.weight), bias=None, stride=module.stride,
                                          padding=module.padding, dilation=module.dilation, groups=module.groups)

                low1 = c_ - r_
                upp1 = c_ + r_

            elif isinstance(module, nn.MaxPool2d):
                low_ = nn.functional.max_pool2d(low1, kernel_size=module.kernel_size, stride=module.stride,
                                                padding=module.padding, dilation=module.dilation,
                                                ceil_mode=module.ceil_mode, return_indices=module.return_indices)
                upp_ = nn.functional.max_pool2d(upp1, kernel_size=module.kernel_size, stride=module.stride,
                                                padding=module.padding, dilation=module.dilation,
                                                ceil_mode=module.ceil_mode, return_indices=module.return_indices)

                low1, upp1 = low_, upp_

            elif isinstance(module, nn.BatchNorm2d):
                low_ = nn.functional.batch_norm(low1, num_features=module.num_features,
                                                eps=module.eps, momentum=module.momentum, affine=True)

                upp_ = nn.functional.batch_norm(upp1, num_features=module.num_features,
                                                eps=module.eps, momentum=module.momentum, affine=True)

                low1, upp1 = low_, upp_

            elif isinstance(module, nn.ReLU):

                low_ = torch.clamp(nn.functional.relu(low1, inplace=True), min=-epsilon, max=255)
                upp_ = torch.clamp(nn.functional.relu(upp1, inplace=True), min=-epsilon, max=255)

                low1, upp1 = low_, upp_

    for layer, module in model.DME.branch2.named_modules():# layer: 0.conv,  0.relu
        if (layer == '0.conv') or (layer == '0.relu'):
            low2, upp2 = low2, upp2
        else:
            if isinstance(module, nn.Conv2d):
                # torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
                c = (low2 + upp2) / 2
                r = (upp2 - low2) / 2

                c_ = nn.functional.conv2d(c, weight=module.weight, bias=module.bias, stride=module.stride,
                                          padding=module.padding, dilation=module.dilation, groups=module.groups)

                r_ = nn.functional.conv2d(r, weight=abs(module.weight), bias=None, stride=module.stride,
                                          padding=module.padding, dilation=module.dilation, groups=module.groups)

                low2 = c_ - r_
                upp2 = c_ + r_

            elif isinstance(module, nn.MaxPool2d):
                low_ = nn.functional.max_pool2d(low2, kernel_size=module.kernel_size, stride=module.stride,
                                                padding=module.padding, dilation=module.dilation,
                                                ceil_mode=module.ceil_mode, return_indices=module.return_indices)
                upp_ = nn.functional.max_pool2d(upp2, kernel_size=module.kernel_size, stride=module.stride,
                                                padding=module.padding, dilation=module.dilation,
                                                ceil_mode=module.ceil_mode, return_indices=module.return_indices)

                low2, upp2 = low_, upp_

            elif isinstance(module, nn.BatchNorm2d):
                low_ = nn.functional.batch_norm(low2, num_features=module.num_features,
                                                eps=module.eps, momentum=module.momentum, affine=True)

                upp_ = nn.functional.batch_norm(upp2, num_features=module.num_features,
                                                eps=module.eps, momentum=module.momentum, affine=True)

                low2, upp2 = low_, upp_

            elif isinstance(module, nn.ReLU):

                low_ = torch.clamp(nn.functional.relu(low2, inplace=True), min=-epsilon, max=255)
                upp_ = torch.clamp(nn.functional.relu(upp2, inplace=True), min=-epsilon, max=255)

                low2, upp2 = low_, upp_

    for layer, module in model.DME.branch3.named_modules():# layer: 0.conv,  0.relu
        if (layer == '0.conv') or (layer == '0.relu'):
            low3, upp3 = low3, upp3

        else:
            if isinstance(module, nn.Conv2d):
                # torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
                c = (low3 + upp3) / 2
                r = (upp3 - low3) / 2

                c_ = nn.functional.conv2d(c, weight=module.weight, bias=module.bias, stride=module.stride,
                                          padding=module.padding, dilation=module.dilation, groups=module.groups)

                r_ = nn.functional.conv2d(r, weight=abs(module.weight), bias=None, stride=module.stride,
                                          padding=module.padding, dilation=module.dilation, groups=module.groups)

                low3 = c_ - r_
                upp3 = c_ + r_

            elif isinstance(module, nn.MaxPool2d):
                low_ = nn.functional.max_pool2d(low3, kernel_size=module.kernel_size, stride=module.stride,
                                                padding=module.padding, dilation=module.dilation,
                                                ceil_mode=module.ceil_mode, return_indices=module.return_indices)
                upp_ = nn.functional.max_pool2d(upp3, kernel_size=module.kernel_size, stride=module.stride,
                                                padding=module.padding, dilation=module.dilation,
                                                ceil_mode=module.ceil_mode, return_indices=module.return_indices)

                low3, upp3 = low_, upp_

            elif isinstance(module, nn.BatchNorm2d):
                low_ = nn.functional.batch_norm(low3, num_features=module.num_features,
                                                eps=module.eps, momentum=module.momentum, affine=True)

                upp_ = nn.functional.batch_norm(upp3, num_features=module.num_features,
                                                eps=module.eps, momentum=module.momentum, affine=True)

                low3, upp3 = low_, upp_

            elif isinstance(module, nn.ReLU):

                low_ = torch.clamp(nn.functional.relu(low3, inplace=True), min=-epsilon, max=255)
                upp_ = torch.clamp(nn.functional.relu(upp3, inplace=True), min=-epsilon, max=255)

                low3, upp3 = low_, upp_

    low = torch.cat((low1, low2, low3), 1)
    upp = torch.cat((upp1, upp2, upp3), 1)

    for layer, module in model.DME.fuse.named_modules():

        if isinstance(module, nn.Conv2d):

            c = (low + upp) / 2
            r = (upp - low) / 2

            c_ = nn.functional.conv2d(c, weight=module.weight, bias=module.bias, stride=module.stride,
                                      padding=module.padding, dilation=module.dilation, groups=module.groups)

            r_ = nn.functional.conv2d(r, weight=abs(module.weight), bias=None, stride=module.stride,
                                      padding=module.padding, dilation=module.dilation, groups=module.groups)

            low = c_ - r_
            upp = c_ + r_

        elif isinstance(module, nn.MaxPool2d):
            low_ = nn.functional.max_pool2d(low, kernel_size=module.kernel_size, stride=module.stride,
                                            padding=module.padding, dilation=module.dilation,
                                            ceil_mode=module.ceil_mode, return_indices=module.return_indices)
            upp_ = nn.functional.max_pool2d(upp, kernel_size=module.kernel_size, stride=module.stride,
                                            padding=module.padding, dilation=module.dilation,
                                            ceil_mode=module.ceil_mode, return_indices=module.return_indices)

            low, upp = low_, upp_

        elif isinstance(module, nn.BatchNorm2d):
            low_ = nn.functional.batch_norm(low, num_features=module.num_features,
                                            eps=module.eps, momentum=module.momentum, affine=True)

            upp_ = nn.functional.batch_norm(upp, num_features=module.num_features,
                                            eps=module.eps, momentum=module.momentum, affine=True)

            low, upp = low_, upp_

        elif isinstance(module, nn.ReLU):

            low_ = torch.clamp(nn.functional.relu(low, inplace=True), min=-epsilon, max=255)
            upp_ = torch.clamp(nn.functional.relu(upp, inplace=True), min=-epsilon, max=255)

            low, upp = low_, upp_

    return low, upp


def epoch_robust_bound(data_loader, model, device, epsilon_try):

    certify_mae = 0.0
    certify_mse = 0.0
    certify_mae_pixel = 0.0
    certify_mse_pixel = 0.0

    model.eval()

    dtype = torch.FloatTensor
    '''
    if not os.path.exists('./certify_MCNN'):
        os.mkdir('./certify_MCNN')

    if not os.path.exists('./certify_MCNN/lower'):
        os.mkdir('./certify_MCNN/lower')

    if not os.path.exists('./certify_MCNN/upper'):
        os.mkdir('./certify_MCNN/upper')

    if not os.path.exists('./certify_MCNN/output'):
        os.mkdir('./certify_MCNN/output')
    '''

    with torch.no_grad():
        for blob in data_loader:
            full_imgname = blob['fname']
            im_data = blob['data']  # (1,1,704,1024)
            gt_data = blob['gt_density']

            img_var = np_to_variable(im_data, is_cuda=True, is_training=False)
            target_var = np_to_variable(gt_data, is_cuda=True, is_training=False)

            # **************certify bound calculation******************************************************************

            X = (torch.from_numpy(im_data).type(dtype)).to(device)
            # initial_bound = (X - epsilon_try, X + epsilon_try)
            lower_bound, upper_bound = L2_bound_propagation(model, X, epsilon_try)

            # ***************normal output*********************************************************

            density_map_var = model(img_var, target_var)
            output = density_map_var.data.detach().cpu().numpy()

            # *************** 存 图 **********************************************************************
            '''
            lower_bound_store = lower_bound.detach().cpu().numpy()
            upper_bound_store = upper_bound.detach().cpu().numpy()

            lower_store = lower_bound_store[0][0]
            upper_store = upper_bound_store[0][0]

            plt.imsave('./certify_MCNN/upper/{}'.format(full_imgname), upper_store, format='png', cmap=plt.cm.jet)
            plt.imsave('./certify_MCNN/lower/{}'.format(full_imgname), lower_store, format='png', cmap=plt.cm.jet)
            plt.imsave('./certify_MCNN/output/{}'.format(full_imgname), (output.squeeze(0)).squeeze(0), format='png',
                       cmap=plt.cm.jet)
            '''
# ***************** MAE MSE Certify / Normal ****************************************************************

            gt_count = np.sum(gt_data)

            low_np = lower_bound.data.detach().cpu().numpy()
            upp_np = upper_bound.data.detach().cpu().numpy()

            # 验证lower_bound 和 upper_bound, 输出都是0的话代表是成功的
            # print(np.sum(output < low_np))
            # print(np.sum(output > upp_np))

            low_GT = gt_data - low_np
            upp_GT = upp_np - gt_data

            max_distance = cal_distance(low_GT, upp_GT)

            certify_mae_pixel += np.sum(max_distance)
            certify_mse_pixel += (np.sum(max_distance)*np.sum(max_distance))

            lower = lower_bound.detach().cpu().numpy()
            upper = upper_bound.detach().cpu().numpy()

            et_lower = np.sum(lower)
            et_upper = np.sum(upper)

            if abs(et_upper - gt_count) > abs(et_lower - gt_count):
                et_certify = et_upper
            else:
                et_certify = et_lower

            certify_mae += abs(et_certify - gt_count)
            certify_mse += ((et_certify - gt_count) * (et_certify - gt_count))

        certify_mae = certify_mae / data_loader.get_num_samples()
        certify_mse = np.sqrt(certify_mse / data_loader.get_num_samples())

        certify_mae_pixel = certify_mae_pixel / data_loader.get_num_samples()
        certify_mse_pixel = np.sqrt(certify_mse_pixel / data_loader.get_num_samples())

        print("Certify_MAE_tight: " + str(certify_mae))
        print("Certify_MSE_tight: " + str(certify_mse))

        print("Certify_MAE_pixel: " + str(certify_mae_pixel))
        print("Certify_MSE_pixel: " + str(certify_mse_pixel))

        '''
        f = open('Certify_Test.txt', 'w')
        f.write('Certify_MAE_tight: %0.2f, Certify_MSE_tight: %0.2f \n' % (certify_mae, certify_mse))
        f.write('Certify_MAE_pixel: %0.2f, Certify_MSE_pixel: %0.2f \n' % (certify_mae_pixel, certify_mse_pixel))
        f.close()
        '''


def pgd_test(data_loader, model, epsilon_try):
    adv_mae = 0.0
    adv_mse = 0.0

    criterion = nn.MSELoss(size_average=False).to(device)

    # L_inf attack config
    # eps=1/255, alpha=1/255,  iter=40
    eps = epsilon_try  # 1/255 约等于0.004
    step_size = 0.001
    iterations = 20

    if not os.path.exists('./certify_MCNN'):
        os.mkdir('./certify_MCNN')

    if not os.path.exists('./certify_MCNN/attack'):
        os.mkdir('./certify_MCNN/attack')

    if not os.path.exists('./certify_MCNN/adv_img'):
        os.mkdir('./certify_MCNN/adv_img')

    for blob in data_loader:
        im_data = blob['data']
        gt_data = blob['gt_density']
        full_imgname = blob['fname']

        # 做PGD attack 一定要图片归一化！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # im_data = normalization(im_data)
        im_data /= 255
        # print("img scale 0-1: ", np.sum(im_data > 1))

        img_var = np_to_variable(im_data, is_cuda=True, is_training=False)
        target_var = np_to_variable(gt_data, is_cuda=True, is_training=False)

        # adv_x = adv_LINF(img_var, target_var, model, criterion,eps,step_size,iterations)

        ################ PGD 的函数里面图片像素必须是0-1 #########################################################
        adv_x = get_adv_examples_LINF(img_var, target_var, model, criterion,eps,step_size,iterations)
                                   # (data, target, model, lossfunc, eps, step_size, iterations)

        adv_x = adv_x.data.detach().cpu().numpy()
        # print(np.max(adv_x - im_data))

        # 送入模型必须得是 0-255之间！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        adv_x *= 255
        adv_x = np_to_variable(adv_x, is_cuda=True, is_training=False)

        # print("adv_x_shape: ", adv_x.shape)
        # adv_x.shape = (1,1,750,1024)

        adv_output_var = model(adv_x, target_var)

        adv_density_map = adv_output_var.data.detach().cpu().numpy()

        plt.imsave('./certify_MCNN/attack/{}'.format(full_imgname), adv_density_map[0][0],
                   format='png', cmap=plt.cm.jet)

        # store adversarial examples

        adv_example = adv_x.data.detach().cpu().numpy()

        adv_img_store = adv_example.squeeze(0).squeeze(0)
        # adv_img_store = normalization(adv_img_store)
        # adv_img_store = array_transpose(adv_img_store)

        plt.imsave('./certify_MCNN/adv_img/{}'.format(full_imgname), adv_img_store, format='png', cmap=plt.cm.jet)

# ******************************************** cal MAE & MSE *************************************************

        adv_count = np.sum(adv_density_map)
        gt_count = np.sum(gt_data)

        adv_mae += abs(gt_count - adv_count)
        adv_mse += ((gt_count - adv_count) * (gt_count - adv_count))

    adv_mae = adv_mae / data_loader.get_num_samples()
    adv_mse = np.sqrt(adv_mse / data_loader.get_num_samples())

    print('\n adv_MAE: %0.2f, adv_MSE: %0.2f' % (adv_mae, adv_mse))

    f = open('Adv_Test.txt', 'w')
    f.write('adv_MAE: %0.2f, adv_MSE: %0.2f' % (adv_mae, adv_mse))
    f.close()


def clean_test(data_loader, model):
    mae = 0.0
    mse = 0.0
    for blob in data_loader:
        im_data = blob['data']
        gt_data = blob['gt_density']

        img_var = np_to_variable(im_data, is_cuda=True, is_training=False)
        target_var = np_to_variable(gt_data, is_cuda=True, is_training=False)

        output_var = model(img_var, target_var)
        density_map = output_var.data.detach().cpu().numpy()

        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) * (gt_count - et_count))

    mae = mae / data_loader.get_num_samples()
    mse = np.sqrt(mse / data_loader.get_num_samples())
    print('\nClean MAE: %0.2f, Clean MSE: %0.2f' % (mae,mse))

    f = open('Clean_Test.txt', 'w')
    f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
    f.close()


if __name__ == '__main__':

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    # data_path = './data/original/shanghaitech/part_B_final/test_data/images/'
    # gt_path = './data/original/shanghaitech/part_B_final/test_data/ground_truth_csv/'

    data_path = './data/original/shanghaitech/part_A_final/test_data/images/'
    gt_path = './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'

    model_path = './Shanghai_A_Retrain_eps_1_5/saved_models//MCNN_Shanghai_A.h5'

    model = CrowdCounter()
    trained_model = os.path.join(model_path)
    network.load_net(trained_model, model)
    model.to(device)
    model.eval()

    data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

    eps_list = [0, 0.25, 0.5, 0.75, 1]

    print(model_path)

    epoch_robust_bound(data_loader, model, device, epsilon_try=0)
