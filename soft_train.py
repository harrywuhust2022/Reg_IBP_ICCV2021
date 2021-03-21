from __future__ import division
import os
import torch
import numpy as np
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
import time

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("using cuda: ", format(device))


def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_cuda:
        v = (torch.from_numpy(x).type(dtype)).to(device)
    if is_training:
        v = Variable(v, requires_grad=True, volatile=False)
    else:
        v = Variable(v, requires_grad=False, volatile=True)
    return v


# collect the max distance of each pixel of 2 matrix
def cal_distance(array_a, array_b):
    return np.where(array_a > array_b, array_a, array_b)


def main():
    lr = 0.00001
    start_epoch = 0
    end_epoch = 1  # for training epochs

    # save trained model

    if not os.path.exists('./Shanghai_B_Retrain'):
        os.makedirs('./Shanghai_B_Retrain')

    if not os.path.exists('./Shanghai_B_Retrain/saved_models_all_have'):
        os.makedirs('./Shanghai_B_Retrain/saved_models_all_have')
    output_dir = './Shanghai_B_Retrain/saved_models_all_have/'

    if not os.path.exists('./Shanghai_B_Retrain/loss_all_have'):
        os.mkdir('./Shanghai_B_Retrain/loss_all_have')

    method = 'MCNN'
    dataset_name = 'Shanghai_B'

    # test_data_path = './data/original/shanghaitech/part_A_final/test_data/images/'
    # test_gt_path = './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'

    train_path = './data/formatted_trainval/shanghaitech_part_B_patches_9/train'
    train_gt_path = './data/formatted_trainval/shanghaitech_part_B_patches_9/train_den'
    val_path = './data/formatted_trainval/shanghaitech_part_B_patches_9/val'
    val_gt_path = './data/formatted_trainval/shanghaitech_part_B_patches_9/val_den'

    train_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
    val_loader = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)

    # load normal models for training
    model = CrowdCounter()
    weights_normal_init(model, dev=0.01)
    model.to(device)

    params = list(model.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    epsilon_initial = 1/255
    
    epsilon_schedule = generate_epsilon_schedule(epsilon_initial)
    kappa = generate_kappa_schedule()

    criterion = nn.MSELoss(size_average=False).to(device)

    mae = 0.0
    mse = 0.0
    certify_mae = 0.0
    certify_mse = 0.0

# ******************************************* Training *********************************************************

    Loss_list = []
    Loss_certify_list = []
    Loss_L1_reg_list = []
    Loss_normal_list = []

    dtype = torch.FloatTensor

    for epoch in range(start_epoch, end_epoch):

        model.train()

        # for training
        epoch_loss = 0.0  # 记录loss

        # for observation
        certify_loss_epoch = 0.0
        reg_loss_epoch = 0.0
        normal_loss_store_epoch = 0.0
        
        for blob in train_loader:

            im_data = blob['data']  # (1,1,704,1024)
            gt_data = blob['gt_density']

            X = np_to_variable(im_data, is_cuda=True, is_training=True)
            initial_bound = (X - epsilon_schedule[epoch], X + epsilon_schedule[epoch])

            img_var = np_to_variable(im_data, is_cuda=True, is_training=True)
            target_var = np_to_variable(gt_data, is_cuda=True, is_training=True)

# ************************certify bound calculation**************************************************************

            lower_bound, upper_bound = bound_propagation(model, initial_bound, epsilon_schedule[epoch])

            lower_train_var = Variable(lower_bound.to(device))
            upper_train_var = Variable(upper_bound.to(device))

# *****************************normal output**************************************************************************

            output_train_var = model(img_var, target_var)

            loss_normal = criterion(output_train_var, target_var)

# ****************************** certify loss *******************************************************

            low_np = lower_train_var.data.detach().cpu().numpy()
            upp_np = upper_train_var.data.detach().cpu().numpy()

            low_GT = abs(low_np - gt_data)
            upp_GT = abs(upp_np - gt_data)

            max_distance = cal_distance(low_GT, upp_GT)

            dis = (torch.from_numpy(max_distance).type(dtype)).to(device)

            # soft learning
            loss_certify = torch.sum(dis * dis)

# ****************** L1 regularization***********************************************************
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))

            # total loss
            loss = kappa[epoch]*loss_normal + (1-kappa[epoch])*loss_certify + (regularization_loss/1000)

            # for observation
            certify_loss_epoch += loss_certify.item()
            reg_loss_epoch += regularization_loss.item()
            normal_loss_store_epoch += loss_normal.item()

            # for training
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # *****************************for training****************************************************
        Loss_list.append(epoch_loss / train_loader.get_num_samples())

        # *****************************for observation*************************************************
        Loss_certify_list.append(certify_loss_epoch/train_loader.get_num_samples())

        Loss_L1_reg_list.append(reg_loss_epoch / train_loader.get_num_samples())

        Loss_normal_list.append(normal_loss_store_epoch / train_loader.get_num_samples())

        # save model parameter
        if epoch % 10 == 9:
            save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method, dataset_name, epoch))
            save_net(save_name, model)

# **************************** validate time ******************************************************

        if epoch == end_epoch - 1:
            # test time
            model.eval()
            with torch.no_grad():
                for blob in val_loader:
                    im_data = blob['data']  # (1,1,704,1024)
                    gt_data = blob['gt_density']

                    img_var = np_to_variable(im_data, is_cuda=True, is_training=False)
                    target_var = np_to_variable(gt_data, is_cuda=True, is_training=False)

                    # **************certify bound calculation******************************************************************

                    X = (torch.from_numpy(im_data).type(dtype)).to(device)
                    initial_bound = (X - epsilon_initial, X + epsilon_initial)
                    lower_bound, upper_bound = bound_propagation(model, initial_bound, epsilon_initial)

                    # ***************normal output*********************************************************

                    density_map_var = model(img_var, target_var)
                    output = density_map_var.data.detach().cpu().numpy()

                    lower = lower_bound.detach().cpu().numpy()
                    upper = upper_bound.detach().cpu().numpy()

                    et_lower = np.sum(lower)
                    et_upper = np.sum(upper)

                    # 验证lower_bound 和 upper_bound, 输出都是0的话代表是成功的
                    print(np.sum(output < lower))
                    print(np.sum(output > upper))

                    # ***************** MAE MSE Certify / Normal *******************************************

                    gt_count = np.sum(gt_data)
                    et_count = np.sum(output)

                    if abs(et_upper - gt_count) > abs(et_lower - gt_count):
                        et_certify = et_upper
                    else:
                        et_certify = et_lower

                    mae += abs(gt_count - et_count)
                    mse += ((gt_count - et_count) * (gt_count - et_count))

                    certify_mae += abs(et_certify - gt_count)
                    certify_mse += ((et_certify - gt_count) * (et_certify - gt_count))

                certify_mae = certify_mae / val_loader.get_num_samples()
                certify_mse = np.sqrt(certify_mse / val_loader.get_num_samples())

                mae = mae / val_loader.get_num_samples()
                mse = np.sqrt(mse / val_loader.get_num_samples())

        train_loss_txt = open('./Shanghai_B_Retrain/loss_all_have/train_loss.txt', 'w')
        for value in Loss_list:
            train_loss_txt.write(str(value))
            train_loss_txt.write('\n')
        train_loss_txt.close()

        # for observation
        train_loss_txt = open('./Shanghai_B_Retrain/loss_all_have/train_loss_certify.txt', 'w')
        for value in Loss_certify_list:
            train_loss_txt.write(str(value))
            train_loss_txt.write('\n')
        train_loss_txt.close()

        train_loss_txt = open('./Shanghai_B_Retrain/loss_all_have/train_loss_L1_reg.txt', 'w')
        for value in Loss_L1_reg_list:
            train_loss_txt.write(str(value))
            train_loss_txt.write('\n')
        train_loss_txt.close()

        train_loss_txt = open('./Shanghai_B_Retrain/loss_all_have/train_normal_loss.txt', 'w')
        for value in Loss_normal_list:
            train_loss_txt.write(str(value))
            train_loss_txt.write('\n')
        train_loss_txt.close()


if __name__ == '__main__':
    # cal time cost
    begin_time = time.perf_counter()

    main()

    end_time = time.perf_counter()
    run_time = end_time - begin_time
    f = open('./Shanghai_B_Retrain/loss_all_have/collect_time.txt', 'w')
    f.write('time: %s ' % str(run_time))
    f.close()
