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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using cuda: ", format(device))


def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_cuda:
        v = (torch.from_numpy(x).type(dtype)).to(device)
    if is_training:
        v = Variable(v, requires_grad=True, volatile=False)
    else:
        v = Variable(v, requires_grad=False, volatile=True)
    return v


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


# certify functions # *************************************************************************


def generate_kappa_schedule():
    kappa_schedule = []

    for j in range(50):
        kappa_schedule.append(1)

    kappa_value = 1.0
    step = 0.5 / 150

    for i in range(150):
        kappa_value -= step
        kappa_schedule.append(kappa_value)

    for k in range(200):
        kappa_schedule.append(0.5)

    return kappa_schedule


def generate_epsilon_schedule(epsilon_train):
    epsilon_schedule = []
    step = epsilon_train / 100

    for i in range(100):
        epsilon_schedule.append(i * step)  # warm-up phase

    for i in range(300):
        epsilon_schedule.append(epsilon_train)

    return epsilon_schedule


def interval_based_bound(model, c, bounds, idx):
    # requires last layer to be linear
    # .t()代表矩阵转置！！！
    cW = c.t() @ model.last_linear.weight
    cb = c.t() @ model.last_linear.bias

    l, u = bounds[-2]
    return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t() + cb[:, None]).t()


def colorize_mask(mask):
    if mask.shape[0] == 3:
        mask = (np.transpose(mask, (1, 2, 0)) + 1) / 2.0 * 255.0
    elif mask.shape[0] == 1:
        mask = (mask[0] + 1) / 2.0 * 255.0
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    return new_mask


def array_transpose(array):
    temp = np.transpose(array, (1, 2, 0))
    return temp


def normalization(data):
    max_val = []
    min_val = []
    # 因为MCNN的图是(1,1,750,1024)这种级别
    for i in range(1):
        max_val.append(np.max(data[i]))
        min_val.append(np.min(data[i]))
        data[i] = (data[i] - min_val[i]) / (max_val[i] - min_val[i])

    return data


def bound_propagation(model, initial_bound, epsilon_try):

    low1, upp1 = initial_bound
    low2, upp2 = initial_bound
    low3, upp3 = initial_bound

    # for layer, module in model.named_modules():

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

# ********************************************* attack function ************************************


def getw(output):
    out = F.softmax(output, dim=1).permute(0, 2, 3, 1)
    values, indices = torch.topk(out, 2)
    vl = values.permute(3, 0, 1, 2)
    vl = torch.clamp(vl, 1e-8, 0.999998)

    from scipy.stats import norm
    surrogate = vl[0] - vl[1]
    vl = norm.ppf(vl[0].detach().cpu()) - norm.ppf(vl[1].detach().cpu())
    vl = torch.from_numpy(vl).float()

    return torch.cat((vl.unsqueeze(1), vl.unsqueeze(1), vl.unsqueeze(1)), dim=1), vl, torch.cat(
        (surrogate.unsqueeze(1), surrogate.unsqueeze(1), surrogate.unsqueeze(1)), dim=1)


class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True,p=1):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad
        self.p=p

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x


class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=self.p, dim=0, maxnorm=self.eps)
        return torch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1),self.p,dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        l = len(x.shape) - 1
        rp = torch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        return torch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)


def adv_L2(data, target, model, lossfunc, eps, step_size, iterations):
    global cnt
    cnt += 1
    m = 1  # for untargeted 1
    iterator = range(iterations)
    x = data.clone()
    step_count = 0

    output = model(x)['out']

    step = L2Step(x, eps, step_size, True, 2)
    for _ in iterator:
        x = x.clone().detach().requires_grad_(True)
        output = model(x)['out']
        correct = calculate_correct_map(output, target, 21)

        losses = lossfunc(output, target.cuda())

        loss = torch.mean(losses)

        if step.use_grad:
            grad, = torch.autograd.grad(m * loss, [x])
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    return x


def adv_LINF(data, target, model, lossfunc, eps, alpha, iterations):
    m = 1  # for untargeted 1
    iterator = range(iterations)

    # random start  代替 x = data.clone()
    data_replace = data.data.detach().cpu().numpy()
    x = data_replace + np.random.uniform(-eps, eps, data_replace.shape)
    x = np_to_variable(x, is_cuda=True, is_training=False)
    x = torch.clamp(x, min=0, max=1) # ensure valid pixel range

    # x = data.clone()
    ori_img = data.clone()
    '''
    for _ in iterator:
        # x = x.clone().requires_grad_(True)
        x.requires_grad = True
        model.zero_grad()
        output = model(x, target)
        losses = lossfunc(output, target)
        loss = torch.mean(losses)
        loss.backward()
        # grad, = torch.autograd.grad(m * loss, [x])

        with torch.no_grad():
            delta = x - data
            # delta = delta + alpha * torch.sign(grad)
            delta = delta + alpha * x.grad.sign()
            delta = torch.clamp(delta, -eps, eps)
            x = torch.clamp(x + delta, 0, 1)
    '''
    '''
    for _ in iterator:
        x.requires_grad = True
        output = model(x, target)
        model.zero_grad()
        cost = lossfunc(output, target).to(device)
        cost.backward()
        adv_images = x + alpha * x.grad.sign()
        eta = torch.clamp(adv_images - ori_img, min=-eps, max=eps)
        x = torch.clamp(ori_img + eta, min=0, max=1).detach()
    '''

    adv = x.clone().detach().requires_grad_(True).to(device)

    for _ in iterator:
        _adv = adv.clone().detach().requires_grad_(True)
        outputs = model(_adv, target)

        model.zero_grad()
        cost = lossfunc(outputs, target)
        cost.backward()

        grad = _adv.grad
        grad = grad.sign()

        assert (data.shape == grad.shape)

        adv = adv + grad * alpha

        # project back onto Lp ball
        adv = torch.max(torch.min(adv, data + eps), data - eps)

        adv = adv.clamp(0.0, 1.0)

    return adv.detach()


def get_adv_examples_LINF(data, target, model, lossfunc, eps, step_size, iterations):
    m = 1  # for untargeted 1
    iterator = range(iterations)

    # random start
    data_replace = data.data.detach().cpu().numpy()
    x = data_replace + np.random.uniform(-eps, eps, data_replace.shape)
    x = np_to_variable(x, is_cuda=True, is_training=False)
    x = torch.clamp(x, min=0, max=1) # ensure valid pixel range

    # x = data.clone()

    for _ in iterator:
        #x = x.clone().detach().requires_grad_(True)
        x.requires_grad_(True)
        output = model(x, target)
        model.zero_grad()
        losses = lossfunc(output, target)

        #print("attack_loss: ", losses)

        #loss = torch.mean(losses)
        losses.backward()
        #grad, = torch.autograd.grad(m * losses, [x])
        grad = x.grad
        with torch.no_grad():
            delta = x - data
            delta = delta + step_size * torch.sign(grad)
            delta = torch.clamp(delta, -eps, eps)
            x = torch.clamp(data + delta, 0, 1)

    return x.detach()


# collect the max distance of each pixel of 2 matrix
def cal_distance(array_a, array_b):
    return np.where(array_a > array_b, array_a, array_b)
