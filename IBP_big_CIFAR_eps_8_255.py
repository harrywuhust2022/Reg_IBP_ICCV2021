# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from utils import epoch, epoch_robust_bound, epoch_calculate_robust_err, Flatten, generate_kappa_schedule_CIFAR, generate_epsilon_schedule_CIFAR
# from ResNet110 import *
import os
import argparse

'''
parser = argparse.ArgumentParser()
parser.add_argument("epsilon_test", type=float, help="evaluate the model")
parser.add_argument("epsilon_train", type=float, help="train the model")
parser.add_argument("epsilon_test_1", type=float, help="evaluate the model with another epsilon")
parser.add_argument("epsilon_test_2", type=float, help="evaluate the model with another epsilon")

args = parser.parse_args()
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("use-cuda: ", format(device))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def clean_test(loader, model, device):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)

        total_err += (yp.max(dim=1)[1] != y).sum().item()

    return total_err / len(loader.dataset)


# IBP 核心代码
def bound_propagation(model, initial_bound):
    # lower = x-ep, upper = x+ep
    l, u = initial_bound

    bounds = []
    # list_of_layers = list(model.children())
    # layer_1 = list_of_layers[0]

    for layer in model:

        # isinstance(a,int) 若a是int型, 则返回true

        if isinstance(layer, Flatten):
            l_ = Flatten()(l)
            u_ = Flatten()(u)

        # 如果是线性层
        # torch.clamp(input, min, max, out=None) → Tensor
        # 这就是文章中的公式，下界[0,...]
        elif isinstance(layer, nn.Linear):
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t() + layer.bias[:, None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t() + layer.bias[:, None]).t()

        elif isinstance(layer, nn.Conv2d):
            l_ = (nn.functional.conv2d(l, layer.weight.clamp(min=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(u, layer.weight.clamp(max=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None, :, None, None])

            u_ = (nn.functional.conv2d(u, layer.weight.clamp(min=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(l, layer.weight.clamp(max=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None, :, None, None])

        # 如果是relu层
        elif isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)

        bounds.append((l_, u_))
        l, u = l_, u_
    return bounds


def translation(mu, translation, y):
    onehot_y = torch.FloatTensor(y.shape[0], 200).cuda()
    onehot_y.zero_()
    onehot_y.scatter_(1, y.view(-1, 1), 1)

    b, c = onehot_y.size()

    # mu : b, 10

    # translation : b, 10, 10

    # onehot_y: b, 10

    delta = translation.bmm(onehot_y.unsqueeze(2)).view(b, c)

    # b,10,10 x b,10,1 -> b,10,1 -> b,10

    # delta: b,10

    translated_logit = mu + ((delta) * (1 - onehot_y))

    return translated_logit


def ibp_translation(ibp_mu_prev, ibp_r_prev, W):  ################# bottle neck
    EPS = 1e-24
    c, h = W.size()
    b = ibp_mu_prev.size()[0]
    WW = W.repeat(c, 1, 1)
    WWW = W.view(c, 1, -1).repeat(1, c, 1)

    wi_wj = (WWW - WW).unsqueeze(0)  # 1,10,10,h
    wi_wj_rep = wi_wj.repeat(b, 1, 1, 1)

    u_rep = ibp_r_prev.view(b, 1, 1, -1).repeat(1, c, c, 1)  # b,10,10,1
    l_rep = -ibp_r_prev.view(b, 1, 1, -1).repeat(1, c, c, 1)  # b,10,10,1

    inner_prod = ((wi_wj_rep >= 0).type(torch.float) * wi_wj_rep * u_rep).sum(-1) + (
            (wi_wj_rep < 0).type(torch.float) * wi_wj_rep * l_rep).sum(-1)

    return inner_prod


def IBP_last(model, bounds, onehot_y):
    l, u = bounds[-2]
    ibp_mu_prev = (l + u) / 2
    ibp_r_prev = (u - l) / 2
    L, U = bounds[-1]
    ibp_mu = (L + U) / 2
    another_t = ibp_translation(ibp_mu_prev, ibp_r_prev, model[-1].weight)  #####
    another_logit = translation(ibp_mu, another_t, onehot_y)
    return another_logit


def interval_based_bound(model, c, bounds, idx):
    # requires last layer to be linear
    cW = c.t() @ model[-1].weight
    cb = c.t() @ model[-1].bias

    l, u = bounds[-2]
    return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t() + cb[:, None]).t()


def robust_train(loader, model, epsilon_schedule, device, kappa_schedule, batch_counter, alpha, opt=None):
    robust_err = 0
    total_robust_loss = 0
    total_combined_loss = 0

    C = [-torch.eye(10).to(device) for _ in range(10)]  # C是一个列表，里面10个元素，每个元素是torch.eye(10)
    for y0 in range(10):
        C[y0][y0, :] += 1  # 例如，y0 == 0, 则C的第一个元素(矩阵)中，第一行的元素全为1

    normal_loss = 0
    certify_loss = 0
    reg_loss = 0

    for X, y in loader:  # 一个是img，一个是target
        X, y = X.to(device), y.to(device)

        # normal prediction
        yp = model(X)
        fit_loss = nn.CrossEntropyLoss()(yp, y)  # normal loss

        initial_bound = (X - epsilon_schedule[batch_counter], X + epsilon_schedule[batch_counter])

        bounds = bound_propagation(model, initial_bound)

        robust_loss = 0

        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(abs(param))

        for y0 in range(10):
            if sum(y == y0) > 0:
                lower_bound = interval_based_bound(model, C[y0], bounds, y == y0)
                robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound, y[y == y0])
                robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item()

        combined_loss = kappa_schedule[batch_counter] * fit_loss + (1 - kappa_schedule[batch_counter]) * (
                robust_loss / alpha) + (
                                regularization_loss / 10000)

        total_combined_loss += combined_loss.item()

        # for observation
        normal_loss += fit_loss.item()
        certify_loss += robust_loss.item()
        reg_loss += regularization_loss.item()

        if opt:
            opt.zero_grad()
            combined_loss.backward()
            opt.step()

    print("normal loss: ", normal_loss / len(loader.dataset))
    print("certify_loss: ", certify_loss / len(loader.dataset))
    print("reg loss: ", reg_loss / len(loader.dataset))

    return model, total_combined_loss / len(loader.dataset)


def robust_test(loader, model, epsilon, device):
    robust_err = 0.0

    C = [-torch.eye(10).to(device) for _ in range(10)]
    for y0 in range(10):
        C[y0][y0, :] += 1

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        initial_bound = (X - epsilon, X + epsilon)
        bounds = bound_propagation(model, initial_bound)

        for y0 in range(10):
            if sum(y == y0) > 0:
                lower_bound = interval_based_bound(model, C[y0], bounds, y == y0)
                robust_err += (lower_bound.min(dim=1)[
                                   0] < 0).sum().item()  # increment when true label is not winning

    return robust_err / len(loader.dataset)


def generate_kappa_schedule_CIFAR(epochs):
    # kappa_schedule = 10000 * [1]  # warm-up phase
    kappa_schedule = []
    # kappa_value = 1.0
    # step = 0.5 / 340000

    for i in range(epochs):
        # kappa_value = kappa_value - step
        if i < 5:
            kappa_schedule.append(1)
        else:
            kappa_schedule.append(max(0, 1 - (i - 5) / 100))

    return kappa_schedule


def generate_epsilon_schedule_CIFAR(epsilon_train,epochs):
    epsilon_schedule = []
    step = epsilon_train / 20

    for i in range(5):
        epsilon_schedule.append(0)
    for i in range(20):
        epsilon_schedule.append((i + 1) * step)

    for j in range(epochs-25):
        epsilon_schedule.append(epsilon_train)

    return epsilon_schedule


def CNN_large(in_ch, in_dim, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim // 2) * (in_dim // 2) * 128, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


def main():
    torch.manual_seed(0)

    BATCH_SIZE = 64
    dataset_path = './cifar10'

    trainset = datasets.CIFAR10(root=dataset_path, train=True, download=True)

    train_mean = trainset.data.mean(axis=(0, 1, 2)) / 255  # [0.49139968  0.48215841  0.44653091]
    train_std = trainset.data.std(axis=(0, 1, 2)) / 255  # [0.24703223  0.24348513  0.26158784]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])
    kwargs = {'num_workers': 2, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root=dataset_path, train=True, download=True,
        transform=transform_train),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=dataset_path, train=False, download=True,
                         transform=transform_test),
        batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    model_cnn_medium = nn.Sequential(nn.Conv2d(3, 32, 3, padding=0, stride=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 4, padding=0, stride=2), nn.ReLU(),
                                     nn.Conv2d(32, 64, 3, padding=0, stride=1), nn.ReLU(),
                                     nn.Conv2d(64, 64, 4, padding=0, stride=2), nn.ReLU(),
                                     Flatten(),
                                     nn.Linear(64 * 5 * 5, 512), nn.ReLU(),
                                     nn.Linear(512, 512), nn.ReLU(),
                                     nn.Linear(512, 10)).to(device)

    model_cnn_medium = CNN_large(3,32,512).to(device)
    opt = optim.Adam(model_cnn_medium.parameters(), lr=1e-3)

    # sota 2/255
    epochs = 1000
    EPSILON = 8 / 255
    EPSILON_TRAIN = 8.8 / 255
    epsilon_schedule = generate_epsilon_schedule_CIFAR(EPSILON_TRAIN,epochs)
    kappa_schedule = generate_kappa_schedule_CIFAR(epochs)
    batch_counter = 0

    alpha = 100  # robust loss / 100

    # eps_1 = args.epsilon_test_1
    # eps_3 = args.epsilon_test_2

    method = 'CIFAR_Large'
    if not os.path.exists('./CIFAR_IBP_Training_Models_eps_8_255_large'):
        os.makedirs('./CIFAR_IBP_Training_Models_eps_8_255_large')

    output_dir = './CIFAR_IBP_Training_Models_eps_8_255_large/'

    print("Epoch   ", "Combined Loss", "Test Err", "Test Robust Err", sep="\t")

    for epoch in range(epochs):  # 350
        model, combined_loss = robust_train(train_loader, model_cnn_medium, epsilon_schedule, device,
                                            kappa_schedule,
                                            batch_counter, alpha, opt)
        batch_counter += 1
        test_err = clean_test(test_loader, model_cnn_medium, device)
        robust_err = robust_test(test_loader, model_cnn_medium, EPSILON, device)
        print(*("{:.6f}".format(i) for i in (epoch, combined_loss, test_err, robust_err)), sep="\t")

        save_name = os.path.join(output_dir, '{}_{}_clean_{}_robust_{}.h5'.format(method, epoch, test_err, robust_err))
        save_net(save_name, model)

        if epoch == 29:  # decrease learning rate after 200 epochs
            for param_group in opt.param_groups:
                param_group["lr"] = 1e-4

        if epoch == 99:  # decrease learning rate after 200 epochs
            for param_group in opt.param_groups:
                param_group["lr"] = 1e-5

        if epoch == 199:  # decrease learning rate after 200 epochs
            for param_group in opt.param_groups:
                param_group["lr"] = 1e-6

        if epoch == 499:  # decrease learning rate after 200 epochs
            for param_group in opt.param_groups:
                param_group["lr"] = 1e-7


if __name__ == '__main__':
    main()
