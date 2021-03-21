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
import torchvision
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--ratio', type=float, default=2)
parser.add_argument('--beta', type=float, default=10000)
parser.add_argument('--warm', type=int, default=5)
parser.add_argument('--up', type=int, default=40)
parser.add_argument('--kappaup', type=int, default=40)
parser.add_argument('--drop1', type=int, default=65)  #epoch to drop lr
parser.add_argument('--drop2', type=int, default=80)  #epoch to drop lr
parser.add_argument('--out', type=str, default="tiny.txt")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using cuda: ", format(device))


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


# flatten 张量扁平化操作(为了输出给FC层而操作)
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


# for the soft output
def cross_entropy(p,q):
    return F.kl_div(F.log_softmax(q), F.softmax(p), reduction='sum')


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


# epoch 代表正常训练的情况下模型测试的error
def epoch(loader, model, device, opt=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        # total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset)


def generate_kappa_schedule_MNIST():
    kappa_schedule = []
    for i in range(300):
        if i<args.warm:
            kappa_schedule.append(1)
        else:
            kappa_schedule.append(max(0, 1-(i-args.warm+1)/args.kappaup))

    return kappa_schedule


def generate_epsilon_schedule_MNIST(epsilon_train):
    epsilon_schedule = []
    step = epsilon_train / args.up

    for i in range(args.warm):
        epsilon_schedule.append(0)
    for i in range(args.up):
        epsilon_schedule.append((i+1) * step)

    for j in range(200):
        epsilon_schedule.append(epsilon_train)

    return epsilon_schedule


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
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t()
                  + layer.bias[:, None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t()
                  + layer.bias[:, None]).t()

        # @ 是tensor之间矩阵相乘，* 是矩阵之间元素点乘, .t()  矩阵转置
        # 如果是卷积层
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


def interval_based_bound(model, c, bounds, idx):
    # requires last layer to be linear
    cW = c.t() @ model[-1].weight
    '''
    print("c.t()", c.t().shape)
    print("model[-1].weight", model[-1].weight.shape)
    '''
    cb = c.t() @ model[-1].bias
    # print("cw", cW.shape)
    l, u = bounds[-2]
    return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t() + cb[:, None]).t()


def epoch_robust_bound(loader, model, epsilon_schedule, device, kappa_schedule, batch_counter, opt):
    robust_err = 0

    total_combined_loss = 0

    normal_loss = 0
    certify_loss = 0
    reg_loss = 0

    # torch.eye(n,m) 生成n行m列的矩阵，其中，对角线元素全为1，其余全为0
    C = [-torch.eye(10).to(device) for _ in range(10)]  # C是一个列表，里面10个元素，每个元素是torch.eye(10)

    for y0 in range(10):
        C[y0][y0, :] += 1  # 例如，y0 == 0, 则C的第一个元素(矩阵)中，第一行的元素全为1

    for X, y in loader:  # 一个是img，一个是target
        X = X.to(device)
        y = y.to(device)
        # normal prediction
        yp = model(X)
        fit_loss = nn.CrossEntropyLoss()(yp, y)  # normal loss

        initial_bound = (X - epsilon_schedule[batch_counter], X + epsilon_schedule[batch_counter])

        bounds = bound_propagation(model, initial_bound)

        # l, u = bounds[-1]

        robust_loss = 0

        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(abs(param))



        for y0 in range(10):
            if sum(y == y0) > 0:
                lower_bound = interval_based_bound(model, C[y0], bounds, y == y0)
                robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound, y[y==y0])
                robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item() # increment when true label is not winning



        combined_loss = kappa_schedule[batch_counter]*fit_loss + (1-kappa_schedule[batch_counter])*args.alpha*robust_loss + (regularization_loss/args.beta)

        # for training
        total_combined_loss += combined_loss.item()

        # for observation
        normal_loss += fit_loss.item()
        certify_loss += robust_loss.item()
        reg_loss += regularization_loss.item()

        if opt:
            opt.zero_grad()
            combined_loss.backward()
            opt.step()

    print("normal loss: ", normal_loss/len(loader.dataset))
    print("certify_loss: ", certify_loss/len(loader.dataset))
    print("reg loss: ", reg_loss/len(loader.dataset))

    return model, normal_loss/len(loader.dataset),certify_loss/len(loader.dataset),reg_loss/len(loader.dataset)


def epoch_calculate_robust_err(loader, model, epsilon, device):
    robust_err = 0.0

    C = [-torch.eye(10).to(device) for _ in range(10)]
    for y0 in range(10):
        C[y0][y0, :] += 1
    total_err=0
    robust_loss=0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp=model(X) 
            initial_bound = (X - epsilon, X + epsilon)
            bounds = bound_propagation(model, initial_bound)
            total_err += (yp.max(dim=1)[1] != y).sum().item()
            for y0 in range(10):
                if sum(y == y0) > 0:
                    lower_bound = interval_based_bound(model, C[y0], bounds, y == y0)
                    robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound, y[y==y0])
                    robust_err += (lower_bound.min(dim=1)[
                                    0] < 0).sum().item()  # increment when true label is not winning

    return robust_err / len(loader.dataset),total_err/len(loader.dataset),robust_loss.item()/len(loader.dataset)

def IBP_large(in_ch, in_dim, linear_size=512): 
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
        nn.Linear((in_dim//2) * (in_dim//2) * 128, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model

def main():
    torch.manual_seed(0)

    mnist_train = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(),
                                             download=True)

    mnist_test = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor(),
                                            download=True)

    train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

    model_cnn_medium = nn.Sequential(nn.Conv2d(1, 32, 3, padding=0, stride=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 4, padding=0, stride=2), nn.ReLU(),
                                     nn.Conv2d(32, 64, 3, padding=0, stride=1), nn.ReLU(),
                                     nn.Conv2d(64, 64, 4, padding=0, stride=2), nn.ReLU(),
                                     Flatten(),
                                     nn.Linear(64 * 4 * 4, 512), nn.ReLU(),
                                     nn.Linear(512, 512), nn.ReLU(),
                                     nn.Linear(512, 10)).to(device)
    model_cnn_medium=IBP_large(1,28,512).to(device)
    opt = optim.Adam(model_cnn_medium.parameters(), lr=args.lr)

    EPSILON = args.eps
    EPS_TRAIN=EPSILON*args.ratio

    epsilon_schedule = generate_epsilon_schedule_MNIST(EPS_TRAIN)
    kappa_schedule = generate_kappa_schedule_MNIST()
    batch_counter = 0

    epochs = 500

    # print("Epoch   ", "Combined Loss", "Test Err", "Test Robust Err", sep="\t")

    if not os.path.exists('./MNIST_IBP_Training_Models_eps_0_1'):
        os.makedirs('./MNIST_IBP_Training_Models_eps_0_1')

    output_dir = './MNIST_IBP_Training_Models_eps_0_1/'
    method = 'MNIST_CNN_Medium'

    print("Epoch   ", "Normal Loss", "Cert Loss","Reg loss", "Test Robust Err","Test clean Err", sep="\t")
    a_txt=open(args.out, 'w')
    print(args,file=a_txt)
         
    a_txt.close()
    for t in range(epochs):
        model, loss_normal,loss_cert,loss_reg = epoch_robust_bound(train_loader, model_cnn_medium, epsilon_schedule, device,
                                                  kappa_schedule,
                                                  batch_counter, opt)
        batch_counter += 1

        robust_err,clean_err,robust_loss = epoch_calculate_robust_err(test_loader, model_cnn_medium, EPSILON, device)
        a_txt=open(args.out, 'a')
        print(*("{:.4f}".format(i) for i in (t, loss_normal,loss_cert,loss_reg,  robust_err,clean_err,robust_loss)), sep="\t",file=a_txt)
        a_txt.close()

        # save model
        save_name = os.path.join(output_dir, '{}_epoch_{}_{}_robust_{}.h5'.format(method,EPSILON, t, robust_err))
        save_net(save_name, model)

        if epoch == args.drop1:  # decrease learning rate after 200 epochs
            for param_group in opt.param_groups:
                param_group["lr"] = args.lr*0.1

        if epoch == args.drop2:  # decrease learning rate after 200 epochs
            for param_group in opt.param_groups:
                param_group["lr"] = args.lr*0.01


if __name__ == '__main__':
    # cal time cost
    begin_time = time.perf_counter()

    main()

    end_time = time.perf_counter()
    run_time = end_time - begin_time
    f = open('./MNIST_IBP_Training_Models_eps_0_1/collect_time.txt', 'w')
    f.write('time: %s ' % str(run_time))
    f.close()
