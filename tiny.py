import os
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torch
import torch
import numpy
import os
import pickle
import numpy as np
from torch.utils.data import Dataset,DataLoader
import argparse
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
# from Img_classifier_models import *
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import shutil
import torch
import torchvision
import math
from PIL import Image

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta', type=float, default=100000)
parser.add_argument('--warm', type=int, default=15)
parser.add_argument('--up', type=int, default=80)
parser.add_argument('--kappaup', type=int, default=100)
parser.add_argument('--drop', type=int, default=160)  #epoch to drop lr
parser.add_argument('--out', type=str, default="tiny.txt")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("using cuda: ", format(device))


labels_t = []
image_names = []
with open('./ImageNet_tiny/tiny-imagenet-200/wnids.txt') as wnid:
    for line in wnid:
        labels_t.append(line.strip('\n'))
for label in labels_t:
    txt_path = './ImageNet_tiny/tiny-imagenet-200/train/'+label+'/'+label+'_boxes.txt'
    image_name = []
    with open(txt_path) as txt:
        for line in txt:
            image_name.append(line.strip('\n').split('\t')[0])
    image_names.append(image_name)
labels = np.arange(200)


val_labels_t = []
val_labels = []
val_names = []
with open('./ImageNet_tiny/tiny-imagenet-200/val/val_annotations.txt') as txt:
    for line in txt:
        val_names.append(line.strip('\n').split('\t')[0])
        val_labels_t.append(line.strip('\n').split('\t')[1])
for i in range(len(val_labels_t)):
    for i_t in range(len(labels_t)):
        if val_labels_t[i] == labels_t[i_t]:
            val_labels.append(i_t)
val_labels = np.array(val_labels)

# IBP 核心代码
def bound_propagation(model, initial_bound):
    l, u = initial_bound
    bounds = []

    for module, layer in model.features.named_modules():

        if isinstance(layer, nn.Linear):
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t()
                  + layer.bias[:, None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t()
                  + layer.bias[:, None]).t()

            l = l_
            u = u_
            #print("linear")

        elif isinstance(layer, nn.MaxPool2d):
            l_ = nn.functional.max_pool2d(l, kernel_size=layer.kernel_size, stride=layer.stride,
                        padding=layer.padding, dilation=layer.dilation, ceil_mode=layer.ceil_mode, return_indices=layer.return_indices)
            u_ = nn.functional.max_pool2d(u, kernel_size=layer.kernel_size, stride=layer.stride,
                        padding=layer.padding, dilation=layer.dilation, ceil_mode=layer.ceil_mode, return_indices=layer.return_indices)

            l, u = l_, u_
            #print("max_pooling")

        # @ 是tensor之间矩阵相乘，* 是矩阵之间元素点乘, .t()  矩阵转置
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

            l = l_
            u = u_
            #print("Conv2d")
        elif isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)
            l = l_
            u = u_
            #print("relu")

        bounds.append((l, u))

    # x = x.view(x.size(0), -1)
    l = l.view(l.size(0), -1)
    u = u.view(u.size(0), -1)

    for module, layer in model.classifier_1.named_modules():

        if isinstance(layer, nn.Linear):

            #print("before lower_bound shape: ", l.shape)
            #print("layer.weight.shape: ", layer.weight.shape)

            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t()
                  + layer.bias[:, None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t()
                  + layer.bias[:, None]).t()

            l = l_
            u = u_
            #print("linear")

        elif isinstance(layer, nn.MaxPool2d):
            l_ = nn.functional.max_pool2d(l, kernel_size=layer.kernel_size, stride=layer.stride,
                        padding=layer.padding, dilation=layer.dilation, ceil_mode=layer.ceil_mode, return_indices=layer.return_indices)
            u_ = nn.functional.max_pool2d(u, kernel_size=layer.kernel_size, stride=layer.stride,
                        padding=layer.padding, dilation=layer.dilation, ceil_mode=layer.ceil_mode, return_indices=layer.return_indices)

            l, u = l_, u_


        # @ 是tensor之间矩阵相乘，* 是矩阵之间元素点乘, .t()  矩阵转置
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

            l = l_
            u = u_
            #print("Conv2d_2")
        elif isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)
            l = l_
            u = u_
            #print("relu_2")
        bounds.append((l, u))
    return bounds
'''
@article{lee2020lipschitz,
  title={Lipschitz-Certifiable Training with a Tight Outer Bound},
  author={Lee, Sungyoon and Lee, Jaewook and Park, Saerom},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
'''
def translation(mu, translation, y):
    onehot_y = torch.FloatTensor(y.shape[0], 200).cuda()
    onehot_y.zero_()



    onehot_y.scatter_(1, y.view(-1,1), 1)

    b, c = onehot_y.size()

    # mu : b, 10

    # translation : b, 10, 10

    # onehot_y: b, 10



    delta = translation.bmm(onehot_y.unsqueeze(2)).view(b,c)



    # b,10,10 x b,10,1 -> b,10,1 -> b,10

    # delta: b,10

    translated_logit = mu+((delta)*(1-onehot_y))

    return translated_logit

def ibp_translation(ibp_mu_prev, ibp_r_prev, W): ################# bottle neck
    EPS = 1e-24
    c, h = W.size()
    b = ibp_mu_prev.size()[0]
    WW = W.repeat(c,1,1)
    WWW = W.view(c,1,-1).repeat(1,c,1)
    
    wi_wj = (WWW-WW).unsqueeze(0) # 1,10,10,h
    wi_wj_rep = wi_wj.repeat(b,1,1,1)

    u_rep = ibp_r_prev.view(b,1,1,-1).repeat(1,c,c,1) # b,10,10,1
    l_rep = -ibp_r_prev.view(b,1,1,-1).repeat(1,c,c,1) # b,10,10,1
    
   
    inner_prod  = ((wi_wj_rep>=0).type(torch.float)*wi_wj_rep*u_rep).sum(-1)+((wi_wj_rep<0).type(torch.float)*wi_wj_rep*l_rep).sum(-1)
    
    return inner_prod

def IBP_last(model,bounds,onehot_y):
    l,u=bounds[-2]
    ibp_mu_prev=(l+u)/2
    ibp_r_prev=(u-l)/2
    L,U=bounds[-1]
    ibp_mu=(L+U)/2
    another_t = ibp_translation(ibp_mu_prev, ibp_r_prev, model.classifier_1[-1].weight) #####
    another_logit = translation(ibp_mu, another_t, onehot_y)
    return another_logit
    
def interval_based_bound(model, c, bounds, idx):
    # requires last layer to be linear
    cW = c.t() @ model.classifier_1[-1].weight
    cb = c.t() @ model.classifier_1[-1].bias

    l, u = bounds[-2]
    return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t() + cb[:, None]).t()


class data(Dataset):
    def __init__(self, type, transform):
        self.type = type
        self.train_cnt=0
        self.val_cnt=0
        if type == 'train':
            i = 0
            self.images = []
            self.train_images=[]
            for label in labels_t:
                image = []
                for image_name in image_names[i]:
                    image_path = os.path.join('./ImageNet_tiny/tiny-imagenet-200/train/', label, 'images/', image_name)
                    img = Image.open(image_path).convert('RGB')
                    image.append(img)
                    self.train_images.append(img)
                    self.train_cnt+=1
                self.images.append(image)
                i += 1

        elif type == 'val':
            self.val_images = []
            for val_image in val_names:
                val_image_path = os.path.join('./ImageNet_tiny/tiny-imagenet-200/val/images/', val_image)
                self.val_images.append(Image.open(val_image_path).convert('RGB'))
                self.val_cnt+=1

        self.transform = transform
    

    def __getitem__(self, index):
        label = []
        image = []

        if self.type == 'train':
            label = index // 500
            image = self.train_images[index]
        if self.type == 'val':
            label = val_labels[index]
            image = self.val_images[index]

        return label, self.transform(image)

    def __len__(self):
        len = 0
        if self.type == 'train':
            len =  self.train_cnt
        if self.type == 'val':
            len =  self.val_cnt
        return len

# *************************************model setup***************************************

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)



class Tiny(nn.Module):
    def __init__(self, num_classes=200, init_weights=False):
        super(Tiny, self).__init__()
        self.features = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2),
        nn.ReLU(),
        
        nn.Conv2d(64, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 4, stride=2),
        nn.ReLU(),
        
        nn.Conv2d(128, 256, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, 4, stride=2),
        nn.ReLU())
        self.classifier_1 = nn.Sequential(
            nn.Linear(9216,256),
        nn.ReLU(),
        nn.Linear(256,200)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_1(x)
        return x

def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def clean_test(loader, model, device):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    top1 = AverageMeter()
    top5 = AverageMeter()

    for y,X in loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y).to(device)
        # measure accuracy and record loss
            prec1, prec5 = accuracy(yp.data, y, topk=(1, 5))
            top1.update(prec1[0], X.size(0))
            top5.update(prec5[0], X.size(0))

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]
    # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return total_loss, top1.avg, top5.avg


def clean_train(loader, model, device,  opt):
    normal_loss = 0
    reg_loss = 0
    total_combined_loss = 0
    # for X, y in loader:
    for y,X in loader:

        X, y = X.to(device), y.to(device)
        yp = model(X)

        fit_loss = nn.CrossEntropyLoss()(yp, y).to(device)

        regularization_loss = 0

        for param in model.parameters():
            regularization_loss += torch.sum(abs(param))

        combined_loss =fit_loss + (regularization_loss/args.beta)

        total_combined_loss += combined_loss.item()

        normal_loss += fit_loss.item()
        reg_loss += regularization_loss.item()

        if opt:
            opt.zero_grad()
            combined_loss.backward()
            opt.step()

    return model, normal_loss/len(loader.dataset), reg_loss/len(loader.dataset)  #, total_combined_loss / len(loader.dataset)


def robust_train(loader, model, epsilon_schedule, device, kappa_schedule, batch_counter, opt):
    robust_err = 0
    total_combined_loss = 0

    loss_fit=0
    loss_robust=0
    loss_reg=0

    xxx=0 
    for y,X in loader:
        xxx+=y.shape[0] 
        X, y = X.to(device), y.to(device)
        yp = model(X)

        fit_loss = nn.CrossEntropyLoss()(yp, y).to(device)

        initial_bound = (X - epsilon_schedule[batch_counter], X + epsilon_schedule[batch_counter])

        bounds = bound_propagation(model, initial_bound)

        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(abs(param))
            
        worst=IBP_last(model,bounds,y)
        robust_loss=nn.CrossEntropyLoss(reduction='sum')(worst, y).to(device)
        robust_err += (worst.max(dim=1)[1] != y).sum().item()
        combined_loss =kappa_schedule[batch_counter]*fit_loss+(1-kappa_schedule[batch_counter])*robust_loss+ (regularization_loss/args.beta)

        total_combined_loss += combined_loss.item()
        loss_fit=loss_fit+fit_loss.item()
        loss_robust=loss_robust+robust_loss.item()
        loss_reg=loss_reg+regularization_loss.item()
        if opt:
            opt.zero_grad()
            combined_loss.backward()
            opt.step()
    return model, loss_fit/len(loader.dataset),loss_robust/len(loader.dataset),loss_reg/len(loader.dataset)


def robust_test(loader, model, epsilon, device):
    robust_err = 0.0
    robust_loss=0

    for y,X in loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            initial_bound = (X - epsilon, X + epsilon)
            bounds = bound_propagation(model, initial_bound)
            worst=IBP_last(model,bounds,y)
            robust_loss+=nn.CrossEntropyLoss(reduction='sum')(worst, y).to(device)
            robust_err += (worst.max(dim=1)[1] != y).sum().item()
    
    return robust_err / len(loader.dataset),robust_loss/ len(loader.dataset)


def generate_kappa_schedule_ImageNet():
    kappa_schedule = []
    for i in range(300):
        if i<args.warm:
            kappa_schedule.append(1)
        else:
            kappa_schedule.append(max(0, 1-(i-args.warm+1)/args.kappaup))

    return kappa_schedule


def generate_epsilon_schedule_ImageNet(epsilon_train):
    epsilon_schedule = []
    step = epsilon_train / args.up

    for i in range(args.warm):
        epsilon_schedule.append(0)
    for i in range(args.up):
        epsilon_schedule.append((i+1) * step)

    for j in range(200):
        epsilon_schedule.append(epsilon_train)

    return epsilon_schedule



def main():
    model=Tiny(200)
    model=init_weight(model)
    model.to(device)

    if not os.path.exists('./Tiny_IBP_Trained_Models_lr_0.001_clean_train'):
        os.makedirs('./Tiny_IBP_Trained_Models_lr_0.001_clean_train')
    method = 'BCP_model'
    output_dir = './Tiny_IBP_Trained_Models_lr_0.001_clean_train/'

    start_epoch = 0
    epochs = 500

    batch_counter = 0
    LR=args.lr
    optimizer = optim.Adam(model.parameters(), lr=LR)

    EPS_TRAIN=1/255
    EPS_TEST=1/255
    eps_sch=generate_epsilon_schedule_ImageNet(EPS_TRAIN)
    kappa_sch=generate_kappa_schedule_ImageNet()
# ***********************************dataset setup ***********************************************

    train_dataset = data('train', transform=transforms.Compose([transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),transforms.ToTensor()]))
    val_dataset = data('val', transform=transforms.Compose([transforms.ToTensor()]))

    batch_size = 32

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# ************************************start training***********************************************

    clean_err_list = []
    normal_loss_list = []
    reg_loss_list = []

  
    a_txt=open(args.out, 'w')
    print(args,file=a_txt)
    a_txt.close()
    
    for epoch in range(start_epoch, epochs):
        print("This is  ## Tiny ImageNet ##  epoch: ", epoch)
        robust_loss=0
        robust_err=0
        if epoch>=args.warm:
          model,normal_loss,robust_loss,reg_loss=robust_train(train_loader, model, eps_sch, device, kappa_sch, batch_counter, optimizer)
        else:
          model, normal_loss, reg_loss = clean_train(train_loader, model, device,opt=optimizer)
        batch_counter += 1
        test_loss, prec1, prec5 = clean_test(val_loader, model, device)
        if epoch>=args.warm:
          robust_err,robust_loss=robust_test(val_loader,model,EPS_TEST,device)

        save_name = os.path.join(output_dir, '{}_epoch_{}_robust_{}.h5'.format(method, epoch,1-robust_err))
        save_net(save_name, model)

        clean_err_list.append(test_loss)
        normal_loss_list.append(normal_loss)
        reg_loss_list.append(reg_loss)
        if epoch==args.drop:
            for param_group in optimizer.param_groups:
                param_group["lr"] = LR*0.1

        train_loss_txt = open(args.out, 'a')
        train_loss_txt.write('\nepoch %s ' % str(epoch))
        train_loss_txt.write('\nnormal_loss %s ' % str(normal_loss_list[epoch]))
        train_loss_txt.write('\nreg_loss %s ' % str(reg_loss_list[epoch]))
        train_loss_txt.write('\nprec5 %s ' % str(prec5 ))
        train_loss_txt.write('\nprec1 %s ' % str( prec1))
        train_loss_txt.write('\nrobust prec1 %s ' % str( 1-robust_err))
        train_loss_txt.write('\nrobust loss %s ' % str( robust_loss))
        train_loss_txt.write('\n')
        train_loss_txt.close()
        print('epoch %s ' % str(epoch))
        print('normal_loss %s ' % str(normal_loss_list[epoch]))
        print('reg_loss %s ' % str(reg_loss_list[epoch]))
        print('prec5 %s ' % str(prec5 ))
        print('prec1 %s ' % str( prec1))
        print('robust prec1 %s ' % str( 1-robust_err))
        print('robust loss %s ' % str( robust_loss))


if __name__ == '__main__':
    main()
