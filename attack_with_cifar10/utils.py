#import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from tqdm import tqdm
import os
from TinyImageNet import TinyImageNet

DATASET = os.getenv('DATASET_NAME')

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

# MNIST meand and std
mnist_mean = 0.1307
mnist_std = 0.3081

# FashionMNIST meand and std
fmnist_mean = 0.2861
fmnist_std = 0.3530

# TinyImagenet
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
data_dir = '../../tiny-imagenet-200/'


if DATASET == 'MNIST' or DATASET == 'mnist':
    dataset_mean = mnist_mean
    dataset_std = mnist_std
    mu = torch.tensor(mnist_mean).view(1,1,1).cuda()
    std = torch.tensor(mnist_std).view(1,1,1).cuda()
elif DATASET.lower() == 'fashionmnist':
    dataset_mean = fmnist_mean
    dataset_std = fmnist_std
    mu = torch.tensor(fmnist_mean).view(1,1,1).cuda()
    std = torch.tensor(fmnist_std).view(1,1,1).cuda()
elif DATASET == 'CIFAR10' or DATASET == 'cifar10':
    dataset_mean = cifar10_mean
    dataset_std = cifar10_std
    mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.tensor(cifar10_std).view(3,1,1).cuda()
elif DATASET.lower() == 'tinyimagenet':
    dataset_mean = imagenet_mean
    dataset_std = imagenet_std
    mu = torch.tensor(imagenet_mean).view(3,1,1).cuda()
    std = torch.tensor(imagenet_std).view(3,1,1).cuda()


upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
    ])
    num_workers = 2
    if DATASET.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)
    elif DATASET.lower() == 'mnist':
        dir_ = dir_.replace('cifar-data','mnist')
        train_dataset = datasets.MNIST(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.MNIST(
            dir_, train=False, transform=test_transform, download=True)
    elif DATASET.lower() == 'fashionmnist':
        dir_ = dir_.replace('cifar-data','fashionmnist')
        train_dataset = datasets.FashionMNIST(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.FashionMNIST(
            dir_, train=False, transform=test_transform, download=True)
    elif DATASET.lower() == 'tinyimagenet':
        dir_ = dir_.replace('cifar-data','tinyimagenet') 
        train_dataset = TinyImageNet(data_dir, train=True,transform=train_transform)
        test_dataset = TinyImageNet(data_dir, train=False, transform=test_transform)  
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                #with amp.scale_loss(loss, opt) as scaled_loss:
                    #scaled_loss.backward()
                loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(tqdm(test_loader)):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n

def evaluate_attack(model,test_loader,attack):
    model.eval()
    correct = 0
    total = 0
    
    for data in test_loader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        adv_images = attack(images, labels)
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(test_loader)):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n
