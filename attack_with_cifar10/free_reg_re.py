import argparse
import logging
import os
import time

#import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preact_resnet import PreActResNet18
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=30, type=int, help='Total number of epochs will be this argument * number of minibatch replays.')
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.04, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--minibatch-replays', default=8, type=int)
    parser.add_argument('--out-dir', default='train_free_reg_output', type=str, help='Output directory')
    parser.add_argument('--opt', default='Adam', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--opt-level', default='O1', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    '''
    if os.path.exists(logfile):
        os.remove(logfile)
    '''
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        filemode='a',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std

    model = PreActResNet18().cuda()
    model.train()

    if args.opt == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    #model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    delta.requires_grad = True

    lr_steps = args.epochs * len(train_loader) * args.minibatch_replays
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            X.requires_grad_(True)  # 使输入X可计算梯度
            for _ in range(args.minibatch_replays):
                opt.zero_grad()
                delta.grad = None  # 清除delta的梯度

                X_adv = X + delta[:X.size(0)]  # 对抗样本
                output = model(X_adv)

                # 计算模型输出的softmax（可选）
                output_softmax = F.softmax(output, dim=1)

                # **计算f1（正确类别与最高错误类别的概率比）**
                out = output.gather(1, y.unsqueeze(1)).squeeze()  # 取出正确类别的输出
                batch = []
                inds = []
                for idx in range(len(output)):
                    # 排除正确类别，找到最高的错误类别分数及索引
                    incorrect_logits = torch.cat([output[idx, :y[idx]], output[idx, y[idx]+1:]])
                    mm, ind = incorrect_logits.max(0)
                    f = torch.exp(out[idx]) / (torch.exp(out[idx]) + torch.exp(mm))
                    batch.append(f)
                    inds.append(ind.item())
                f1 = torch.stack(batch)

                # **计算分类损失**
                loss_cls = criterion(output, y)

                # **计算f1关于X_adv的梯度**
                grad_f1 = torch.autograd.grad(f1.sum(), X_adv, retain_graph=True)[0]
                v = grad_f1.detach().sign()  # 扰动方向

                # **生成扰动后的输入**
                h = 1e-2  # 扰动幅度，可根据需要调整
                X_adv2 = X_adv + h * v

                # **计算扰动后输入的模型输出**
                output2 = model(X_adv2)

                # **计算f2（扰动后正确类别与相同错误类别的概率比）**
                out2 = output2.gather(1, y.unsqueeze(1)).squeeze()
                batch = []
                for idx in range(len(output2)):
                    incorrect_logits = torch.cat([output2[idx, :y[idx]], output2[idx, y[idx]+1:]])
                    mm = incorrect_logits[inds[idx]]
                    f = torch.exp(out2[idx]) / (torch.exp(out2[idx]) + torch.exp(mm))
                    batch.append(f)
                f2 = torch.stack(batch)

                # **计算差分商和正则化损失**
                dl = (f2 - f1) / h
                loss_reg = dl.pow(2).mean()

                # **总损失**
                lamb = 0.1  # 正则化系数，可根据需要调整
                loss = loss_cls + lamb * loss_reg

                # **反向传播和优化**
                loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + epsilon * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                opt.step()
                delta.grad.zero_()
                scheduler.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            X.requires_grad_(False)  # 关闭X的梯度计算，以节省内存
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_train_time, lr, train_loss/train_n, train_acc/train_n)

    train_time = time.time()
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == "__main__":
    main()