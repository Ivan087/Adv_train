import argparse
import logging
import os
import time
from tqdm import tqdm
# import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dct import *
from preact_resnet import PreActResNet18


logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=10, type=int, help='Total number of epochs will be this argument * number of minibatch replays.')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.04, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--minibatch-replays', default=8, type=int)
    parser.add_argument('--out-dir', default='train_free_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument('--ssa', action='store_true',
        help='Whether to inculde SSA to process the feature maps')
    parser.add_argument('--rho', type=float, default=0.5,
        help='Parameter for SSA modify the uniform mask')
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)
        
    os.environ["DATASET_NAME"] = args.dataset
    from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard)
    
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std

    if args.dataset.lower() in ('mnist','fashionmnist'):
         model = PreActResNet18(in_channel=1).cuda()
    elif args.dataset.lower() in ('tinyimagenet'):
        model = PreActResNet18(in_channel=3,num_classes=200).cuda()
    else:
        model = PreActResNet18().cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    # amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    # if args.opt_level == 'O2':
    #     amp_args['master_weights'] = args.master_weights
    # model, opt = amp.initialize(model, opt, **amp_args)
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
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(tqdm(train_loader)):
            X, y = X.cuda(), y.cuda()
            if args.delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                if args.ssa:
                    ## TODO: initialize delta with Gaussian faster
                    for j in range(len(epsilon)):
                        delta[:, j, :, :].normal_(mean=0., std=epsilon[j][0][0].item())
                else:
                    for j in range(len(epsilon)):
                        delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True      
            for _ in range(args.minibatch_replays):
                if args.ssa:
                    x_dct = dct_2d(X+delta).cuda()
                    mask = (torch.rand_like(x_dct)* 2 * args.rho + 1 - args.rho).cuda()
                    x_idct = idct_2d(mask * x_dct)
                    output = model(x_idct)
                else:
                    output = model(X + delta[:X.size(0)])
                loss = criterion(output, y)
                opt.zero_grad()
                # with amp.scale_loss(loss, opt) as scaled_loss:
                #     scaled_loss.backward()
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
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
    train_time = time.time()
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    # best_state_dict = torch.load(os.path.join(args.out_dir, 'model.pth'))
    best_state_dict = model.state_dict()
    if args.dataset.lower() in ('mnist','fashionmnist'):
        model_test = PreActResNet18(in_channel=1).cuda()
    elif args.dataset.lower() in ('tinyimagenet'):
        model_test = PreActResNet18(in_channel=3,num_classes=200).cuda()
    else:
        model_test = PreActResNet18().cuda()
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 1, 1)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == "__main__":
    main()
