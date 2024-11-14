import argparse
import copy
import logging
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dct import *
#from apex import amp

from preact_resnet import PreActResNet18

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='fast_reg_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt', default='Adam', type=str)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument('--rho', type=float, default=0.5,
        help='Parameter for SSA modify the uniform mask')
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
    
    os.environ["DATASET_NAME"] = args.dataset
    from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)

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
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std
    if args.dataset.lower() in ('mnist','fashionmnist'):
         model = PreActResNet18(in_channel=1).cuda()
    elif args.dataset.lower() in ('tinyimagenet'):
        model = PreActResNet18(in_channel=3,num_classes=200).cuda()
    else:
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

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(tqdm(train_loader)):
            X, y = X.cuda(), y.cuda()
            # X_adv = X
            if i == 0:
                first_batch = (X, y)

            # Clone X to avoid altering its computation graph during updates
            X_adv = X.detach().clone()  # Clone X and enable gradient calculation
            X_adv.requires_grad_()

            # Initialize perturbation delta
            if args.delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()

            # Initialize delta as random if specified
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                        delta[:, j, :, :].normal_(mean=0., std=epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                # if args.ssa:
                #     ## TODO: initialize delta with Gaussian faster
                #     for j in range(len(epsilon)):
                #         delta[:, j, :, :].normal_(mean=0., std=epsilon[j][0][0].item())
                # else:
                #     for j in range(len(epsilon)):
                #         delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                # delta.data = clamp(delta, lower_limit - X, upper_limit - X)

            delta.requires_grad = True
            x_dct = dct_2d(X_adv+delta).cuda()
            mask = (torch.rand_like(x_dct)* 2 * args.rho + 1 - args.rho).cuda()
            x_idct = idct_2d(mask * x_dct)
            output = model(x_idct)
            # Forward pass with perturbed inputs
            # output = model(X_adv + delta[:X.size(0)])
            loss = F.cross_entropy(output, y)

            # Compute gradients
            loss.backward(retain_graph=True)  # Retain the computation graph for further use
            grad = delta.grad.detach()

            # Update perturbation delta with sign of gradients (PGD update rule)
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()

            # Forward pass with updated perturbations
            output = model(X_adv + delta[:X.size(0)])
            loss = criterion(output, y)

            # Compute gradient of model parameters
            opt.zero_grad()
            loss.backward(retain_graph=True)  # Retain the graph for the next backward pass

            # Apply Sparse Regularization (SR)
            grad_x = torch.autograd.grad(loss, X_adv, create_graph=True)[0]  # Compute gradient w.r.t. X_adv
            grad_norm_1 = torch.sum(torch.abs(grad_x), dim=(1, 2, 3))  # L1 norm per sample

            # Add the regularization term to the loss
            reg_loss = torch.mean(grad_norm_1)  # Mean L1 norm across the batch
            lambda_reg = 0.01
            total_loss = loss + lambda_reg * reg_loss

            # Perform the optimization step
            opt.zero_grad()
            total_loss.backward()
            opt.step()

            # Logging the training statistics
            train_loss += total_loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            # Step the scheduler
            scheduler.step()



        if args.early_stop:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.2:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
    train_time = time.time()
    if not args.early_stop:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
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
