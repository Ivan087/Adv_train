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
# from svrg import SVRG
# from apex import amp

from preact_resnet import PreActResNet18
# from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
#     attack_pgd, evaluate_pgd, evaluate_standard)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
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
    if os.path.exists(logfile):
        os.remove(logfile)

    os.environ["DATASET_NAME"] = args.dataset

    from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std

    if args.dataset.lower() == 'mnist':
         model = PreActResNet18(in_channel=1).cuda()
    else:
        model = PreActResNet18().cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    # amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    # if args.opt_level == 'O2':
    #     amp_args['master_weights'] = args.master_weights
    # model, opt = amp.initialize(model, opt, **amp_args)
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
            if i == 0:
                first_batch = (X, y)
            if args.delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    # len(epsilon) == 3: initialize noise for each channel
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True

            ### Gradient one
            output = model(X + delta[:X.size(0)])

            # --- regularization ---
            output_softmax = F.softmax(output, dim=1)
            y_pred = output_softmax.gather(1, y.unsqueeze(1)).squeeze()
            # --- regularization end ---


            loss = F.cross_entropy(output, y)
            # with amp.scale_loss(loss, opt) as scaled_loss:
            #     scaled_loss.backward()

            # --- regularization ---
            model.zero_grad()
            loss.backward(retain_graph=True)
            dx = delta.grad.sign()
            # --- regularization end ---

            grad = delta.grad.detach()

            # --- regularization ---
            X_perturbed = X + 0.01 * dx
            output_perturbed = model(X_perturbed)
            y_pred_perturbed = F.softmax(output_perturbed, dim=1).gather(1, y.unsqueeze(1)).squeeze()

            regularization =  0.1 / 0.01 * (y_pred_perturbed - y_pred).abs()
            # --- regularization end ---


            ### Gradient two
            num_class = 10
            X_bar = torch.mean(X,axis=0,keepdim=True)
            y_bar = (torch.mean(y.float(),axis=0,keepdim=True)-1/num_class).long()
            delta_bar = torch.mean(delta,axis=0,keepdim=True)
            delta_bar.retain_grad()
            output = model(X_bar + delta_bar)
            # print(output.shape,y_bar.shape)
            loss = F.cross_entropy(output, y_bar)
            # print(loss)
            loss.backward()
            grad_bar = delta_bar.grad.detach().repeat(grad.size(0),1,1,1)

            ### Gradient Three
            avg_grad = torch.mean(grad,axis=0).repeat(grad.size(0),1,1,1)

            # Update delta for attack
            delta.data = clamp(delta + alpha * torch.sign(grad-grad_bar+avg_grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)

            delta = delta.detach()
            output = model(X + delta[:X.size(0)])

            ## forward training
            loss = criterion(output, y)
            opt.zero_grad()
            # with amp.scale_loss(loss, opt) as scaled_loss:
            #     scaled_loss.backward()
            # --- regularization ---
            total_loss = loss + regularization.mean()
            model.zero_grad()
            total_loss.backward()
            # --- regularization end ---
            
            opt.step()
            if i % 50 == 0:
                print('Epoch {} Iteration {} Loss {:.6f}'.format(epoch,i,loss.item()* y.size(0)))
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
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
    if args.dataset.lower() == 'mnist':
         model_test = PreActResNet18(in_channel=1).cuda()
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