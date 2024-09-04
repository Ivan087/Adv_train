import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from preact_resnet import PreActResNet18
# from utils import (clamp, get_loaders, attack_pgd, evaluate_pgd, evaluate_standard)
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model-path',help='checkpoint path to evaluate', type=str)
parser.add_argument('--batch-size',default=1024, type=int)
parser.add_argument('--dataset',default='cifar10',type = str)
parser.add_argument('--attack-iters',default=10,type = int)
parser.add_argument('--attack-restarts',default=1,type = int)
parser.add_argument('--out-logfile', default='test_output.log', type=str)
args = parser.parse_args()

model_path = args.model_path
os.environ["DATASET_NAME"] = args.dataset
from utils import (clamp, get_loaders, attack_pgd, evaluate_pgd, evaluate_standard)
# model_path = './train_alg2_output/model.pth'
if args.dataset.lower() in ('mnist','fashionmnist'):
	model_test = PreActResNet18(in_channel=1).cuda()
else:
    model_test = PreActResNet18().cuda()

model_test.load_state_dict(torch.load(model_path))
model_test.float()
model_test.eval()

BATCHSIZE = args.batch_size
ATTACK_ITER = args.attack_iters
RESTARTS = args.attack_restarts

if args.dataset.lower() ==  'cifar10':
    data_dir = '../../cifar-data'
elif args.dataset.lower() == 'mnist':
     data_dir = '../../mnist'
elif args.dataset.lower() == 'fashionmnist':
     data_dir = '../../fashionmnist'
     
train_loader, test_loader = get_loaders(data_dir, BATCHSIZE)
pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, ATTACK_ITER, RESTARTS)
test_loss, test_acc = evaluate_standard(test_loader, model_test)

logger = logging.getLogger(__name__)
logfile = args.out_logfile
logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
# logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
# logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
logger.info('')
logger.info('Output model {}'.format(model_path))
logger.info('Batch size {} \t attack iteration {} \t restarts {}'.format(BATCHSIZE,ATTACK_ITER,RESTARTS))
logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
logger.info('{:.4f} \t \t {:.4f} \t {:.4f} \t {:.4f}'.format(test_loss, test_acc, pgd_loss, pgd_acc))