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
parser.add_argument('--attack',default='fgsm',type = str)
parser.add_argument('--attack-iters',default=10,type = int)
parser.add_argument('--attack-restarts',default=1,type = int)
parser.add_argument('--out-logfile', default='test_output.log', type=str)
args = parser.parse_args()

model_path = args.model_path
if 'fgsm' in model_path: 
    ALGORITHM = 'FGSM'
else:
    ALGORITHM = 'Free'
os.environ["DATASET_NAME"] = args.dataset
from utils import (clamp, get_loaders, attack_pgd, evaluate_pgd, evaluate_standard,evaluate_attack)
# model_path = './train_alg2_output/model.pth'
if args.dataset.lower() in ('mnist','fashionmnist'):
	model_test = PreActResNet18(in_channel=1).cuda()
elif args.dataset.lower() in ('tinyimagenet'):
    model_test = PreActResNet18(in_channel=3,num_classes=200).cuda()
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
elif args.dataset.lower() == 'tinyimagenet':
     data_dir = '../../tinyimagenet'
     
_, test_loader = get_loaders(data_dir, BATCHSIZE)

from torchattacks import FGSM, PGD, BIM
if args.attack.lower() == 'fgsm':
    attack = FGSM(model_test, eps=8/255)
elif args.attack.lower() == 'pgd':
    attack = PGD(model_test, eps=8/255, alpha=2/255, steps=ATTACK_ITER)
elif args.attack.lower() == 'bim':
    attack = BIM(model_test, eps=8/255, alpha=2/255, steps=ATTACK_ITER)
pgd_acc = evaluate_attack(model_test,test_loader,attack)
pgd_loss = 0

# pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, ATTACK_ITER, RESTARTS)
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
logger.info('Algorithm {} \t Attack Method {} \t Dataset {}'.format(ALGORITHM,args.attack,args.dataset))
logger.info('Output model {}'.format(model_path))
logger.info('Batch size {} \t attack iteration {} \t restarts {}'.format(BATCHSIZE,ATTACK_ITER,RESTARTS))
logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
logger.info('{:.4f} \t \t {:.4f} \t {:.4f} \t {:.4f}'.format(test_loss, test_acc, pgd_loss, pgd_acc))