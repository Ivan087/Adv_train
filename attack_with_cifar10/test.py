import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from preact_resnet import PreActResNet18
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)

model_path = './train_alg2_output/model.pth'
model_test = PreActResNet18().cuda()
model_test.load_state_dict(torch.load(model_path))
model_test.float()
model_test.eval()

BATCHSIZE = 1024
ATTACK_ITER = 20
RESTARTS = 5

train_loader, test_loader = get_loaders('../../cifar-data', 1024)
pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, ATTACK_ITER, RESTARTS)
test_loss, test_acc = evaluate_standard(test_loader, model_test)

logger = logging.getLogger(__name__)
logfile = 'test_output.log'
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