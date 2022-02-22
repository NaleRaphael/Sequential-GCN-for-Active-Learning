'''
GCN Active Learning
'''

# Python
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import argparse
# Custom
import models.resnet as resnet
from models.resnet import vgg11
from models.query_models import LossNet
from train_test import train, test
from load_dataset import load_dataset
from selection_methods import query_samples
from config import *

from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-l","--lambda_loss",type=float, default=1.2, 
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d","--dataset", type=str, default="cifar100",
                    help="")
parser.add_argument("-e","--no_of_epochs", type=int, default=20,
                    help="Number of epochs for the active learner")
parser.add_argument("-m","--method_type", type=str, default="lloss",
                    help="")
parser.add_argument("-c","--cycles", type=int, default=5,
                    help="Number of active learning cycles")
parser.add_argument("-t","--total", type=bool, default=False,
                    help="Training on the entire dataset")

args = parser.parse_args()


# Copied from `train_test.py`
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


class ModelWrapper(nn.Module):
    # The following code follows this implementation:
    # https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning/blob/a0402725/train_test.py#L54
    def __init__(self, models, method):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.method = method

    def forward(self, inputs):
        scores, _, features = self.models['backbone'](inputs)

        if self.method == 'lloss':
            # Here we also return `features` since it would be used to calculate
            # another loss. See also here:
            # https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning/blob/a0402725/train_test.py#L82-L86
            # And since `LRFinder` expects the output of `model.forward()` is a
            # single object, we pack these outputs as a tuple. See also here:
            # https://github.com/davidtvs/pytorch-lr-finder/blob/acc5e7ee/torch_lr_finder/lr_finder.py#L377
            return (scores, features)
        else:
            return scores


class LossWrapper(nn.Module):
    def __init__(self, loss_func, models, method):
        super().__init__()
        self.loss_func = loss_func
        self.method = method
        if self.method == 'lloss' and 'module' not in models:
            raise ValueError('`models["module"]` is required when specified `method` is `lloss`.')
        self.models = models

    def forward(self, inputs, labels):
        # unpack
        if self.method == 'lloss':
            scores, features = inputs
        else:
            scores = inputs

        target_loss = criterion(scores, labels)

        if self.method == 'lloss':
            # NOTE: We just ignore this part currently since this wrapper is not used for real training.
            # if epoch > epoch_loss:
            #     features[0] = features[0].detach()
            #     features[1] = features[1].detach()
            #     features[2] = features[2].detach()
            #     features[3] = features[3].detach()

            pred_loss = self.models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            loss            = m_backbone_loss + WEIGHT * m_module_loss
        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            loss            = m_backbone_loss
        return loss


class OptimizerWrapper(optim.Optimizer):
    # Since the original implementation using the same configuration for 2 models (backbone and module),
    # we can adopt this setup to create a single optimizer:
    # https://pytorch.org/docs/stable/optim.html#per-parameter-options
    # See also the original implementation:
    # - optimizer for backbone:
    # https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning/blob/a0402725/main.py#L116-L117
    # - optimizer for module:
    # https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning/blob/a0402725/main.py#L123-L124
    #
    # However, if we want to use 2 different optimizers at the same time, `LRFinder` does not support
    # this scenario currently. You might need to consider using other hyperparameter search tools
    # (e.g., Bayesian optimization, hyperband)
    def __init__(self, optimizer_dict):
        optim_args = []
        for _optim in optimizer_dict.values():
            optim_args.append(_optim.param_groups[0])
        defaults = {k: v for k, v in optim_args[0].items() if k != 'params'}

        super().__init__(optim_args, defaults)

        self.optimizer_dict = optimizer_dict

    def step(self, *args, **kwargs):
        for v in self.optimizer_dict.values():
            v.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        for v in self.optimizer_dict.values():
            v.zero_grad(*args, **kwargs)

##
# Main
if __name__ == '__main__':

    method = args.method_type
    methods = ['Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL']
    datasets = ['cifar10', 'cifar100', 'fashionmnist','svhn']
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s'%(args.dataset, datasets)
    '''
    method_type: 'Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL'
    '''
    results = open('results_'+str(args.method_type)+"_"+args.dataset +'_main'+str(args.cycles)+str(args.total)+'.txt','w')
    print("Dataset: %s"%args.dataset)
    print("Method type:%s"%method)
    if args.total:
        TRIALS = 1
        CYCLES = 1
    else:
        CYCLES = args.cycles
    for trial in range(TRIALS):

        # Load training and testing dataset
        data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args.dataset)
        # Don't predefine budget size. Configure it in the config.py: ADDENDUM = adden
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)

        if args.total:
            labeled_set= indices
        else:
            labeled_set = indices[:ADDENDUM]
            unlabeled_set = [x for x in indices if x not in labeled_set]

        train_loader = DataLoader(data_train, batch_size=BATCH, 
                                    sampler=SubsetRandomSampler(labeled_set), 
                                    pin_memory=True, drop_last=True)
        test_loader  = DataLoader(data_test, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}

        for cycle in range(CYCLES):
            
            # Randomly sample 10000 unlabeled data points
            if not args.total:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:SUBSET]

            # Model - create new instance for every cycle so that it resets
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                if args.dataset == "fashionmnist":
                    resnet18    = resnet.ResNet18fm(num_classes=NO_CLASSES).cuda()
                else:
                    #resnet18    = vgg11().cuda() 
                    resnet18    = resnet.ResNet18(num_classes=NO_CLASSES).cuda()
                if method == 'lloss':
                    #loss_module = LossNet(feature_sizes=[16,8,4,2], num_channels=[128,128,256,512]).cuda()
                    loss_module = LossNet().cuda()

            models      = {'backbone': resnet18}
            if method =='lloss':
                models = {'backbone': resnet18, 'module': loss_module}
            torch.backends.cudnn.benchmark = True
            
            # Loss, criterion and scheduler (re)initialization
            criterion      = nn.CrossEntropyLoss(reduction='none')

            # NOTE: We wrote this part. Setup for optimizer and LR scheduler are separated now
            # because `LRFinder` have to run before the optimizer being attached with the actual
            # LR scheduler specified by user.
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                momentum=MOMENTUM, weight_decay=WDECAY)
            optimizers = {'backbone': optim_backbone}
            if method == 'lloss':
                optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                    momentum=MOMENTUM, weight_decay=WDECAY)
                optimizers = {'backbone': optim_backbone, 'module': optim_module}

            # ----- Code related to LRFinder ----
            model_wrapper = ModelWrapper(models, method)
            loss_wrapper = LossWrapper(criterion, models, method)
            optimizer_wrapper = OptimizerWrapper(optimizers)

            # Manually create an axis and pass it into `LRFinder.plot()` to avoid popping window
            # of figure blocking the procedure.
            fig, ax = plt.subplots()

            lr_finder = LRFinder(model_wrapper, optimizer_wrapper, loss_wrapper, device='cuda')
            lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
            ax, suggested_lr = lr_finder.plot(ax=ax, skip_start=0, skip_end=0, suggest_lr=True)

            # Uncomment this to save the result figure of range test to file
            # fig.savefig('lr_loss_history.png')

            # Remember to reset model and optimizer to original state
            lr_finder.reset()

            # Set suggested LR
            for name in optimizers:
                optimizers[name].param_groups[0]['lr'] = suggested_lr

            print('----- Updated optimizers -----')
            print(optimizers)
            # ^^^^^ Code related to LRFinder ^^^^^

            # Attach LR scheduler
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            schedulers = {'backbone': sched_backbone}
            if method == 'lloss':
                sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and testing
            train(models, method, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs, EPOCHL)
            acc = test(models, EPOCH, method, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))
            np.array([method, trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
            results.write("\n")


            if cycle == (CYCLES-1):
                # Reached final training cycle
                print("Finished.")
                break
            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args)

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) 
            unlabeled_set = listd + unlabeled_set[SUBSET:]
            print(len(labeled_set), min(labeled_set), max(labeled_set))
            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH, 
                                            sampler=SubsetRandomSampler(labeled_set), 
                                            pin_memory=True)

    results.close()
