## klrs
import argparse
import os
import random
import warnings
from sklearn.exceptions import DataConversionWarning
import sys
warnings.filterwarnings(action='ignore')
from torchvision import transforms, datasets
from myDatasets import IMBALANCECIFAR10, IMBALANCECIFAR100
import sys
import numpy as np
import torch, torch.nn.parallel, torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.utils.data
from torchvision import models
# from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from myutils import accuracy, saved_path_res, AverageMeter, save_checkpoint_epoch, get_tsne_of_sample_feature, \
    adjust_learning_rate, network_frozen, loaded_pretrained_models, save_best_checkpoint_epoch, \
    get_weights_of_majority_minority_class, get_wieghts_of_each_class,\
    load_checkpoint_iter, adjust_learning_rate, get_train_rule_hyperparameters
from myDatasets import get_num_classes, get_cls_num_list
from myDataLoader import get_train_val_test_loader
from mylosses import get_train_loss, CBCELoss, FocalLoss, LDAMLoss
import time
import models
import pandas as pd

from tqdm import tqdm



model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--results_dir', metavar="RESULTS_DIR", default='./PMAI_TrainingResults', help='pic_results dir')
parser.add_argument('--save', metavar='SAVE', default='', help='save folder')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('--model', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--loss_type', default="ce", type=str,
                    choices=['focal', 'ldam', 'abldam', 'abfocal', 'abce', 'abcet', 'ce', 'nebce', 'klrs', 'klrsfocal', 'klrsldam',\
                    'cvardro'], help='loss type')
parser.add_argument('--clip', default=True, type=eval)
parser.add_argument('--clip_threshold', default = 2, type=float)

parser.add_argument('--target', default=0.05, type=float)
parser.add_argument('--epsilon', default=0.1, type=float)
parser.add_argument('--labelbatch_size', default=10, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--subbatch_size', default= 100, type=int)
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=1, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, choices=['None', 'resample', 'reweight'],
                    help='data sampling strategy for train loader')

parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--bisec_freq', default=5, type=int)
parser.add_argument('--start_epochs', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--steps', default=50, type=int)

parser.add_argument('--samplebylabel', default = False, type=eval)

parser.add_argument('--lr', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--resume', default=0, type=int,
                    help='resume from which epoch')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=False, type=eval, choices=[True, False],
                    help='use pre-trained model')
parser.add_argument('--topK', default=None, type=int,
                    help='use pre-trained model')
parser.add_argument('--checkpoint_freq', default=5, type=int)
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpus', default='0', help='gpus used for training - e.g 0,1,2,3')
parser.add_argument('--load_checkpoint_epoch', default=None, type=int)

parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--store_name', type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--lamda', type=float, default=1)
parser.add_argument('--neb_tau', type=float, default=1, help='soft neighboring parameter')
parser.add_argument('--init_lamda', type=float, default=200)
parser.add_argument('--start_time', type=float, default=100)
parser.add_argument('--repeats', type=int, default=0)
parser.add_argument('--alg', type=str, help='Algorithm')
parser.add_argument('--gamma', type=float, default=1, help='smooth parameter of focal loss')
parser.add_argument('--drogamma', type=float, default=0.1, help='moving average parameter of ABSGD')
parser.add_argument('--alpha', type=float, default=1, help='balance parameter of focal loss')
parser.add_argument('--RENORM', default=True, type=eval, choices=[True, False],
                    help='Renormalized MSCGD or MSCGD')
parser.add_argument('--lamda_shots', type=int, default=160, help='Number of epochs to decrease lamda')
parser.add_argument('--CB_shots', type=int, default=60, help='Number of epochs to apply Class-Balanced Weighting')
parser.add_argument('--rho', default = 0.1, type=float)
parser.add_argument('--beta', default=0.9999, type=float, help=" beta in Reweighting")
parser.add_argument('--num_classes', default=10, type=int, help="classes of different datasets")
parser.add_argument('--cls_num_list', default=None, help="# of class distributions")
parser.add_argument('--frozen_aside_fc', default=False, type=eval, choices=[True, False],
                    help='whether frozen the feature layers (First three block)')
parser.add_argument('--not_frozen_last_block', default=False, type=eval, choices=[True, False],
                    help='whether frozen the feature layers (First three block)')

parser.add_argument('--abAlpha', default=0.5, type=float, help='Normalization Parameter for the Normalization Term')
parser.add_argument('--isTau', default=False, type=eval, choices=[True, False],
                    help='Whether Normalize the calssifier layer.')
parser.add_argument('--use_BN', default=False, type=eval, choices=[True, False],
                    help='Whether use BN before the fully connected layer.')
parser.add_argument('--ngroups', default=1, type=int, help='number of groups in a minibatch')
parser.add_argument('--option', default='I', type=str, help='Group Choice')
parser.add_argument('--train_defer', default=1, type=int, help='defer or not')
parser.add_argument('--DP', default=0, type=float, help='value of percentage of save samples in drop out')
parser.add_argument('--class_tau', default=0, type=float, help="# adaptive normalization for softamx")
parser.add_argument('--frozen_start', default=160, type=int,
                    help='# number of epochs that start to frozen the feature layers.')
parser.add_argument('--stage', default=1, type=int, help="which stage are you in by myself.")
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='types of tensor - e.g torch.cuda.FloatTensor')
parser.add_argument('--lr_schedule', default='epochLR', type=str, help="training straties")
parser.add_argument('--u', default=0, type=float, help="the average moving stochastic estimator ")
parser.add_argument('--res_name', default=None, type=str, help="results name of file")
parser.add_argument('--works', default=8, type=int, help='number of threads used for loading data')

def main():
    args = parser.parse_args()

    maxLambda = 200
    minLambda = 0.01

    best_acc1 = 0
    global best_prec1, z_t
    z_t = dict()
    
    overall_training_time = 0
       
    print(args)

    save_path, results = saved_path_res(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if 'cuda' in args.type:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.benchmark = True
        print("Use GPU: {} for training".format(args.gpus))
    else:
        args.gpus = None

    # create model
    print("=> creating model '{}'".format(args.model))
    use_norm = True if 'ldam' in args.loss_type else False
    args.num_classes = get_num_classes(args)
    print('numclasses: {}'.format(args.num_classes))
    # args.cls_num_list = get_cls_num_list(args)
    # np.save('places_LT.npy', args.cls_num_list)

    print('>>>>>>>>>>>>>> :', args.cls_num_list)
    print("Models Arch:", args.model)
    if 'cifar' in args.dataset:
        model = models.__dict__[args.model](num_classes=args.num_classes,  use_norm=use_norm)
    elif 'imagenet' in args.dataset:
        feat_dim = 2048
        use_fc_add = False
        model = models.__dict__[args.model](args.num_classes, pretrained=args.pretrained)
    elif 'places' in args.dataset:
        use_fc_add = False
        if args.stage == 2:
            feat_dim = 2048
        elif args.stage == 3:
            use_fc_add = True
            feat_dim = 512

        model = models.__dict__[args.model](args.num_classes, pretrained=args.pretrained, data=args.dataset,
                                            dropout=args.DP, use_BN=args.use_BN, isTau=args.isTau,
                                            use_fc_add=use_fc_add, feat_dim=feat_dim)

    if args.gpus and len(args.gpus) >= 1:
        print("We are running the model in GPU :", args.gpus)
        model = torch.nn.DataParallel(model)
        model.type(args.type)

    if 'klrs' in args.loss_type:
        if 'klrs'==args.loss_type:
            file = pd.read_csv("/home/yanhaojie/Document/project/SDS/cifar_experiment/PAMI_TrainingResults/"+str(args.dataset)+str(args.imb_type)+str(args.imb_factor)+"/ceFalsepretFalseresume0frozenFalse/lbth_10sbth_100/seed_"+str(args.seed)+'/ceFalse_pretFalseresume0frozenFalselbth_10_sbth_100_seed_'+str(args.seed)+'_results.csv')
        elif 'klrsfocal'==args.loss_type:
            file = pd.read_csv("/home/yanhaojie/Document/project/SDS/cifar_experiment/PAMI_TrainingResults/"+str(args.dataset)+str(args.imb_type)+str(args.imb_factor)+"/focalFalsepretFalseresume0frozenFalse/lbth_10sbth_100/seed_"+str(args.seed)+'/focalFalse_pretFalseresume0frozenFalselbth_10_sbth_100_seed_'+str(args.seed)+'_results.csv')
        elif 'klrsldam' == args.loss_type:
            file = pd.read_csv("/home/yanhaojie/Document/project/SDS/cifar_experiment/PAMI_TrainingResults/"+str(args.dataset)+str(args.imb_type)+str(args.imb_factor)+"/ldamFalsepretFalseresume0frozenFalse/lbth_10sbth_100/seed_"+str(args.seed)+'/ldamFalse_pretFalseresume0frozenFalselbth_10_sbth_100_seed_'+str(args.seed)+'_results.csv')

        index = int(args.resume/5-1)
        
        ermtrainloss = file['train_loss'][index]

        #print(ermtrainloss)
        #sys.exit() 
        args.target = ermtrainloss*(1+args.epsilon)
        print(f'target {args.target}')


    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.lr_schedule == 'coslr':  # learning rates: coslr or epochLR
        print("we are using CosineAnnealingLR")
        args.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0)
        print("Initial Learning rates for the first epochs:", args.scheduler.get_lr())
    if args.frozen_aside_fc:
        print("We are just training part of the neural network")
        network_frozen(args, model)
        print("frozen finished")


    # Load check points from certain number of epochs.
    if args.pretrained:
        loaded_pretrained_models(args, model)
        print("Pretrained Model Loaded Success.")
    if args.resume:
        if 'klrs'==args.loss_type:
            resumefile = "/home/yanhaojie/Document/project/SDS/cifar_experiment/PAMI_TrainingResults/"+str(args.dataset)+str(args.imb_type)+str(args.imb_factor)+"/ceFalsepretFalseresume0frozenFalse/lbth_10sbth_100/seed_"+str(args.seed)+'/'+str(args.resume)+'-th_epoch_checkpoint.pth.tar'
        elif 'klrsfocal' ==args.loss_type:
             resumefile = "/home/yanhaojie/Document/project/SDS/cifar_experiment/PAMI_TrainingResults/"+str(args.dataset)+str(args.imb_type)+str(args.imb_factor)+"/focalFalsepretFalseresume0frozenFalse/lbth_10sbth_100/seed_"+str(args.seed)+'/'+str(args.resume)+'-th_epoch_checkpoint.pth.tar'
        elif 'klrsldam'==args.loss_type:
            resumefile = "/home/yanhaojie/Document/project/SDS/cifar_experiment/PAMI_TrainingResults/"+str(args.dataset)+str(args.imb_type)+str(args.imb_factor)+"/ldamFalsepretFalseresume0frozenFalse/lbth_10sbth_100/seed_"+str(args.seed)+'/'+str(args.resume)+'-th_epoch_checkpoint.pth.tar'
        checkpointfile = torch.load(resumefile)
        args.start_epochs = checkpointfile['epoch']
        args.model = checkpointfile['model']
        args.labelbatch_size = checkpointfile['labelbatch_size']
        args.subbatch_size = checkpointfile['subbatch_size']
        model.load_state_dict(checkpointfile['statd_dict'])
        overall_training_time = checkpointfile['time']
        optimizer.load_state_dict(checkpointfile['optimizer_dict'])

    train_cls = []

    train_loader, val_loader, test_loader, trainval_loader = get_train_val_test_loader(args, train_sampler=None)
    """
    for i, (inputs, targets) in enumerate(train_loader):
        print(targets)
        sys.exit()
    """
    if args.cls_num_list is None:
        args.cls_num_list = get_cls_num_list(args)

    #inifiniteloaderlist = get_infinitetrainloader(args, train_sampler = None)

    criterion = get_train_loss(args, args.loss_type)

    test_loss, best_prec1, training_time, best_epoch, bestworstacc = 0, 0, 0, 0, 0
    CE_criterion = nn.CrossEntropyLoss(reduction='none')
    if test_loader is not None:
        _, pretrain_val_prec1, _, _, _, _, _, _, _, _, _, _, _ = validate(args, test_loader, model, criterion, args.start_epochs,
                                                              optimizer, None, args.init_lamda, True)
        print("pretrain_testl_prec1  {:.4f}".format(pretrain_val_prec1))
    else:
        _, pretrain_val_prec1, _, _, _, _, _, _, _, _, _, _, _ = validate(args, val_loader, model, criterion, args.start_epochs,
                                                              optimizer, None, args.init_lamda, True)
        print("pretrain_val_prec1  {:.4f}".format(pretrain_val_prec1))

    if 'klrs' in args.loss_type:
        myLambda = maxLambda
    else:
        myLambda = args.init_lamda
    
    # plt.figure()
    cls_probability = 0.1*torch.ones(10)
    # get_tsne_of_sample_feature(args, train_loader, model, 'train', 0)
    # get_tsne_of_sample_feature(args, val_loader, model, 'val', 0)
    # get_tsne_of_sample_feature(args, val_loader, model, 'test', 0)
    glossdict = {}
    gweightdict = {}
    gtrainaccdict = {}
    gvalaccdict = {}
    gvallossdict = {}


    start_time = time.time()
    for epoch in tqdm(range(args.start_epochs, args.epochs)):
        adjust_learning_rate(optimizer, epoch, args)
        
        #if  'klrs' in args.loss_type:
        #    lr=1e-3
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = lr
        
        print("lr : ", optimizer.param_groups[0]['lr'])

        cls_weights, _ = get_train_rule_hyperparameters(args, args.train_rule, epoch)

        _, _, _, epoch_training_time, _, _, _, _, _, _, _ = train(
            args, train_loader, model, criterion, epoch, optimizer, cls_weights, myLambda)
        overall_training_time += epoch_training_time


        if (epoch+1)%args.bisec_freq==0:

            train_loss, train_prec1, train_prec5, _, majority_train_loss, minority_train_loss,majority_P, minority_P, cls_p, trainlossdict, trainaccdict, trainclass_num, trainallloss = validate(args, trainval_loader, model, criterion, epoch, optimizer, None, myLambda, False)
            trainlosslist=[]
            trainacclist = []
            for j in range(args.num_classes):
                trainlosslist.append(trainlossdict[j].avg)
                trainacclist.append(trainaccdict[j].avg)

            val_loss, val_prec1, val_prec5, _, _, _, _, _, _, vallossdict, valaccdict, _,_ = validate(
                args, val_loader, model, criterion, epoch, optimizer, None, myLambda, True)
            if test_loader is not None:
                test_loss, test_prec1, test_prec5, _, _, _, _, _, _, testlossdict, testaccdict, _, _ = validate(
                    args, test_loader, model, criterion, epoch, optimizer, None, myLambda, True)
            train_cls.append(cls_p)
            
            testlosslist = []
            testacclist = []
            vallosslist = []
            valacclist = []
            for j in range(args.num_classes):

                if test_loader is None:
                    vallosslist.append(vallossdict[j].avg)
                    valacclist.append(valaccdict[j].avg)
                    
                else:
                    testlosslist.append(testlossdict[j].avg)
                    testacclist.append(testaccdict[j].avg)

            print("majority train loss:{}".format(majority_train_loss))

        if 'klrs' in args.loss_type:
            if (epoch + 1) % args.bisec_freq == 0:
                iniLambda = myLambda
                total_num = np.sum(trainclass_num)
                for j in range(args.num_classes):
                    if j ==0:
                        tiltedloss = trainclass_num[j]/total_num*np.exp(trainlosslist[j]/myLambda)
                    else:
                        tiltedloss += trainclass_num[j]/total_num*np.exp(trainlosslist[j]/myLambda)
                tiltedloss = myLambda*np.log(tiltedloss)

                print('epoch:{}, myLambda:{}, tiltedloss:{}'.format(epoch, myLambda, tiltedloss))
                if tiltedloss < args.target:
                    lowLambda = minLambda
                    highLambda = maxLambda
                    while lowLambda / highLambda < 0.99:
                        tempLambda = (highLambda + lowLambda) / 2

                        for j in range(args.num_classes):
                            if j ==0:
                                temptiltedloss = trainclass_num[j]/total_num*np.exp(trainlosslist[j]/tempLambda)
                            else:
                                temptiltedloss += trainclass_num[j]/total_num*np.exp(trainlosslist[j]/tempLambda)
                        temptiltedloss = tempLambda*np.log(temptiltedloss)
                        if temptiltedloss < args.target:
                            myLambda = tempLambda
                            highLambda = tempLambda
                        else:
                            lowLambda = tempLambda

                for j in range(args.num_classes):
                    if j ==0:
                        tiltedloss = trainclass_num[j]/total_num*np.exp(trainlosslist[j]/myLambda)
                    else:
                        tiltedloss += trainclass_num[j]/total_num*np.exp(trainlosslist[j]/myLambda)
                tiltedloss = myLambda*np.log(tiltedloss)
                print('epoch:{}, iniLambda:{}, myLambda:{}, tiltedloss:{}'.format(epoch, iniLambda, myLambda,
                                                                                  tiltedloss))
        if 'abce' in args.loss_type:
            if (epoch+1)>=args.lamda_shots:
                myLambda = args.lamda 

        if (epoch + 1) % args.checkpoint_freq == 0:
            save_checkpoint_epoch({
                'epoch': epoch+1,
                'model': args.model,
                'labelbatch_size': args.labelbatch_size,
                'subbatch_size': args.subbatch_size,
                'statd_dict': model.state_dict(),
                'time': overall_training_time,
                'optimizer_dict':optimizer.state_dict()},
                is_best=False, path=save_path
            )

            valaverageacc = sum(valacclist)/len(valacclist) if test_loader is None else sum(testacclist)/len(testacclist)
            valaverageloss = sum(vallosslist)/len(vallosslist) if test_loader is None else sum(testlosslist)/len(testlosslist)
            trainworstacc = min(trainacclist)
            trainbestacc = max(trainacclist)
            trainworstloss = max(trainlosslist)
            trainbestloss = min(trainlosslist)
            valworstacc = min(valacclist)
            valbestacc = max(valacclist)
            valworstloss = max(vallosslist)
            valbestloss = min(vallosslist)
            testworstacc = min(valacclist) if test_loader is None else min(testacclist)
            testbestacc = max(valacclist) if test_loader is None else max(testacclist)
            testworstloss = max(vallosslist) if test_loader is None else max(testlosslist)
            testbestloss = min(vallosslist) if test_loader is None else min(testlosslist) 
            tmp_prec1 = val_prec1 if test_loader is None else test_prec1
            is_best = testworstacc > bestworstacc
            print(">>>>>>>>>>>>> :", is_best, ": <<<<<<<<<<<<<<")
            if is_best:
                bestworstacc = testworstacc
                best_epoch = epoch+1
            if is_best:
                save_best_checkpoint_epoch({
                    'epoch': epoch,
                    'model': args.model,
                    'labelbatch_size': args.labelbatch_size,
                    'subbbatch_size': args.subbatch_size,
                    'state_dict': model.module.state_dict(),
                    'time': overall_training_time
                }, is_best=is_best, path=save_path)
            
            print('train loss list :{}'.format(trainlosslist))
            tempweight = []
            if myLambda<200:
                tempweight = np.exp(np.array(trainlosslist)/myLambda)/np.exp(np.array(trainlosslist)/myLambda).sum()
            else:
                tempweight = 1/args.num_classes*np.ones(args.num_classes)
            glossdict[epoch+1] = trainlosslist
            gweightdict[epoch+1] = tempweight
            gtrainaccdict[epoch+1] = trainacclist
            gvalaccdict[epoch+1] = valacclist
            gvallossdict[epoch+1] = vallosslist
            print('train class weight: {}'.format(tempweight))

            print('\n epoch: {0}\t'
                    'Train Loss {train_loss:.4f} \t'
                    'Train Worst Loss {train_worst_loss:.4f} \t'
                    'Train Best Loss {train_best_loss:.4f} \t'
                    'Train Prec@1 {train_prec1:.3f} \t'
                    'Train Worst Prec@1 {train_worst_prec1:.3f} \t'
                    'Train Best Prec@1 {train_best_prec1:.3f} \n'

                    'Validation Loss {val_loss:.4f} \t'
                    'Validation Worst Loss {val_worst_loss:.4f} \t'
                    'Validation Best Loss {val_best_loss:.4f} \t'
                    'Validation Prec@1 {val_prec1:.3f} \t'
                    'Validation Worst Prec@1 {val_worst_prec1:.3f} \t'
                    'Validation Best Prec@1 {val_best_prec1:.3f} \n'                

                    'Test Loss {test_loss:.4f} \t'
                    'Test Worst Loss {test_worst_loss:.4f} \t'
                    'Test Best Loss {test_best_loss:.4f} \t'
                    'Test Prec@1 {test_prec1:.3f} \t'
                    'Test Worst Prec@1 {test_worst_prec1:.3f} \t'
                    'Test Best Prec@1 {test_best_prec1:.3f} \n'
                    
                    
                    'Best Worst Class Acc {best_worst_class_acc:.3f} \t'   
                    'Best epoch {best_epoch:.3f} \n'

                    'myLambda {myLambda_value:.4f} \n'
                    .format(epoch + 1, train_loss=train_loss, train_worst_loss= trainworstloss, train_best_loss = trainbestloss,
                            train_prec1=train_prec1, train_worst_prec1 = trainworstacc, train_best_prec1 = trainbestacc,
                            val_loss=val_loss, val_worst_loss = valworstloss, val_best_loss = valbestloss,
                            val_prec1=val_prec1, val_worst_prec1 = valworstacc, val_best_prec1 = valbestacc, 
                            test_loss=test_loss if test_loader is not None else val_loss,
                            test_worst_loss = testworstloss if test_loader is not None else valworstloss,
                            test_best_loss = testbestloss if test_loader is not None else valbestloss,
                            test_prec1=test_prec1 if test_loader is not None else val_prec1,
                            test_worst_prec1 = testworstacc if test_loader is not None else valworstacc,
                            test_best_prec1 = testbestacc if test_loader is not None else valbestacc,
                            best_worst_class_acc=bestworstacc, best_epoch=best_epoch, myLambda_value=myLambda))
            results.add(epoch=epoch + 1, train_loss=train_loss, train_worst_loss= trainworstloss, train_best_loss = trainbestloss,
                        train_prec1=train_prec1, train_worst_prec1 = trainworstacc, train_best_prec1 = trainbestacc,
                        val_loss=val_loss, val_worst_loss = valworstloss, val_best_loss = valbestloss,
                        val_prec1=val_prec1, val_worst_prec1 = valworstacc, val_best_acc = valbestacc,
                        test_loss=test_loss if test_loader is not None else val_loss,
                        test_worst_loss = testworstloss if test_loader is not None else valworstloss,
                        test_best_loss = testbestloss if test_loader is not None else valbestacc,
                        test_prec1=test_prec1 if test_loader is not None else val_prec1,
                        test_worst_prec1 = testworstacc if test_loader is not None else valworstacc,
                        test_best_prec1 =testbestacc if test_loader is not None else valbestacc,
                        best_worst_class_acc = bestworstacc, best_epoch=best_epoch, myLambda_value = myLambda,
                        overall_training_time=overall_training_time // 60, lr = optimizer.param_groups[0]['lr'])
            results.save()
            glosspd = pd.DataFrame.from_dict(glossdict, orient='index')
            glosspd.to_csv(args.root_log+args.res_name+'_train_class_loss.csv')
            gweightpd = pd.DataFrame.from_dict(gweightdict, orient='index')
            gweightpd.to_csv(args.root_log+args.res_name+'_class_weight.csv')

            gtrainaccpd = pd.DataFrame.from_dict(gtrainaccdict, orient='index')
            gtrainaccpd.to_csv(args.root_log+args.res_name+'_train_class_acc.csv')

            gvalaccpd = pd.DataFrame.from_dict(gvalaccdict, orient = 'index')
            gvalaccpd.to_csv(args.root_log+args.res_name+'_val_class_acc.csv')

            gvallosspd = pd.DataFrame.from_dict(gvallossdict, orient = 'index')
            gvallosspd.to_csv(args.root_log+args.res_name+'_val_class_loss.csv')
    end_time = time.time()

    train_time =end_time-start_time

    print("The training time for "+str(args.loss_type) +": "+str(overall_training_time))

    

    if myLambda >= 200:
        print("We use the method of SGD")
    else:
        print("We implement DRO with lambd : ", myLambda)


def forward(args, data_loader, model, criterion, epoch, optimizer, cls_weights, myLambda=0, trueval=True, training=True):
    run_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    majority_losses = AverageMeter('Loss', ':.4e')
    minority_losses = AverageMeter('Loss', ':.4e')
    majority_P = AverageMeter('P', ':.4e')
    minority_P = AverageMeter('P', ':.4e')
    lossdict = {}
    accdict ={}
    for j in range(args.num_classes):
        lossdict[j] = AverageMeter('Loss', ':.4e')
        accdict[j] = AverageMeter('Acc', ':.6f')

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    covid_top1 = AverageMeter('COVID-Acc@1', ':6.2f')
    end = time.time()
    majP, minP = 0, 0
    cls_p = None
    weights = 1/args.num_classes*np.ones(args.num_classes)

    if training:
        for i, (inputs, targets) in enumerate(data_loader):
            #print(targets)
            #l = []
            #for j in range(args.labelbatch_size):
            #    l.append(targets[j*args.subbatch_size])
            #print(l)
            inputs, targets = inputs.cuda(), targets.cuda()
 
            outputs = model(inputs)
        
            outputs = outputs / (torch.norm(outputs, p=2, dim=1, keepdim=True) ** args.class_tau)

            if 'ab' in args.loss_type:
                loss = criterion(outputs, targets, cls_weights, myLambda)
                #args.u = criterion.u
            elif 'klrs' in args.loss_type:
                loss = criterion(outputs, targets, cls_weights, myLambda)

            elif 'cvardro' in args.loss_type:
                loss = criterion(outputs, targets, cls_weights)
            else:
                loss = criterion(outputs, targets, None)
                loss = loss.mean()
            # accuracy

            if args.num_classes >= 5:
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            else:
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 3))
            # 损失函数，是直接根据目标损失函数的那个来进行计算的。
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0].item(), inputs.size(0))
            top5.update(acc5[0].item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_time.update(time.time() - end)
            end = time.time()
            
        
       
        return losses.avg, top1.avg, top5.avg, run_time.sum, majority_losses.avg, minority_losses.avg, majority_P.avg, minority_P.avg, cls_p, lossdict, accdict


    else:
        all_preds = []
        all_targets = []
        all_losses = []
        if trueval == True:
            model.eval()

        elif trueval == False:
            model.train()
            #model.apply(fix_bn)
        class_num = np.zeros(args.num_classes)
        
        for i, (inputs, targets) in enumerate(data_loader):

            inputs, targets = inputs.cuda(), targets.cuda()
            #print(targets)
            with torch.no_grad():
                outputs = model(inputs)

                if 'ldam' in args.loss_type:
                    tempcriterion = LDAMLoss(cls_num_list=args.cls_num_list, max_m=0.5, s=30)
                    loss = tempcriterion(outputs, targets, cls_weights)
                elif 'focal' in args.loss_type:
                    tempcriterion = FocalLoss(gamma=1)
                    loss = tempcriterion(outputs, targets, cls_weights)
                else:
                    tempcriterion = CBCELoss(reduction='none')
                    loss = tempcriterion(outputs, targets, cls_weights)


                #print('val', loss)
                if i==0:
                    trainallloss = loss
                else:
                    trainallloss = torch.cat((trainallloss, loss), 0) 

            for j in range(args.num_classes):
                index = torch.where(targets==j)[0]
                if len(index)!=0:
                
                    lossdict[j].update(loss[index].mean().item(), len(index))
                    class_num[j] = class_num[j]+len(index)
                
                    tempprec1 = accuracy(outputs[index].data, targets[index], topk=(1, ))
                #print('tempprec1:{}'.format(tempprec1[0].item()))
                    accdict[j].update(tempprec1[0].item(), len(index))

            if type(outputs) is list:
                outputs = outputs[0]
            if args.num_classes >= 5:
                prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            else:
                prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 3))

            losses.update(loss.mean().item(), inputs.size(0))

                # print(majP, minP)

            # print(">>>>>>>>>>>>>:", minority_loss.size(), ">>>>>>>>>>>>>:",)
            top1.update(prec1[0].item(), inputs.size(0))
            top5.update(prec5[0].item(), inputs.size(0))
            run_time.update(time.time() - end)
            end = time.time()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_losses.extend(loss.detach().cpu().numpy())



        print('{phase} - Step: [{0}/{1}]\t'
            'Data {run_time.sum:.3f} ({run_time.val:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        epoch, args.epochs,
        phase='TRAINING' if training else 'EVALUATING',
        run_time=run_time, loss=losses, top1=top1, top5=top5))
        # wandb.log({"iter val loss": losses.avg, 'iter val acc1': top1.avg, 'iter val acc5': top5.avg})
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('epoch: {epoch} {flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(epoch=epoch, flag=training, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
        training, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        # print('Class Average Acc : ', out_cls_acc)
        print('args.u | ', args.u, ' myLambda | ', myLambda)
        #print(all_losses)
        #print(myLambda)

        return losses.avg, top1.avg, top5.avg, run_time.sum, majority_losses.avg,\
               minority_losses.avg, majority_P.avg, minority_P.avg, cls_p, lossdict, accdict, \
                class_num, trainallloss


def train(args, data_loader, model_new, criterion, epoch, optimizer, cls_weights, myLambda):
    model_new.train()
    return forward(args, data_loader, model_new, criterion, epoch, optimizer, cls_weights, myLambda, trueval=True, training=True)

def validate(args, data_loader, model_new, criterion, epoch, optimizer, cls_weights, myLambda, trueval):
    # switch to evaluate model
    model_new.eval()
    return forward(args, data_loader, model_new, criterion, epoch, optimizer, cls_weights, myLambda, trueval, training=False)


if __name__ == '__main__':
    main()


