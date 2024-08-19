import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore')
from collections import Counter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from myutils import *
from sklearn.decomposition import PCA
import torch

import joint_dro


#
# def my_warn():
#     pass
# warnings.warn = my_warn
#
class KLRS(nn.Module):
    def __init__(self, args, budget, loss_type, abAlpha=1):
        super(KLRS, self).__init__()
        self.loss_type= loss_type
        self.clip = args.clip
        self.clip_threshold = args.clip_threshold
        self.gamma= args.drogamma
        self.budget = budget
        self.robAlpha = abAlpha
        self.criterion = CBCELoss(reduction='none')
        self.num_classes = args.num_classes
        self.gpus = args.gpus
        if 'ldam' in self.loss_type:
            self.criterion = LDAMLoss(cls_num_list=args.cls_num_list, max_m=0.5, s=30, reduction='none')
        elif 'focal' in self.loss_type:
            self.criterion = FocalLoss(gamma=args.gamma, reduction='none')
    def forward(self, output, target, cls_weights, myLambda):
        datalist = []
        labellist = []
        loss = self.criterion(output, target, cls_weights)
        #print('loss is', loss)
        #print('mylambda is {}'.format(myLambda))
        if myLambda>=200:
            p= 1/len(loss)
            abloss = torch.sum(p*loss)
        else: 
            plist = []
            class_loss = []
            for i in range(self.num_classes):
                index = torch.where(target==i)
                if len(index)!=0:
                    with torch.no_grad():
                        if self.clip ==True:
                            p = torch.exp(torch.min((torch.mean(loss[index])-self.budget)/myLambda, torch.tensor(self.clip_threshold)))
                        else:
                            p = torch.exp((torch.mean(loss[index])-self.budget)/myLambda)
                    p = p.detach_()
                    plist.append(p)
                    class_loss.append(torch.mean(loss[index]).detach_())
                    
                    if i ==0:
                        abloss = p*torch.mean(loss[index])*torch.tensor(len(index))
                    else:
                        abloss += p*torch.mean(loss[index])*torch.tensor(len(index))
                else:
                    abloss += 0
            abloss = abloss/len(target)
        return abloss


def get_train_loss(args, loss_type):
    if args.loss_type == 'ce':
        criterion = CBCELoss()
    elif args.loss_type == 'ldam':
        criterion = LDAMLoss(cls_num_list=args.cls_num_list, max_m=0.5, s=30)
    elif args.loss_type == 'focal':
        criterion = FocalLoss(gamma=1)
    elif 'klrs' in args.loss_type:
        criterion = KLRS(args, args.target, args.loss_type, args.abAlpha)
    elif 'cvardro' in args.loss_type:
        criterion = CVaRDRO(args, args.loss_type)
    else:
        warnings.warn('Loss type is not listed')
        return

    return criterion

class CVaRDRO(nn.Module):
    def __init__(self, args, loss_type):
        super(CVaRDRO, self).__init__()
        self.loss_type = loss_type
        self.criterion = CBCELoss(reduction = 'none')
        self.rho = args.rho
        self._joint_dro_loss_computer= joint_dro.RobustLoss(self.rho, 0, 'cvar')
        if 'ldam' in self.loss_type:
            self.criterion = LDAMLoss(cls_num_list=args.cls_num_list, max_m=0.5, s=30, reduction='none')
        elif 'focal' in self.loss_type:
            self.criterion = FocalLoss(gamma=args.gamma, reduction='none')
    
    def forward(self, output, target, cls_weights):
        #print('======cvarupdating=======')
        loss = self.criterion(output, target, cls_weights)
        actual_loss = self._joint_dro_loss_computer(loss)
        return actual_loss

def focal_loss(input_values, alpha, gamma, reduction='mean'):
    """Computes the focal loss"""

    '''
    input_values = -\log(p_t)
    loss = - \alpha_t (1-\p_t)\log(p_t)
    '''
    p = torch.exp(-input_values)
    loss = alpha * (1 - p) ** gamma * input_values

    if reduction == 'none':
        return loss
    else:
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, reduction='none'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, input, target, weight=None):
        cbloss = CBCELoss(reduction='none')
        return focal_loss(cbloss(input, target, weight=weight), self.alpha, self.gamma,
                          reduction=self.reduction)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30, reduction='mean'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))  # 1/n_j^{1/4}
        m_list = m_list * (max_m / np.max(m_list))  # control the length of margin
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.reduction = reduction
        self.cbloss = CBCELoss(reduction='none')
    def forward(self, output, target, weight='none'):

        index = torch.zeros_like(output, dtype=torch.uint8)
        #target = target.type(torch.cuda.LongTensor)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = output - batch_m

        output = torch.where(index, x_m, output)

        return self.cbloss(self.s * output, target, weight=weight)

class CBCELoss(nn.Module):
    def __init__(self, reduction='none'):
        super(CBCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, out, target, weight=None):
        criterion = nn.CrossEntropyLoss(weight=weight, reduction=self.reduction)
        cbloss = criterion(out, target)
        return cbloss

