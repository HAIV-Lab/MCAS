import os
import os.path as osp
import numpy as np
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from methods.MCAS.core import evaluation

from sklearn.metrics import average_precision_score
import math
from scipy.spatial.distance import cdist

def cos_similarity(a, b):
    # return np.dot(a, b) / np.sum(a)
    return np.dot(a, b) / (math.sqrt(np.sum(np.square(a))) * (math.sqrt(np.sum(np.square(b)))))


def get_attscore(pt_result, prototype):
    prototype = np.array(prototype)
    pt_result = sigmoid(pt_result)
    scorek = []
    for att in pt_result:
        score = []
        # att = 2 * (att - 0.5)
        for value in prototype:

            score.append(cos_similarity(value,att))
        scorek.append(score)
        # print(np.max(score))
    scorek = np.array(scorek)

    return scorek

def get_attscore_hire(pt_result, super_prototype,super_cls_pred_,super_mask,hyre_dict_idx,num_classes):
    pt_result = sigmoid(pt_result)
    scorek = []
    for ind,att in enumerate(pt_result):
        score = np.zeros((num_classes))
        super_cls_pre = super_cls_pred_[ind]
        mask = super_mask[super_cls_pre]
        fine_class = hyre_dict_idx[super_cls_pre]
        for idx in mask:
            att[idx] = 0
        for i,p in enumerate(super_prototype[super_cls_pre]):
            s = p.copy()
            for idx in mask:
                s[idx] = 0
            score[fine_class[i]] = cos_similarity(att, s)
        scorek.append(score)
    scorek = np.array(scorek)
    return scorek

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _scale(x, target_min, target_max):
    y = (x - x.min()) / (x.max() - x.min())
    y *= target_max - target_min
    y += target_min
    return y


def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))

    probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    return probabilities

import random
def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test_att(net, criterion, testloader, outloader, **options):
    set_random_seed(options['seed'])
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()
    att_criterion = F.binary_cross_entropy_with_logits
    _pred_k, _pred_u, _labels = [], [], []
    _att_k, _att_u = [], []
    right = 0
    with torch.no_grad():
        for data, labels, idx, att in tqdm(testloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
                att = att.cuda()
            with torch.set_grad_enabled(False):

                x, y, a = net(data, True)
                logits, loss_cls = criterion(x, y, labels)
                loss_att = att_criterion(a, att.float()) * att.shape[1]

                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
                right = int(right+correct.detach().cpu())
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())
                _att_k.append(a.data.cpu().numpy())
        try:
            for batch_idx, (data, labels, idx, att) in enumerate(tqdm(outloader)):
                if options['use_gpu']:
                    data, labels = data.cuda(), labels.cuda()

                with torch.set_grad_enabled(False):
                    x, y, a = net(data, True)
                    logits, _ = criterion(x, y)

                    _pred_u.append(logits.data.cpu().numpy())
                    _att_u.append(a.data.cpu().numpy())
        except:
            for batch_idx, (data, labels) in enumerate(tqdm(outloader)):
                if options['use_gpu']:
                    data, labels = data.cuda(), labels.cuda()

                with torch.set_grad_enabled(False):
                    x, y, a = net(data, True)
                    logits, _ = criterion(x, y)

                    _pred_u.append(logits.data.cpu().numpy())
                    _att_u.append(a.data.cpu().numpy())
    # Accuracy



    _att_k = np.concatenate(_att_k)
    _att_u = np.concatenate(_att_u)

    prototype = np.array(testloader.dataset.known_prototype.copy())

    hyre_dict_idx = testloader.dataset.hyre_dict_idx
    cls2super = {}
    for k, v in hyre_dict_idx.items():
        for item in v:
            cls2super[item] = k
    super_mask = testloader.dataset.super_mask
    super_prototype = testloader.dataset.hyre_dict_name_index
    
    att_logit_k = get_attscore(_att_k, prototype)
    att_logit_u = get_attscore(_att_u, prototype)

    _labels = np.concatenate(_labels, 0)

    cls_logit_k = np.concatenate(_pred_k, 0)
    cls_logit_u = np.concatenate(_pred_u, 0)


    cls_logit_k = softmax(cls_logit_k)
    cls_logit_u = softmax(cls_logit_u)

    
    tmpk = softmax(cls_logit_k)+att_logit_k
    tmpu = softmax(cls_logit_u) + att_logit_u
    cls_pred_k = [np.argmax(item) for item in tmpk]
    cls_pred_u = [np.argmax(item) for item in tmpu]
    super_cls_pred_k = [cls2super[item] for item in cls_pred_k]
    super_cls_pred_uk = [cls2super[item] for item in cls_pred_u]


    att_logit_k = get_attscore_hire(_att_k, super_prototype.copy(),super_cls_pred_k,super_mask,hyre_dict_idx,cls_logit_k.shape[1])
    att_logit_u = get_attscore_hire(_att_u, super_prototype.copy(), super_cls_pred_uk, super_mask, hyre_dict_idx,
                                    cls_logit_k.shape[1])

      
    aurlist = []
    oscrlist = []
    betalist = []

    _pred_k = cls_logit_k +  att_logit_k
    _pred_u = cls_logit_u +  att_logit_u

    prediction = [np.argmax(item) for item in _pred_k]
    acc = (prediction == _labels).sum() / len(_labels)
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    # OSCR
    aur, fpr = evaluation.compute_auroc(_pred_k, _pred_u, _labels)
    print('atthead:', 'ACC:', acc * 100, 'OSCR:', _oscr_socre * 100, 'FPR95:', fpr * 100, 'AUROC:', aur * 100)



def test_att_score_auroc(net, dataloader, **options):
    net.eval()
    _att_k =[]
    total=0
    correct=0
    with torch.no_grad():
        for data, labels, idx, att in dataloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
                att = att.cuda()
                x, y, a = net(data, True)
                _att_k.append(a.data.cpu().numpy())

                predictions = y.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

    _att_k = np.concatenate(_att_k, 0)
    _att_k = sigmoid(_att_k)
    _att_gt = np.array(dataloader.dataset.attributes)
    columns_pre = [_att_k[:, i] for i in range(_att_k.shape[1])]
    columns_gt = [_att_gt[:, i] for i in range(_att_gt.shape[1])]
    aur_score = []
    for i in range(len(columns_pre)):
        att_pre = columns_pre[i]
        att_gt = columns_gt[i]
        try:
            auroc = roc_auc_score(att_gt, att_pre)
            aur_score.append(auroc)
        except:
            aur_score.append(1.0)
    sorted_indices = np.argsort(aur_score)
    return np.mean(np.array(aur_score)),correct/total

def test_att_score_closeacc(net, dataloader, **options):
    net.eval()
    total = 0
    correct =0
    with torch.no_grad():
        for data, labels, idx, att in dataloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
                att = att.cuda()
                x, y  = net(data, True)
                predictions = y.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
    print(correct/total)
    return correct/total