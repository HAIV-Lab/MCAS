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


def test(net, criterion, testloader, outloader, epoch=None, **options):

    net.eval()
    correct, total = 0, 0
    # attmask = options['att_mask']
    torch.cuda.empty_cache()
    if options['use_attribute'] == True:
        att_criterion = F.binary_cross_entropy_with_logits
    _pred_k, _pred_u, _labels = [], [], []
    _att_k ,_att_u = [], []
    with torch.no_grad():
        for data, labels, idx, att in tqdm(testloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
                att = att.cuda()
            with torch.set_grad_enabled(False):
                if options['use_attribute'] == False:
                    x, y = net(data, True)
                    logits, loss = criterion(x, y, labels)

                else:
                    if options['use_attribute_only'] == False:
                        x, y, a = net(data, True)
                        logits, loss_cls = criterion(x, y, labels)
                        loss_att = att_criterion(a, att.float()) * att.shape[1]
                    else:
                        x, y, a = net(data, True)
                if options['use_attribute_only'] == False:
                    predictions = logits.data.max(1)[1]
                    total += labels.size(0)
                    correct += (predictions == labels.data).sum()
                    if options['use_softmax_in_eval']:
                        logits = torch.nn.Softmax(dim=-1)(logits)

                    _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())
                if options['use_attribute'] == True:
                    _att_k.append(a.data.cpu().numpy())
        for batch_idx, (data, labels, idx,att) in enumerate(tqdm(outloader)):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                if options['use_attribute'] == False:
                    x, y = net(data, True)

                    logits, _ = criterion(x, y)
                else:
                    if options['use_attribute_only'] == False:
                        x, y, a = net(data, True)

                        logits, _ = criterion(x, y)
                    else:
                        x, y, a = net(data, True)
                if options['use_attribute_only'] == False:
                    if options['use_softmax_in_eval']:
                        logits = torch.nn.Softmax(dim=-1)(logits)

                    _pred_u.append(logits.data.cpu().numpy())
                if options['use_attribute'] == True:
                    _att_u.append(a.data.cpu().numpy())
    # Accuracy
    if options['use_attribute'] == False:
        Acc = float(correct)  / float(total)
        print('Acc: {:.5f}'.format(Acc))

        _pred_k = np.concatenate(_pred_k, 0)
        _pred_u = np.concatenate(_pred_u, 0)
        _labels = np.concatenate(_labels, 0)

        # Out-of-Distribution detction evaluation
        x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
        results = evaluation.metric_ood(x1, x2)['Bas']

        # OSCR
        Oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)
        Aur, Fpr = evaluation.compute_auroc(_pred_k, _pred_u, _labels)
        # Average precision
        ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u),
                                           list(-np.max(_pred_k, axis=-1)) + list(-np.max(_pred_u, axis=-1)))
    else:
        _att_k = np.concatenate(_att_k)
        _att_u = np.concatenate(_att_u)
        prototype = np.array(testloader.dataset.known_prototype)

        att_logit_k = get_attscore(_att_k, prototype)
        att_logit_u = get_attscore(_att_u, prototype)


        _labels = np.concatenate(_labels, 0)
        if options['use_attribute_only'] == False:
            cls_logit_k = np.concatenate(_pred_k, 0)
            cls_logit_u = np.concatenate(_pred_u, 0)
            bestscore=-99

            _pred_k = att_logit_k
            _pred_u = att_logit_u
            aur, fpr = evaluation.compute_auroc(_pred_k, _pred_u, _labels)
            print(aur)
            for beta in range(0, 100, 1):  # in_score + 1.5*
                # beta = 0
                # Out-of-Distribution detction evaluation
                _pred_k = cls_logit_k + beta * att_logit_k
                _pred_u = cls_logit_u + beta * att_logit_u

                prediction = [np.argmax(item) for item in _pred_k]
                acc = (prediction == _labels).sum() / len(_labels)
                _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

                # OSCR
                aur, fpr = evaluation.compute_auroc(_pred_k, _pred_u, _labels)


                score = acc + aur + _oscr_socre - fpr
                if aur > bestscore:
                    Aur = aur
                    Acc = acc
                    Oscr_socre = _oscr_socre
                    Fpr = fpr
                    bestbeta = beta
                    bestscore = aur
                # print(beta, aur)
            print('best_beta:' , bestbeta)
        else:
            _pred_k = att_logit_k
            _pred_u =  att_logit_u

            prediction = [np.argmax(item) for item in _pred_k]
            Acc = (prediction == _labels).sum() / len(_labels)
            Oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

            # OSCR
            Aur, Fpr = evaluation.compute_auroc(_pred_k, _pred_u, _labels)
    results = {}
    results['ACC'] = Acc * 100
    results['OSCR'] = Oscr_socre * 100.
    results['FPR95'] = Fpr * 100.
    results['AUROC'] = Aur * 100

    return results



def test_att(net, criterion, testloader, outloader, use_softmax,use_h, **options):
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
    if use_h == True:
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

    bestscore = -99
    if use_softmax == True:
        betarange = range(0, 100, 1)
        betarange = [item/20 for item in betarange]
        cls_logit_k = softmax(cls_logit_k)
        cls_logit_u = softmax(cls_logit_u)
    else:
        betarange = range(0, 100, 1)

    if use_h == True:
        tmpk = softmax(cls_logit_k)+att_logit_k
        tmpu = softmax(cls_logit_u) + att_logit_u
        cls_pred_k = [np.argmax(item) for item in tmpk]
        cls_pred_u = [np.argmax(item) for item in tmpu]
        super_cls_pred_k = [cls2super[item] for item in cls_pred_k]
        super_cls_pred_uk = [cls2super[item] for item in cls_pred_u]


        att_logit_k = get_attscore_hire(_att_k, super_prototype.copy(),super_cls_pred_k,super_mask,hyre_dict_idx,cls_logit_k.shape[1])
        att_logit_u = get_attscore_hire(_att_u, super_prototype.copy(), super_cls_pred_uk, super_mask, hyre_dict_idx,
                                        cls_logit_k.shape[1])

        # for i in range(len(_att_k)):
        #         image = testloader.dataset.data[i]
        #         att = _att_k[i]
        #         sorted_indexes = sorted(range(len(att)), key=lambda i: att[i], reverse=True)[:7]
        #
        #         print("已知类图像：", image, "最高的7个的属性：", sorted_indexes, "粗类标签：", super_cls_pred_k[i], "logits",
        #               print(np.max(att_logit_k[i])))
        #
        # for i in range(len(_att_u)):
        #         image = outloader.dataset.data[i]
        #         att = _att_k[i]
        #         sorted_indexes = sorted(range(len(att)), key=lambda i: att[i], reverse=True)[:7]
        #
        #         print("未知类图像：", image, "最高的7个的属性：", sorted_indexes, "粗类标签：", super_cls_pred_uk[i], "logits",
        #               print(np.max(att_logit_u[i])))

    aurlist = []
    oscrlist = []
    betalist = []
    betarange = range(0, 100, 1)
    for beta in betarange:  # in_score + 1.5*
        beta = beta/100
        _pred_k = cls_logit_k + beta * att_logit_k
        _pred_u = cls_logit_u + beta * att_logit_u

        prediction = [np.argmax(item) for item in _pred_k]
        acc = (prediction == _labels).sum() / len(_labels)
        _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)


        # OSCR
        aur, fpr = evaluation.compute_auroc(_pred_k, _pred_u, _labels)
        aurlist.append(aur)
        oscrlist.append(_oscr_socre)
        betalist.append(beta)
        if aur > bestscore:
            Aur = aur
            Acc = acc
            Oscr_socre = _oscr_socre
            Fpr = fpr
            bestbeta = beta
            bestscore = aur

            # sns.set_theme(style="white", font_scale=1.5)
            #
            # fig = plt.figure(figsize=(7, 6))
            # sns.distplot(np.amax(_pred_u, axis=1), bins=100, hist=False, kde=True, norm_hist=False,
            #              color='#1E90FF', label='Unknown Class', kde_kws={
            #         'shade': True,  # 开启填充
            #     })
            #
            # sns.distplot(np.amax(_pred_k, axis=1), bins=100, hist=False, kde=True, norm_hist=False,
            #              color='#2E8B57', label='Known Class', kde_kws={
            #         'shade': True,  # 开启填充
            #     })
            #
            # plt.legend(loc='upper left')
            # plt.savefig('all.pdf', bbox_inches='tight')
        if beta ==0 :
            print('clshead:','ACC:',acc * 100, 'OSCR:', _oscr_socre * 100, 'FPR95:' ,fpr * 100, 'AUROC:', aur*100)
        if beta == 1.0:
            print('mix:', 'ACC:', acc * 100, 'OSCR:', _oscr_socre * 100, 'FPR95:', fpr * 100, 'AUROC:',
                  aur * 100)
            # sns.set_theme(style="white", font_scale=1.5)
            #
            # fig = plt.figure(figsize=(7, 6))
            # sns.distplot(np.amax(_pred_u, axis=1), bins=100, hist=False, kde=True, norm_hist=False,
            #              color='#1E90FF', label='Unknown Class', kde_kws={
            #         'shade': True,  # 开启填充
            #     })
            #
            # sns.distplot(np.amax(_pred_k, axis=1), bins=100, hist=False, kde=True, norm_hist=False,
            #              color='#2E8B57', label='Known Class', kde_kws={
            #         'shade': True,  # 开启填充
            #     })
            #
            # plt.legend(loc='upper left')
            # plt.savefig('cls.pdf', bbox_inches='tight')
    # print(aurlist,oscrlist,betalist)
    _pred_k =  att_logit_k
    _pred_u =  att_logit_u
    prediction = [np.argmax(item) for item in _pred_k]
    acc = (prediction == _labels).sum() / len(_labels)
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)
    # OSCR
    aur, fpr = evaluation.compute_auroc(_pred_k, _pred_u, _labels)
    print('atthead:', 'ACC:', acc * 100, 'OSCR:', _oscr_socre * 100, 'FPR95:', fpr * 100, 'AUROC:', aur * 100)

    # sns.set_theme(style="white", font_scale=1.5)
    #
    # fig = plt.figure(figsize=(7, 6))
    # sns.distplot(np.amax(_pred_u, axis=1), bins=100, hist=False, kde=True, norm_hist=False,
    #              color='#1E90FF', label='Unknown Class', kde_kws={
    #         'shade': True,  # 开启填充
    #     })
    #
    # sns.distplot(np.amax(_pred_k, axis=1), bins=100, hist=False, kde=True, norm_hist=False,
    #              color='#2E8B57', label='Known Class', kde_kws={
    #         'shade': True,  # 开启填充
    #     })
    #
    # plt.legend(loc='upper left')
    #
    #
    # if use_h == True:
    #     plt.savefig('att-h.pdf', bbox_inches='tight')
    # else:
    #     plt.savefig('att.pdf', bbox_inches='tight')
    results = {}
    results['ACC'] = Acc * 100
    results['OSCR'] = Oscr_socre * 100.
    results['FPR95'] = Fpr * 100.
    results['AUROC'] = Aur * 100
    print('mix: best_beta:', bestbeta)
    print(results)

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