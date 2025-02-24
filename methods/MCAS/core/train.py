import torch
import torch.nn.functional as F
from torch.autograd import Variable
from methods.MCAS.mcas_utils import AverageMeter
from methods.MCAS.loss.LabelSmoothing import smooth_one_hot
import numpy as np
from tqdm import tqdm
import torch.nn as nn

def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()

    if options['use_attribute'] == True:
        if options['att_loss'] =='Focal':
            att_criterion = FocalLoss()
        else:
            att_criterion = F.binary_cross_entropy_with_logits
    loss_all = 0
    for batch_idx, (data, labels, idx, att) in enumerate(tqdm(trainloader)):

        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()
            att = att.cuda()

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            if options['use_attribute'] == False:
                x, y = net(data, True)
                logits, loss_cls = criterion(x, y, labels)
                loss = loss_cls

            else:
                if options['use_attribute_only'] == False:
                    x, y, a = net(data,True)
                    logits, loss_cls = criterion(x, y, labels)
                    loss_att = att_criterion(a,  att.float())
                    loss = loss_cls  + loss_att*att.shape[1]*options['att_loss_weight']

                else:
                    x, y, a = net(data, True)
                    loss_att = att_criterion(a, att.float())
                    loss = loss_att * att.shape[1]*options['att_loss_weight']

            loss.backward()
            optimizer.step()
        
        losses.update(loss.item(), data.size(0))
        
        loss_all += losses.avg

    print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))

    return loss_all
