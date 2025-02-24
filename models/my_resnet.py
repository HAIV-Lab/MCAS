import torch.nn as nn
import timm
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models.resnet_changed import resnet50,resnet101,resnet18,resnet34,resnet152



class Attribute_Head(nn.Module):
    def __init__(
            self,
            att_num,
            input_fea,
            **kwargs
    ):
        super(Attribute_Head, self).__init__()

        self.att_num = att_num
        self.last_conv_stride = 2
        self.drop_pool5 = True
        self.drop_pool5_rate = 0.5
        self.pretrained = True
        self.classifier1 = nn.Linear(input_fea, 512)
        self.classifier2 = nn.Linear(512, self.att_num)

        init.normal(self.classifier1.weight, std=0.001)
        init.constant(self.classifier1.bias, 0)
        init.normal(self.classifier2.weight, std=0.001)
        init.constant(self.classifier2.bias, 0)

    def forward(self, feature): #x为 896
        attpre = self.classifier1(feature)
        attpre = self.classifier2(attpre)
        return attpre


class MyCustomRes50(nn.Module):
    def __init__(
            self,
            num_classes=100,
            num_att=86,
            **kwargs
    ):
        super(MyCustomRes50, self).__init__()

        # init the necessary parameter for netwokr structure
        self.num_att1 = num_att
        self.last_conv_stride = 2
        self.drop_pool5 = True
        self.drop_pool5_rate = 0.5
        self.pretrained = True
        self.base = resnet50(pretrained=self.pretrained, last_conv_stride=self.last_conv_stride)
        self.atthead1 = Attribute_Head(att_num=num_att,input_fea=2048)
        self.linear2 = nn.Linear(2048, 512)
        self.classifier2 = nn.Linear(512, num_classes)
    def forward(self, x, return_fea):
        [x1,x2,x3,x4] = self.base(x)
        x4 = F.avg_pool2d(x4, x4.shape[2:])
        x4 = x4.view(x4.size(0), -1)
        if self.drop_pool5:
            feature = F.dropout(x4, p=self.drop_pool5_rate, training=self.training)
        att1 = self.atthead1(feature)
        clsfeature = self.linear2(feature)
        cls = self.classifier2(clsfeature)

        return x4,cls,att1


class MyCustomRes18_CLS(nn.Module):
    def __init__(
            self,
            num_classes=100,
            num_att=86,
            **kwargs
    ):
        super(MyCustomRes18_CLS, self).__init__()

        # init the necessary parameter for netwokr structure
        self.num_att1 = num_att
        self.last_conv_stride = 2
        self.drop_pool5 = True
        self.drop_pool5_rate = 0.5
        self.pretrained = True
        self.base = resnet18(pretrained=self.pretrained, last_conv_stride=self.last_conv_stride)

        # self.linear2 = nn.Linear(2048, 512)
        self.classifier2 = nn.Linear(512, num_classes)
    def forward(self, x, return_fea):
        [x1,x2,x3,x4] = self.base(x)
        x4 = F.avg_pool2d(x4, x4.shape[2:])
        x4 = x4.view(x4.size(0), -1)
        if self.drop_pool5:
            feature = F.dropout(x4, p=self.drop_pool5_rate, training=self.training)

        # clsfeature = self.linear2(feature)
        cls = self.classifier2(feature)

        return x4,cls

class MyCustomRes18(nn.Module):
    def __init__(
            self,
            num_classes=100,
            num_att=86,
            **kwargs
    ):
        super(MyCustomRes18, self).__init__()

        # init the necessary parameter for netwokr structure
        self.num_att1 = num_att
        self.last_conv_stride = 2
        self.drop_pool5 = True
        self.drop_pool5_rate = 0.5
        self.pretrained = True
        self.base = resnet18(pretrained=self.pretrained, last_conv_stride=self.last_conv_stride)
        self.atthead1 = Attribute_Head(att_num=num_att,input_fea=512)
        self.linear2 = nn.Linear(512, 512)
        self.classifier2 = nn.Linear(512, num_classes)

    def forward(self, x, return_fea):
        [x1, x2, x3, x4] = self.base(x)
        x4 = F.avg_pool2d(x4, x4.shape[2:])
        x4 = x4.view(x4.size(0), -1)
        if self.drop_pool5:
            feature = F.dropout(x4, p=self.drop_pool5_rate, training=self.training)
        att1 = self.atthead1(feature)
        clsfeature = self.linear2(feature)
        cls = self.classifier2(clsfeature)

        return x4, cls, att1

class MyCustomRes50_CLS(nn.Module):
    def __init__(
            self,
            num_classes=100,
            num_att=86,
            **kwargs
    ):
        super(MyCustomRes50_CLS, self).__init__()

        # init the necessary parameter for netwokr structure
        self.num_att1 = num_att
        self.last_conv_stride = 2
        self.drop_pool5 = True
        self.drop_pool5_rate = 0.5
        self.pretrained = True
        self.base = resnet50(pretrained=self.pretrained, last_conv_stride=self.last_conv_stride)

        # self.linear2 = nn.Linear(2048, 512)
        self.classifier2 = nn.Linear(2048, num_classes)

    def forward(self, x, return_fea):
        [x1, x2, x3, x4] = self.base(x)
        x4 = F.avg_pool2d(x4, x4.shape[2:])
        x4 = x4.view(x4.size(0), -1)
        if self.drop_pool5:
            feature = F.dropout(x4, p=self.drop_pool5_rate, training=self.training)

        # clsfeature = self.linear2(feature)
        cls = self.classifier2(feature)

        return x4, cls