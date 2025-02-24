from models.my_resnet import MyCustomRes50,MyCustomRes50_CLS,MyCustomRes18_CLS,MyCustomRes18

import torch
import argparse

from functools import partial

def get_model(args, wrapper_class=None, evaluate=False, *args_, **kwargs):


    if args.model == 'rs50':
        if args.use_attribute == False:
            model = MyCustomRes50_CLS(num_classes=len(args.train_classes))
        else:
            model = MyCustomRes50(num_classes=len(args.train_classes) ,num_att=args.train_attributes)

    elif args.model == 'rs18':
        if args.use_attribute == False:
            model = MyCustomRes18_CLS(num_classes=len(args.train_classes))
        else:
            model = MyCustomRes18(num_classes=len(args.train_classes) ,num_att=args.train_attributes)
    else:

        raise NotImplementedError

    if wrapper_class is not None:
        model = wrapper_class(model, *args_, **kwargs)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='cls',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    parser.add_argument('--model', default='timm_resnet50_pretrained', type=str)
    parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
    parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
    parser.add_argument('--loss', type=str, default='ARPLoss')
    args = parser.parse_args()

    args.train_classes = (0, 1, 8, 9)
    model = get_model(args)
    x, y = model(torch.randn(64, 3, 32, 32), True)
    debug = True