import os
import argparse
import datetime
import time
import pandas as pd
import importlib

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn

from methods.MCAS.mcas_utils import save_networks
from methods.MCAS.core import train,test
from utils.utils import init_experiment, seed_torch, str2bool, get_default_hyperparameters
from utils.schedulers import get_scheduler
from data.open_set_datasets import get_class_splits, get_datasets
from models.model_utils import get_model

from config import exp_root

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='cub_ood', help="")
parser.add_argument('--type_choose', type=str, default='V', help="")
parser.add_argument('--use_attribute', type=str2bool, default=False, help="")

parser.add_argument('--use_hyre', type=str2bool, default=False, help="")

parser.add_argument('--use_default_attribute', type=str2bool, default=False, help="")
parser.add_argument('--att_file', type=str, default='cub_100_200_select150att.pkl', help="")
parser.add_argument('--use_attribute_only', type=str2bool, default=False, help="")

parser.add_argument('--image_size', type=int, default=448)

parser.add_argument('--att_loss_weight', type=float, default=0.5)
parser.add_argument('--att_choose_min', type=float, default=0.1)
parser.add_argument('--att_choose_max', type=float, default=0.9)
parser.add_argument('--att_loss', type=str, default='CEloss')
# optimization
parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0005, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts')
parser.add_argument('--temp', type=float, default=1.0, help="temp")

# model
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--label_smoothing', type=float, default=0.3, help="Smoothing constant for label smoothing."
                                                                        "No smoothing if None or 0")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='rs50')
parser.add_argument('--feat_dim', type=int, default=2048, help="Feature vector dim, only for classifier32 at the moment")

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=30)
parser.add_argument('--rand_aug_n', type=int, default=2)

# misc
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--split_train_val', default=False, type=str2bool,
                    help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--use_default_parameters', default=False, type=str2bool,
                    help='Set to True to use optimized hyper-parameters from paper', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                    help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--checkpt_freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--out-num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool,
                    help='Do we use softmax or logits for evaluation', metavar='BOOL')

def get_optimizer(args, params_list):
    if args.optim is None:

        if options['dataset'] == 'tinyimagenet':
            optimizer = torch.optim.Adam(params_list, lr=args.lr)
        else:
            optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    elif args.optim == 'sgd':

        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    elif args.optim == 'adam':

        optimizer = torch.optim.Adam(params_list, lr=args.lr)

    else:

        raise NotImplementedError

    return optimizer


def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()


# TODO: Args and options are largely duplicates: tidy up
def main_worker(options, args):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = False
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # -----------------------------
    # DATALOADERS
    # -----------------------------
    trainloader = dataloaders['train']
    testloader = dataloaders['val']
    valloader = dataloaders['test_known']
    outloader = dataloaders['test_unknown']

    # -----------------------------
    # MODEL
    # -----------------------------
    print("Creating model: {}".format(options['model']))
    wrapper_class = None
    net = get_model(args, wrapper_class=wrapper_class)

    feat_dim = args.feat_dim


    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu': use_gpu
        }
    )

    # -----------------------------
    # GET LOSS
    # -----------------------------
    Loss = importlib.import_module('methods.MCAS.loss.' + options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    # -----------------------------
    # PREPARE EXPERIMENT
    # -----------------------------
    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()


    model_path = os.path.join(args.log_dir, 'mcas_models', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    params_list = [{'params': net.parameters()},
                   {'params': criterion.parameters()}]

    # Get base network and criterion
    optimizer = get_optimizer(args=args, params_list=params_list)


    # -----------------------------
    # GET SCHEDULER
    # ----------------------------
    scheduler = get_scheduler(optimizer, args)

    start_time = time.time()

    Best_result = dict()
    Best_result['best_score'] = -99
    Best_result['best_score_h'] = -99
    Best_AUR_score = -99
    # -----------------------------
    # TRAIN
    # -----------------------------
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))


        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch + 1) % options['eval_freq'] == 0 or (epoch + 1) == options['max_epoch']:
            print("==> Test", options['loss'])

            # AUR_score = test_att_score(net, testloader, **options)

            if options['use_hyre'] == False:
                results = test(net, criterion, valloader, outloader, epoch=epoch, **options)
            else:
                results = test(net, criterion, valloader, outloader, epoch=epoch, **options)

            print(
                "Epoch {}: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t  FPR95 (%): {:.3f}\t  OSCR (%): {:.3f}\t".format(epoch,
                                                                                                                  results[
                                                                                                                      'ACC'],
                                                                                                                  results[
                                                                                                                      'AUROC'],
                                                                                                                  results[
                                                                                                                      'FPR95'],
                                                                                                                  results[
                                                                                                                      'OSCR']))

            OSR_score = results['AUROC']
            if OSR_score > Best_result['best_score']:
                Best_result['ACC'] = results['ACC']
                Best_result['AUROC'] = results['AUROC']
                Best_result['FPR95'] = results['FPR95']
                Best_result['OSCR'] = results['OSCR']
                Best_result['best_score'] = OSR_score

                save_networks(net, model_path, 'model_best',
                              options['loss'],
                              criterion=criterion)

 
            if epoch % options['checkpt_freq'] == 0 or epoch == options['max_epoch'] - 1:
                save_networks(net, model_path, file_name.split('.')[0] + '_{}'.format(epoch),
                              options['loss'],
                              criterion=criterion)

            # ----------------
            # LOG
            # ----------------
            args.writer.add_scalar('Test Acc Top 1', results['ACC'], epoch)
            args.writer.add_scalar('AUROC', results['AUROC'], epoch)

        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        # -----------------------------
        # STEP SCHEDULER
        # ----------------------------
        if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
            scheduler.step(results['ACC'], epoch)
        elif args.scheduler == 'multi_step':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return results, Best_result


if __name__ == '__main__':

    args = parser.parse_args()

    # ------------------------
    # Update parameters with default hyperparameters if specified
    # ------------------------
    if args.use_default_parameters:
        print('NOTE: Using default hyper-parameters...')
        args = get_default_hyperparameters(args)

    args.exp_root = exp_root
    args.epochs = args.max_epoch
    img_size = args.image_size
    results = dict()

    for i in range(1):

        # ------------------------
        # INIT
        # ------------------------
        if args.feat_dim is None:
            args.feat_dim =  2048

        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx,
                                                                     cifar_plus_n=args.out_num)

        img_size = args.image_size

        args.save_name = '{}_{}_{}_{}'.format(args.model, args.seed, args.dataset, args.split_idx)
        runner_name = os.path.dirname(__file__).split("/")[-2:]
        args = init_experiment(args, runner_name=runner_name)

        # ------------------------
        # SEED
        # ------------------------
        seed_torch(args.seed)

        # ------------------------
        # DATASETS
        # ------------------------
        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=False,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                                args=args)
        args.train_attributes = datasets['train'].att_num
        # ------------------------
        # RANDAUG HYPERPARAM SWEEP
        # ------------------------
        if args.transform == 'rand-augment':
            if args.rand_aug_m is not None:
                if args.rand_aug_n is not None:
                    datasets['train'].transform.transforms[0].m = args.rand_aug_m
                    datasets['train'].transform.transforms[0].n = args.rand_aug_n

        # ------------------------
        # DATALOADER
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=args.num_workers)

        # ------------------------
        # SAVE PARAMS
        # ------------------------
        options = vars(args)
        options.update(
            {
                'item': i,
                'known': args.train_classes,
                'unknown': args.open_set_classes,
                'img_size': img_size,
                'dataloaders': dataloaders,
                'num_classes': len(args.train_classes)
            }
        )

        dir_name = '{}_{}'.format(options['model'], options['loss'])
        dir_path = os.path.join('/'.join(args.log_dir.split("/")[:-2]), 'results', dir_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


        file_name = options['dataset'] + '.csv'

        print('result path:', os.path.join(dir_path, file_name))
        # ------------------------
        # TRAIN
        # ------------------------
        res, Best_result = main_worker(options, args)

        # ------------------------
        # LOG
        # ------------------------
        res['split_idx'] = args.split_idx
        res['unknown'] = args.open_set_classes
        res['known'] = args.train_classes
        res['ID'] = args.log_dir.split("/")[-1]
        for k, v in Best_result.items():
            k = 'best_' + k
            res[k] = str(v)
        results[str(args.split_idx)] = res
        df = pd.DataFrame(results)

        df.to_csv(os.path.join(dir_path, file_name), mode='a', header=False)


