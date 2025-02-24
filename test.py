from test.utils import EvaluateOpenSet, ModelTemplate
from utils.utils import strip_state_dict
import importlib
import torch
import argparse
import numpy as np
import pickle
from methods.MCAS.core import train,test_att
from torch.utils.data import DataLoader
from data.open_set_datasets import get_datasets
from models.model_utils import get_model
import sys, os
import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import str2bool
from config import save_dir, osr_split_dir, root_model_path, root_criterion_path
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
from utils.utils import init_experiment, seed_torch, str2bool, get_default_hyperparameters
class EnsembleModelEntropy(ModelTemplate):

    def __init__(self, all_models, mode='entropy', num_classes=4, use_softmax=False):

        super(ModelTemplate, self).__init__()

        self.all_models = all_models
        self.max_ent = torch.log(torch.Tensor([num_classes])).item()
        self.mode = mode
        self.use_softmax = use_softmax

    def entropy(self, preds):

        logp = torch.log(preds + 1e-5)
        entropy = torch.sum(-preds * logp, dim=-1)

        return entropy

    def forward(self, imgs):

        all_closed_set_preds = []

        for m in self.all_models:

            closed_set_preds = m(imgs, return_features=False)

            if self.use_softmax:
                closed_set_preds = torch.nn.Softmax(dim=-1)(closed_set_preds)

            all_closed_set_preds.append(closed_set_preds)

        closed_set_preds = torch.stack(all_closed_set_preds).mean(dim=0)

        if self.mode == 'entropy':
            open_set_preds = self.entropy(closed_set_preds)
        elif self.mode == 'max_softmax':
            open_set_preds = -closed_set_preds.max(dim=-1)[0]

        else:
            raise NotImplementedError

        return closed_set_preds, open_set_preds

def load_models(path, args):

    if args.loss == 'ARPLoss':

        model = get_model(args, wrapper_class=None, evaluate=True)
        state_dict_list = [torch.load(p) for p in path]
        model.load_state_dict(state_dict_list)

    else:

        model = get_model(args, wrapper_class=None)
        state_dict = torch.load(path[0])
        state_dict = {k[k.index('.') + 1:]:v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
from sklearn.metrics import roc_auc_score
def enablePrint():
    sys.stdout = sys.__stdout__
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sampleing_atts(net, dataloader,scale=0.1, **options):
    _att_k =[]
    with torch.no_grad():
        for data, labels, idx, att in dataloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
                att = att.cuda()
                x, y, a = net(data, True)
                _att_k.append(a.data.cpu().numpy())
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
    return sorted_indices[0:int(len(sorted_indices)*scale)]
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='cls',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_id', type=str, default='AMPF_cub')
    parser.add_argument('--type_choose', type=str, default='A', help="")
    parser.add_argument('--dataset', type=str, default='cub')
    parser.add_argument('--att_file', type=str, default='cub_100_200_select150att.pkl')
    parser.add_argument('--att_choose_min', type=float, default=0.0)

    parser.add_argument('--att_choose_max', type=float, default=1.0)

    parser.add_argument('--use_attribute', type=str2bool, default=False, help="")
    parser.add_argument('--use_default_attribute', type=str2bool, default=False, help="")
    parser.add_argument('--use_hyre', type=str2bool, default=False, help="")
    parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool,
                        help='Do we use softmax or logits for evaluation', metavar='BOOL')
    parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
    # General
    parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
    parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
    parser.add_argument('--device', default='None', type=str, help='Which GPU to use')
    parser.add_argument('--osr_mode', default='max_softmax', type=str, help='{entropy, max_softmax}')
    parser.add_argument('--seed', default=0, type=int)





    parser.add_argument('--use_attribute_only', type=str, default=False, help="")


    # Model
    parser.add_argument('--model', type=str, default='rs50')
    parser.add_argument('--loss', type=str, default='Softmax')
    parser.add_argument('--feat_dim', default=2048, type=int)
    parser.add_argument('--max_epoch', default=599, type=int)
    parser.add_argument('--train_feat_extractor', default=True, type=str2bool,
                        help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
    parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
    parser.add_argument('--temp', type=float, default=1.0, help="temp")
    parser.add_argument('--label_smoothing', type=float, default=0.3, help="Smoothing constant for label smoothing."
                                                                           "No smoothing if None or 0")
    # Data params
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    #aircraft

    parser.add_argument('--transform', type=str, default='rand-augment')


    # Train params
    args = parser.parse_args()
    args.save_dir = save_dir
    args.use_supervised_places = False

    device = torch.device('cuda:0')

    assert args.exp_id is not None

    # Define experiment IDs
    exp_ids = [
        args.exp_id,
    ]

    # Define paths
    all_paths_combined = [[x.format(i, args.dataset, args.dataset, args.max_epoch, args.loss)
                           for x in (root_model_path, root_criterion_path)] for i in exp_ids]

    all_preds = []

    # Get OSR splits
    try:
        osr_path = os.path.join(osr_split_dir, '{}_osr_splits.pkl'.format(args.dataset))

        with open(osr_path, 'rb') as f:
            class_info = pickle.load(f)

        train_classes = class_info['known_classes']
        open_set_classes = class_info['unknown_classes']
    except:
        pass
    seed_torch(args.seed)
    if (args.dataset in ['awa','lad']) == True:
        difficultys = ['Easy']
    else:
        difficultys=['Easy', 'Hard']
    for difficulty in difficultys:

        # ------------------------
        # DATASETS
        # ------------------------

        try:
            args.train_classes, args.open_set_classes = train_classes, open_set_classes[difficulty]

            if difficulty == 'Hard' and args.dataset != 'imagenet':
                args.open_set_classes += open_set_classes['Medium']
        except:
            if args.dataset == 'awa':
                args.train_classes, args.open_set_classes = range(0,40),range(40,50)
            if args.dataset == 'lad':
                args.train_classes, args.open_set_classes = range(0,40),range(40,50)

        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                image_size=args.image_size, balance_open_set_eval=False,
                                split_train_val=False, open_set_classes=args.open_set_classes,seed=args.seed,
                                args=args)

        # ------------------------
        # DATALOADERS
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=args.num_workers)

        options = vars(args)
        options.update(
            {
                'known': args.train_classes,
                'unknown': args.open_set_classes,
                'dataloaders': dataloaders,
                'num_classes': len(args.train_classes)
            }
        )
        use_gpu = torch.cuda.is_available()
        options['use_gpu'] = use_gpu
        torch.manual_seed(options['seed'])
        use_gpu = torch.cuda.is_available()
        torch.cuda.manual_seed_all(options['seed'])

        # ------------------------
        # MODEL
        # ------------------------
        print('Loading model...')
        all_models = [load_models(path=all_paths_combined[0], args=args)]
        model = all_models[0]

        model = model.to(device)
        model.eval()

        # ------------------------
        # EVALUATE
        # ------------------------
        Loss = importlib.import_module('methods.MCAS.loss.' + options['loss'])
        criterion = getattr(Loss, options['loss'])(**options)
        criterion = criterion.cuda()
        if args.use_attribute == False:
            test_cls(model, criterion, dataloaders['test_known'], dataloaders['test_unknown'], epoch=600, **options)

        if args.use_attribute == True:

            att_mask = sampleing_atts(model, dataloaders['val'], 0.2, **options)
            options['attmask'] = att_mask
            print('*'*60,'mls','*'*60)
            print('*' * 60, 'mls+hyre', '*' * 60)

            print('*'*60,'msp','*'*60)
            test_att(model, criterion, dataloaders['test_known'], dataloaders['test_unknown'], True,False, **options)
            print('*'*60,'msp+hyre','*'*60)
            test_att(model, criterion, dataloaders['test_known'], dataloaders['test_unknown'], True,True, **options)

 
