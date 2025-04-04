from data.cub import get_cub_datasets
from data.stanford_cars import get_scars_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.awa import get_awa_datasets
from data.open_set_splits.osr_splits import osr_splits
from data.augmentations import get_transform
from data.lad import get_lad_datasets
from config import osr_split_dir

import os
import sys
import pickle
import torch

"""
For each dataset, define function which returns:
    training set
    validation set
    open_set_known_images
    open_set_unknown_images
"""

get_dataset_funcs = {
    'cub': get_cub_datasets,
    'scars': get_scars_datasets,
    'aircraft': get_aircraft_datasets,
    'awa': get_awa_datasets,
    'lad': get_lad_datasets,
}

def get_datasets(name, transform='default', image_size=224, train_classes=(0, 1, 8, 9),
                 open_set_classes=range(10), balance_open_set_eval=False, split_train_val=True, seed=0, args=None):

    """
    :param name: Dataset name
    :param transform: Either tuple of train/test transforms or string of transform type
    :return:
    """

    print('Loading datasets...')

    if isinstance(transform, tuple):
        train_transform, test_transform = transform
    else:
        train_transform, test_transform = get_transform(transform_type=transform, image_size=image_size, args=args)

    if name in get_dataset_funcs.keys():
        datasets = get_dataset_funcs[name](train_transform, test_transform,
                                  train_classes=train_classes,
                                  open_set_classes=open_set_classes,
                                  balance_open_set_eval=balance_open_set_eval,
                                  split_train_val=split_train_val,
                                  seed=seed,
                                  args=args)

    else:
        raise NotImplementedError

    return datasets

def get_class_splits(dataset, split_idx=0, cifar_plus_n=10):


    if dataset == 'cub':

        osr_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
        with open(osr_path, 'rb') as f:
            class_info = pickle.load(f)

        train_classes = class_info['known_classes']

        open_set_classes = class_info['unknown_classes']
        open_set_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

    elif dataset == 'aircraft':

        osr_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
        with open(osr_path, 'rb') as f:
            class_info = pickle.load(f)

        train_classes = class_info['known_classes']

        open_set_classes = class_info['unknown_classes']
        open_set_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']


    elif dataset == 'scars':

        osr_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
        with open(osr_path, 'rb') as f:
            class_info = pickle.load(f)

        train_classes = class_info['known_classes']

        open_set_classes = class_info['unknown_classes']
        open_set_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

    elif dataset == 'awa':
        train_classes = list(range(40))
        open_set_classes = list(range(40,50))

    elif dataset == 'lad':
        train_classes = list(range(40))
        open_set_classes = list(range(40,50))

    else:

        raise NotImplementedError

    return train_classes, open_set_classes

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__