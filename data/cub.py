import os
import pandas as pd
import numpy as np
from copy import deepcopy
#from data.clustering import Cluster
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from config import cub_root
from data.utils import get_super_class_targets
import pickle

class CustomCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=False, hyre=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.hyre = hyre

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        att = np.array(self.attributes[idx])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # target = int(self.fine2super[target])
        return img, target, self.uq_idxs[idx], att


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]


    idx2name = {}
    with open("./CUB_200_2011/CUB_200_2011/classes.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            tmp = line.split(' ')
            idx2name[int(tmp[0])-1] = tmp[1].split('.')[1][0:-1]

    dataset.class_to_realname = idx2name
    dataset.target_dict= target_xform_dict

    return dataset


def get_train_val_split(train_dataset, val_split=0.05):

    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    # Get training/validation datasets based on selected idxs
    train_dataset = subsample_dataset(train_dataset, train_idxs)
    val_dataset = subsample_dataset(val_dataset, val_idxs)
    train_dataset.train_classes = train_classes
    val_dataset.train_classes = train_classes
    return train_dataset, val_dataset


def get_equal_len_datasets(dataset1, dataset2):

    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2,)), replace=False)
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1,)), replace=False)
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2



def get_cub_datasets(train_transform, test_transform, train_classes=range(160),
                       open_set_classes=range(160, 200), balance_open_set_eval=False, split_train_val=True, seed=0, args=None):

    np.random.seed(seed)
    # split_train_val = True
    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCub2011(root=cub_root, transform=train_transform, train=True, hyre=args.use_hyre)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = CustomCub2011(root=cub_root, transform=test_transform, train=False, hyre=args.use_hyre)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCub2011(root=cub_root, transform=test_transform, train=False, hyre=args.use_hyre)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known


    # allclassname = train_dataset.classes
    test_dataset_known.train_classes =[item-1 for item in train_dataset_split.train_classes]
    class_to_realname = train_dataset.class_to_realname #{0:name...}
    realname_to_class = {v:k for k,v in class_to_realname.items()}  # {name:0...}
    known_class = [class_to_realname[item] for item in test_dataset_known.train_classes]
    test_dataset_known.known_class = known_class
    # attribute_matrix, hyre_dict_name = Cluster('cub', known_class)
    with open("./json/"+args.att_file, 'rb') as f:
        data = pickle.load(f)
    from sklearn.cluster import KMeans
    attribute_matrix, hyre_dict_name = data['att_dict'],data['hyre_dict']
    attribute_matrix = {label: attribute_matrix[label] for label in known_class}
    kmeans = KMeans(n_clusters=5)  
    kmeans.fit(list(attribute_matrix.values()))
    cluster_labels1 = kmeans.labels_
    resultdict = {}

    for i in range(len(cluster_labels1)):
        idx = cluster_labels1[i]
        if (idx in resultdict) == False:
            resultdict[idx] = []
        resultdict[idx].append(list(attribute_matrix.keys())[i])
    hyre_dict_name = resultdict

    hyre_dict_idx = {k: [train_dataset_split.target_dict[realname_to_class[item]] for item in v] for k, v in hyre_dict_name.items()}
    attribute_matrix = {k:v[int(args.att_choose_min*len(v)):int(args.att_choose_max*len(v))]  for k,v in attribute_matrix.items()}
    hyre_dict_name_index = {k: [attribute_matrix[item] for item in v] for k, v in hyre_dict_name.items()}
    args.train_attributes = int(len(list(attribute_matrix.values())[0]))
    super_mask = get_super_class_targets(hyre_dict_name_index,0.75)
    fine2super =  {}
    for k,v in hyre_dict_idx.items():
        for item in v:
            fine2super[item] = k
    known_prototype=  [attribute_matrix[class_to_realname[k]] for k, v in
    test_dataset_known.target_dict.items()]

    def add_dataset_infomation(dataset, **kwargs):
        dataset.super_mask = kwargs['super_mask']
        dataset.hyre_dict_name_index = kwargs['hyre_dict_name_index']
        dataset.hyre_dict_idx =  kwargs['hyre_dict_idx']
        dataset.fine2super = kwargs['fine2super']
        dataset.known_prototype = kwargs['known_prototype']
        dataset.attribute_matrix = kwargs['attribute_matrix']
        dataset.targets = [item.target - 1 for item in dataset.data.iloc]
        try:
            dataset.attributes = [attribute_matrix[class_to_realname[item]] for item in dataset.targets]
        except:
            dataset.attributes = [np.zeros((len(test_dataset_known.attributes[0]))) for item in
                                           dataset.targets]
        dataset.att_num = len(dataset.attributes[0])
        return dataset


    train_dataset = add_dataset_infomation(train_dataset,super_mask=super_mask,hyre_dict_name_index=hyre_dict_name_index
                                           ,hyre_dict_idx=hyre_dict_idx,fine2super=fine2super,known_prototype=known_prototype,
                                           attribute_matrix=attribute_matrix)

    val_dataset = add_dataset_infomation(val_dataset,super_mask=super_mask,hyre_dict_name_index=hyre_dict_name_index
                                           ,hyre_dict_idx=hyre_dict_idx,fine2super=fine2super,known_prototype=known_prototype,
                                           attribute_matrix=attribute_matrix)
    test_dataset_known = add_dataset_infomation(test_dataset_known,super_mask=super_mask,hyre_dict_name_index=hyre_dict_name_index
                                           ,hyre_dict_idx=hyre_dict_idx,fine2super=fine2super,known_prototype=known_prototype,
                                           attribute_matrix=attribute_matrix)
    test_dataset_unknown = add_dataset_infomation(test_dataset_unknown,super_mask=super_mask,hyre_dict_name_index=hyre_dict_name_index
                                           ,hyre_dict_idx=hyre_dict_idx,fine2super=fine2super,known_prototype=known_prototype,
                                           attribute_matrix=attribute_matrix)


    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }
    return all_datasets


if __name__ == '__main__':

    x = get_cub_datasets(None, None, split_train_val=False, train_classes=np.random.choice(range(200), size=100, replace=False))
    print([len(v) for k, v in x.items()])
    # z = x['train'][0]
    # debug = 0