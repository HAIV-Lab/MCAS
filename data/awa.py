import numpy as np
from copy import deepcopy
from scipy import io as mat_io
import scipy.io
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from data.utils import get_super_class_targets
from config import awa_root
#from data.clustering import Cluster
import os
from sklearn.cluster import KMeans
class AWADataset(Dataset):
    def __init__(self, train=True, limit=0, data_dir=awa_root, transform=None):
        self.data_dir = data_dir
        label_file = os.path.join(self.data_dir, 'classes.txt')
        data_dir = data_dir.format('train') if train else data_dir.format('test')
        image_dir = os.path.join(self.data_dir, 'JPEGImages')
        class_labels = []
        with open(label_file, 'r') as f:
            for line in f:
                label = line.strip().split('\t')[1]
                class_labels.append(label)

        self.classes = class_labels
        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []

        for id, label in enumerate(class_labels):
            for filename in os.listdir(os.path.join(image_dir, label)):
                file_path = os.path.join(os.path.join(image_dir, label), filename)
                self.data.append(file_path)
                self.target.append(id)

        self.train = train
        self.transform = transform
        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx]
        att = np.array(self.attributes[idx])
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx, att

    def __len__(self):
        return len(self.data)


def subsample_dataset(dataset, idxs):

    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cars = np.array(include_classes)     # SCars classes are indexed 1 --> 196 instead of 0 --> 195
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i
    dataset.class_to_realname = {i: dataset.classes[i] for i in range(len(dataset.classes))}
    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]
    dataset.target_dict = target_xform_dict
    return dataset


def get_train_val_split(train_dataset, val_split=0.05):

    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

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

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2,)))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1,)))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2


def get_awa_datasets(train_transform, test_transform, train_classes=range(40),
                       open_set_classes=range(40, 50), balance_open_set_eval=True, split_train_val=True, seed=0, args=None):

    np.random.seed(seed)
    # Init train dataset and subsample training classes
    train_dataset_whole = AWADataset(data_dir=awa_root, transform=train_transform, train=True)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, test_dataset_known = get_train_val_split(train_dataset_whole,val_split=0.2)

    # train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_split,val_split=0.05)


    # Get testset for unknown classes
    test_dataset_whole = AWADataset(data_dir=awa_root, transform=test_transform, train=False)
    test_dataset_whole = subsample_classes(test_dataset_whole, include_classes=open_set_classes)
    _, test_dataset_unknown = get_train_val_split(test_dataset_whole,val_split=0.2)
    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split
    val_dataset =  test_dataset_known
    val_dataset.transform = test_transform
    test_dataset_known.transform = test_transform

    test_dataset_known.train_classes =[item for item in train_dataset_split.train_classes]
    class_to_realname = train_dataset.class_to_realname #{0:name...}
    realname_to_class = {v:k for k,v in class_to_realname.items()}  # {name:0...}
    known_class = [class_to_realname[item] for item in test_dataset_known.train_classes]
    test_dataset_known.known_class = known_class
    test_dataset_unknown
    awa_dataset = {
    'train_uqidx':train_dataset.uq_idxs,
    'train_data':train_dataset.data,
    'train_target':train_dataset.target,
    'test_k_uqidx':test_dataset_known.uq_idxs,
    'test_k_data': test_dataset_known.data,
    'test_k_target': test_dataset_known.target,
    'test_uk_uqidx':test_dataset_unknown.uq_idxs,
    'test_uk_data': test_dataset_unknown.data,
    'test_uk_target': test_dataset_unknown.target,
    }
    import pickle
    # with open('awa_dataset.pkl', 'wb') as f:
    #     pickle.dump(awa_dataset, f)
    with open('./pkl/awa_dataset.pkl', 'rb') as f:
        awa_data = pickle.load(f)
    train_dataset.uq_idxs = awa_data['train_uqidx']
    train_dataset.data = awa_data['train_data']
    train_dataset.target = awa_data['train_target']
    val_dataset.uq_idxs = awa_data['test_k_uqidx']
    val_dataset.data = awa_data['test_k_data']
    val_dataset.target = awa_data['test_k_target']
    test_dataset_known.uq_idxs = awa_data['test_k_uqidx']
    test_dataset_known.data = awa_data['test_k_data']
    test_dataset_known.target = awa_data['test_k_target']
    test_dataset_unknown.uq_idxs = awa_data['test_uk_uqidx']
    test_dataset_unknown.data = awa_data['test_uk_data']
    test_dataset_unknown.target = awa_data['test_uk_target']

    # use default attribute
    if args.use_default_attribute == True:
        attributes = np.loadtxt("./AWA/predicate-matrix-binary.txt", delimiter=' ', dtype=np.float32)
        attribute_matrix = {}
        for i in range(50):
            attribute_matrix[train_dataset.classes[i]] = attributes[i]

        k_att_dict = {label: attribute_matrix[label] for label in known_class}
        kmeans = KMeans(n_clusters=4)  
        kmeans.fit(list(k_att_dict.values()))
        cluster_labels1 = kmeans.labels_
        resultdict = {}

        for i in range(len(cluster_labels1)):
            idx = cluster_labels1[i]
            if (idx in resultdict) == False:
                resultdict[idx] = []
            resultdict[idx].append(list(k_att_dict.keys())[i])

        hyre_dict_name=resultdict
    else:
        import pickle
        # attribute_matrix, hyre_dict_name = Cluster('awa', known_class)
        with open("./json/"+args.att_file, 'rb') as f:
            data = pickle.load(f)

        attribute_matrix, hyre_dict_name = data['att_dict'],data['hyre_dict']

        # kmeans = KMeans(n_clusters=10)  
        # kmeans.fit(list(attribute_matrix.values()))
        # cluster_labels1 = kmeans.labels_
        # resultdict = {}
        #
        # for i in range(len(cluster_labels1)):
        #     idx = cluster_labels1[i]
        #     if (idx in resultdict) == False:
        #         resultdict[idx] = []
        #     resultdict[idx].append(list(attribute_matrix.keys())[i])
        #
        # hyre_dict_name = resultdict


    print(hyre_dict_name)
    hyre_dict_idx = {k: [train_dataset_split.target_dict[realname_to_class[item]] for item in v] for k, v in hyre_dict_name.items()}
    attribute_matrix = {k:v[int(args.att_choose_min*len(v)):int(args.att_choose_max*len(v))]  for k,v in attribute_matrix.items()}
    hyre_dict_name_index = {k: [attribute_matrix[item] for item in v] for k, v in hyre_dict_name.items()}
    args.train_attributes = int(len(list(attribute_matrix.values())[0]))
    super_mask = get_super_class_targets(hyre_dict_name_index,0.8)
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
        # tgt2raw ={v:k for k,v in dataset.target_dict.items()}

        try:
            dataset.targets = [item for item in dataset.target]
            dataset.attributes = [attribute_matrix[class_to_realname[item]] for item in dataset.targets]
        except:
            dataset.targets = [0 for item in dataset.target]
            dataset.attributes = [np.zeros(len((list(attribute_matrix.values())[0]))) for item in  dataset.targets]

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

    x = get_awa_datasets(None, None, split_train_val=False)

    val = x['val']
    for i in range(len(val.target)):
        if val.target[i] == 7:
            print(val.data[i])
    print([len(v) for k, v in x.items()])
    z = [np.unique(v.target) for k, v in x.items()]
    print(z[0])
    print(z[1])
    print(z[2])
    print(z[3])