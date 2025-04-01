import os
import numpy as np
from copy import deepcopy
from data.utils import get_super_class_targets
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from config import aircraft_root
#from data.clustering import Cluster
import pickle
from sklearn.cluster import KMeans
def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


def find_classes(classes_file):

    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


class FGVCAircraft(Dataset):

    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft>`_ Dataset.

    Args:
        root (string): Root directory path to dataset.
        class_type (string, optional): The level of FGVC-Aircraft fine-grain classification
            to label data with (i.e., ``variant``, ``family``, or ``manufacturer``).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, root, class_type='variant', split='train', transform=None,
                 target_transform=None, loader=default_loader, download=True, hyre=False):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))
        self.root = os.path.expanduser(root)
        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = find_classes(self.classes_file)
        samples = make_dataset(self.root, image_ids, targets)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = True if split == 'train' else False

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        att = np.array(self.attributes[index])
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.uq_idxs[index],att

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'data', 'images')) and \
            os.path.exists(self.classes_file)

    def download(self):
        """Download the FGVC-Aircraft data if it doesn't exist already."""
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s ... (may take a few minutes)' % self.url)
        parent_dir = os.path.abspath(os.path.join(self.root, os.pardir))
        tar_name = self.url.rpartition('/')[-1]
        tar_path = os.path.join(parent_dir, tar_name)
        data = urllib.request.urlopen(self.url)

        # download .tar.gz file
        with open(tar_path, 'wb') as f:
            f.write(data.read())

        # extract .tar.gz to PARENT_DIR/fgvc-aircraft-2013b
        data_folder = tar_path.strip('.tar.gz')
        print('Extracting %s to %s ... (may take a few minutes)' % (tar_path, data_folder))
        tar = tarfile.open(tar_path)
        tar.extractall(parent_dir)

        # if necessary, rename data folder to self.root
        if not os.path.samefile(data_folder, self.root):
            print('Renaming %s to %s ...' % (data_folder, self.root))
            os.rename(data_folder, self.root)

        # delete .tar.gz file
        print('Deleting %s ...' % tar_path)
        os.remove(tar_path)

        print('Done!')


def subsample_dataset(dataset, idxs):

    dataset.samples = [(p, t) for i, (p, t) in enumerate(dataset.samples) if i in idxs]
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(60)):

    cls_idxs = [i for i, (p, t) in enumerate(dataset.samples) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    dataset.class_to_realname = {i:dataset.classes[i][0:-1] for i in range(len(dataset.classes))}
    dataset.target_dict = target_xform_dict
    return dataset


def get_train_val_split(train_dataset, val_split=0.05):

    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    all_targets = [t for i, (p, t) in enumerate(train_dataset.samples)]
    train_classes = np.unique(all_targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(all_targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    # Get training/validation datasets based on selected idxs
    train_dataset = subsample_dataset(train_dataset, train_idxs)
    val_dataset = subsample_dataset(val_dataset, val_idxs)
    # add
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

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2,)), replace=False)
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2


def get_aircraft_datasets(train_transform, test_transform, train_classes=range(60),
                       open_set_classes=range(60, 100), balance_open_set_eval=False, split_train_val=True, seed=0, args=None):

    np.random.seed(seed)
    # split_train_val = True
    # Init train dataset and subsample training classes
    train_dataset_whole = FGVCAircraft(root=aircraft_root, transform=train_transform, split='trainval', hyre=args.use_hyre)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = FGVCAircraft(root=aircraft_root, transform=test_transform, split='test', hyre=args.use_hyre)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = FGVCAircraft(root=aircraft_root, transform=test_transform, split='test', hyre=args.use_hyre)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    test_dataset_known.train_classes =[item for item in train_dataset_split.train_classes]
    class_to_realname = train_dataset.class_to_realname #{0:name...}
    realname_to_class = {v:k for k,v in class_to_realname.items()}  # {name:0...}
    known_class = [class_to_realname[item] for item in test_dataset_known.train_classes]
    test_dataset_known.known_class = known_class

    # attribute_matrix, hyre_dict_name = Cluster('fgvc', known_class)

    with open("./json/"+args.att_file, 'rb') as f:
        data = pickle.load(f)

    attribute_matrix, hyre_dict_name = data['att_dict'],data['hyre_dict']
    

    k_att_dict = {label: attribute_matrix[label] for label in known_class}
    kmeans = KMeans(n_clusters=10)  
    kmeans.fit(list(k_att_dict.values()))
    cluster_labels1 = kmeans.labels_
    resultdict = {}

    for i in range(len(cluster_labels1)):
        idx = cluster_labels1[i]
        if (idx in resultdict) == False:
            resultdict[idx] = []
        resultdict[idx].append(list(k_att_dict.keys())[i])

    hyre_dict_name = resultdict

    print(hyre_dict_name)
    hyre_dict_idx = {k: [train_dataset_split.target_dict[realname_to_class[item]] for item in v] for k, v in hyre_dict_name.items()}
    attribute_matrix = {k:v[int(args.att_choose_min*len(v)):int(args.att_choose_max*len(v))]  for k,v in attribute_matrix.items()}
    hyre_dict_name_index = {k: [attribute_matrix[item] for item in v] for k, v in hyre_dict_name.items()}
    args.train_attributes = int(len(list(attribute_matrix.values())[0]))
    super_mask = get_super_class_targets(hyre_dict_name_index,0.5)
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

        try:
            dataset.targets = [item[1] for item in dataset.samples]
            dataset.attributes = [attribute_matrix[class_to_realname[item]] for item in dataset.targets]
        except:
            dataset.targets = [0 for item in dataset.samples]
            dataset.attributes = [np.zeros((attribute_matrix.values()[0])) for item in  dataset.targets]

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

    x = get_aircraft_datasets(None, None, split_train_val=False)
    print(x)
