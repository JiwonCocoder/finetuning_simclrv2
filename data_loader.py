import torch
import numpy as np
import pdb
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils_img import plot_images

imsize_dict = {'CIFAR10': 32, 'STL10': 96}
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)

MLCC_mean = (0.1778, 0.04714, 0.16583)
MLCC_std = (0.26870, 0.1002249, 0.273526)

dataset_stats = {
    'CIFAR10': {
        'mean': cifar10_mean,
        'std': cifar10_std
    },
    'STL10': {
        'mean': stl10_mean,
        'std': stl10_std
    }
}
def get_transforms(dataset, dataset_resolution=-1, mode = None):  #mode == train, val, test
    if mode == 'train':
        if dataset_resolution == -1:
            if dataset == 'STL10':
                transform = transforms.Compose([
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_stats[dataset]['mean'], dataset_stats[dataset]['std']),
                ])

            elif dataset == 'CIFAR10':
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_stats[dataset]['mean'], dataset_stats[dataset]['std'])
                ])
        else:
            if dataset == 'STL10':
                transform = transforms.Compose([
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(dataset_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(stl10_mean, stl10_std),
                ])

            elif dataset == 'CIFAR10':
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(dataset_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(stl10_mean, stl10_std),
                ])
    else: #val & test
        if dataset_resolution == -1:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'], dataset_stats[dataset]['std'])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(dataset_resolution),
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'], dataset_stats[dataset]['std'])
            ])
    return transform

def get_train_valid_loader(data_dir,
                           dataset,
                           batch_size,
                           limit_data,
                           num_labels,
                           num_classes,
                           augment,
                           random_seed,
                           dataset_resolution=-1,
                           valid_size = 0.1,
                           shuffle = True,
                           show_sample = False,
                           num_workers = 4,
                           pin_memory = False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """


    # get_transforms(dataset)
    # load the dataset
    if dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=get_transforms(dataset, dataset_resolution, mode = 'train')
        )

        valid_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=get_transforms(dataset, dataset_resolution, mode = 'val'),
        )
    elif dataset =='STL10':
        train_dataset = datasets.STL10(
            root = data_dir, split = 'train',
            download = True, transform = get_transforms(dataset, dataset_resolution, mode = 'train')
        )
        valid_dataset = datasets.STL10(
            root=data_dir, split = 'train',
            download=True, transform=get_transforms(dataset, dataset_resolution, mode = 'val'),
        )


    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if limit_data == True:
        print("only " + str(num_labels) + "data")
        if dataset == 'CIFAR10':
            data, targets = np.array(train_dataset.data), np.array(train_dataset.targets)
        elif dataset == 'STL10':
            data, targets = np.array(train_dataset.data), np.array(train_dataset.labels)
            print(len(targets))
        samples_per_class = int(num_labels / num_classes)

        lb_data = []
        lbs = []
        lb_idx = []
        for c in range(num_classes):
            idx = np.where(targets == c)[0]
            idx = np.random.choice(idx, samples_per_class, False)
            lb_idx.extend(idx)

            lb_data.extend(data[idx])
            lbs.extend(targets[idx])
        train_idx = lb_idx
        indices_for_valid = [item for item in indices if item not in train_idx ]
        valid_idx = indices_for_valid[:split]
    else:
        train_idx, valid_idx = indices[split:], indices[:split]



    #To check limit_data & ALL_data is well-defined 
    print("train_dataset_len:", str(len(train_idx)), "// val_dataset_len:", str(len(valid_idx)))
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    dataset,
                    batch_size = 1,
                    dataset_resolution = False,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    if dataset == 'CIFAR10':
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=get_transforms(dataset, dataset_resolution, mode = 'test'),
        )
    elif dataset == 'STL10':
        dataset = datasets.STL10(
            root=data_dir, split = 'test',
            download = True, transform = get_transforms(dataset, dataset_resolution, mode = 'test')
        )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
