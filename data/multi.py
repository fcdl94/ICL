from torch.utils.data import Dataset
import torchvision as tv
from torchvision.datasets import ImageFolder
from torchvision import transforms
import bisect
import torch


class MultiDataset(Dataset):
    """
    Code adapted from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataset.py#L42 (ConcatDataset)
    """
    def __init__(self, datasets):
        super().__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx][0], self.datasets[dataset_idx][sample_idx][1], torch.tensor(dataset_idx)

    def __len__(self):
        return self.cumulative_sizes[-1]


def office_home(ROOT, sources, target):
    paths = {"p": ROOT + "office/Product",
             "a": ROOT + "office/Art",
             "c": ROOT + "office/Clipart",
             "r": ROOT + "office/Real World"}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Normalize to have range between -1,1 : (x - 0.5) * 2
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])
    # Create data augmentation transform
    augmentation = transforms.Compose([transforms.RandomResizedCrop(224, (0.6, 1.)),
                                       transforms.RandomHorizontalFlip(),
                                       transform])

    sources_ = MultiDataset([ImageFolder(paths[char], augmentation) for char in sources])
    target_ = ImageFolder(paths[target], transform)
    return sources_, target_


def pacs(ROOT, sources, target):
    paths = {"ptr": ROOT + "pacs/train/photo", "pte": ROOT + "pacs/test/photo",
             "atr": ROOT + "pacs/train/art_painting", "ate": ROOT + "pacs/test/art_painting",
             "ctr": ROOT + "pacs/train/cartoon", "cte": ROOT + "pacs/test/cartoon",
             "str": ROOT + "pacs/train/sketch", "ste": ROOT + "pacs/test/sketch"}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Normalize to have range between -1,1 : (x - 0.5) * 2
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])
    # Create data augmentation transform
    augmentation = transforms.Compose([transforms.RandomResizedCrop(224, (0.6, 1.)),
                                       transforms.RandomHorizontalFlip(),
                                       transform])

    sources_ = MultiDataset([ImageFolder(paths[char+"tr"], augmentation) for char in sources])
    target_ = ImageFolder(paths[target+"te"], transform)
    return sources_, target_